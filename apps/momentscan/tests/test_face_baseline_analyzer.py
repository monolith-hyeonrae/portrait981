"""Tests for FaceBaselineAnalyzer."""

import math

import pytest
from unittest.mock import MagicMock

from momentscan.algorithm.analyzers.face_baseline import (
    FaceBaselineAnalyzer,
    FaceBaselineProfile,
    FaceBaselineOutput,
)
from vpx.sdk import Observation
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput
from momentscan.algorithm.analyzers.face_classifier.output import FaceClassifierOutput
from momentscan.algorithm.analyzers.face_classifier.types import ClassifiedFace


def _make_face(face_id: int, area_ratio: float, cx: float = 0.5, cy: float = 0.5) -> FaceObservation:
    """Create a FaceObservation with specified center and area."""
    w = h = math.sqrt(area_ratio)
    x = cx - w / 2
    y = cy - h / 2
    return FaceObservation(
        face_id=face_id,
        confidence=0.9,
        bbox=(x, y, w, h),
        inside_frame=True,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        area_ratio=area_ratio,
        center_distance=abs(cx - 0.5),
        signals={},
    )


def _make_frame(frame_id: int = 1):
    frame = MagicMock()
    frame.frame_id = frame_id
    frame.t_src_ns = frame_id * 100_000_000
    return frame


def _classify_obs(classified_faces, frame_id: int = 1) -> Observation:
    main = None
    passengers = []
    for cf in classified_faces:
        if cf.role == "main":
            main = cf
        elif cf.role == "passenger":
            passengers.append(cf)
    return Observation(
        source="face.classify",
        frame_id=frame_id,
        t_ns=frame_id * 100_000_000,
        signals={},
        data=FaceClassifierOutput(
            faces=classified_faces,
            main_face=main,
            passenger_faces=passengers,
        ),
    )


def _face_obs(faces, frame_id: int = 1) -> Observation:
    return Observation(
        source="face.detect",
        frame_id=frame_id,
        t_ns=frame_id * 100_000_000,
        signals={},
        data=FaceDetectOutput(faces=faces, detected_faces=[], image_size=(640, 480)),
    )


class TestFaceBaselineAnalyzer:

    @pytest.fixture
    def analyzer(self):
        a = FaceBaselineAnalyzer()
        a.initialize()
        return a

    def test_name_and_depends(self, analyzer):
        assert analyzer.name == "face.baseline"
        assert "face.detect" in analyzer.depends
        assert "face.classify" in analyzer.depends

    def test_no_profile_with_single_frame(self, analyzer):
        """n < 2 should produce no profiles."""
        face = _make_face(1, 0.05)
        cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=5, avg_area=0.05)
        deps = {
            "face.detect": _face_obs([face]),
            "face.classify": _classify_obs([cf]),
        }
        obs = analyzer.process(_make_frame(1), deps=deps)

        assert obs is not None
        assert obs.data.main_profile is None
        assert obs.data.profiles == []

    def test_profile_after_two_frames(self, analyzer):
        """n >= 2 should produce a valid profile."""
        for i in range(1, 3):
            face = _make_face(1, 0.05, cx=0.5, cy=0.5)
            cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=i, avg_area=0.05)
            deps = {
                "face.detect": _face_obs([face], frame_id=i),
                "face.classify": _classify_obs([cf], frame_id=i),
            }
            obs = analyzer.process(_make_frame(i), deps=deps)

        assert obs.data.main_profile is not None
        p = obs.data.main_profile
        assert p.face_id == 1
        assert p.role == "main"
        assert p.n == 2
        assert abs(p.area_ratio_mean - 0.05) < 1e-6
        assert p.area_ratio_std == 0.0  # same value each time

    def test_welford_accuracy(self, analyzer):
        """Welford mean/std should match known values."""
        values = [0.03, 0.05, 0.04, 0.06, 0.02]
        for i, val in enumerate(values, 1):
            face = _make_face(1, val, cx=0.5 + i * 0.01, cy=0.4)
            cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=i, avg_area=val)
            deps = {
                "face.detect": _face_obs([face], frame_id=i),
                "face.classify": _classify_obs([cf], frame_id=i),
            }
            obs = analyzer.process(_make_frame(i), deps=deps)

        p = obs.data.main_profile
        assert p.n == 5

        import numpy as np
        expected_mean = np.mean(values)
        expected_std = np.std(values, ddof=1)
        assert abs(p.area_ratio_mean - expected_mean) < 1e-10
        assert abs(p.area_ratio_std - expected_std) < 1e-10

    def test_transient_ignored(self, analyzer):
        """Transient/noise roles should not be profiled."""
        for i in range(1, 5):
            face = _make_face(1, 0.05)
            cf = ClassifiedFace(face=face, role="transient", confidence=0.5, track_length=i, avg_area=0.05)
            deps = {
                "face.detect": _face_obs([face], frame_id=i),
                "face.classify": _classify_obs([cf], frame_id=i),
            }
            obs = analyzer.process(_make_frame(i), deps=deps)

        assert obs.data.main_profile is None
        assert obs.data.profiles == []

    def test_main_and_passenger(self, analyzer):
        """Both main and passenger should be profiled."""
        for i in range(1, 4):
            main_face = _make_face(1, 0.06, cx=0.5)
            pass_face = _make_face(2, 0.03, cx=0.3)
            cf_main = ClassifiedFace(face=main_face, role="main", confidence=0.9, track_length=i, avg_area=0.06)
            cf_pass = ClassifiedFace(face=pass_face, role="passenger", confidence=0.8, track_length=i, avg_area=0.03)
            deps = {
                "face.detect": _face_obs([main_face, pass_face], frame_id=i),
                "face.classify": _classify_obs([cf_main, cf_pass], frame_id=i),
            }
            obs = analyzer.process(_make_frame(i), deps=deps)

        assert obs.data.main_profile is not None
        assert obs.data.passenger_profile is not None
        assert obs.data.main_profile.face_id == 1
        assert obs.data.passenger_profile.face_id == 2
        assert len(obs.data.profiles) == 2

    def test_no_deps(self, analyzer):
        """No deps should return observation with empty output."""
        obs = analyzer.process(_make_frame(1), deps={})
        assert obs is not None
        assert obs.data.main_profile is None

    def test_initialize_resets_state(self, analyzer):
        """initialize() should clear all accumulators."""
        for i in range(1, 5):
            face = _make_face(1, 0.05)
            cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=i, avg_area=0.05)
            deps = {
                "face.detect": _face_obs([face], frame_id=i),
                "face.classify": _classify_obs([cf], frame_id=i),
            }
            analyzer.process(_make_frame(i), deps=deps)

        analyzer.initialize()

        # After reinit, single frame should give no profile
        face = _make_face(1, 0.05)
        cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=1, avg_area=0.05)
        deps = {
            "face.detect": _face_obs([face]),
            "face.classify": _classify_obs([cf]),
        }
        obs = analyzer.process(_make_frame(1), deps=deps)
        assert obs.data.main_profile is None

    def test_signals(self, analyzer):
        """Observation signals should report convergence."""
        for i in range(1, 4):
            face = _make_face(1, 0.05)
            cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=i, avg_area=0.05)
            deps = {
                "face.detect": _face_obs([face], frame_id=i),
                "face.classify": _classify_obs([cf], frame_id=i),
            }
            obs = analyzer.process(_make_frame(i), deps=deps)

        assert obs.signals["main_converged"] == 1
        assert obs.signals["profiles_count"] == 1


class TestFaceBaselineExtract:
    """Test FrameRecord extraction for face.baseline."""

    def test_extract_face_baseline(self):
        from momentscan.algorithm.batch.types import FrameRecord
        from momentscan.algorithm.batch.extract import _extract_face_baseline

        profile = FaceBaselineProfile(
            face_id=1, role="main", n=10,
            area_ratio_mean=0.05, area_ratio_std=0.01,
            center_x_mean=0.5, center_x_std=0.02,
            center_y_mean=0.4, center_y_std=0.01,
        )
        output = FaceBaselineOutput(
            profiles=[profile],
            main_profile=profile,
        )
        obs = Observation(
            source="face.baseline",
            frame_id=1,
            t_ns=100_000_000,
            signals={},
            data=output,
        )
        record = FrameRecord(frame_idx=1, timestamp_ms=100.0)
        _extract_face_baseline(record, obs)

        assert record.baseline_n == 10
        assert abs(record.baseline_area_mean - 0.05) < 1e-8
        assert abs(record.baseline_area_std - 0.01) < 1e-8

    def test_extract_face_baseline_none(self):
        from momentscan.algorithm.batch.types import FrameRecord
        from momentscan.algorithm.batch.extract import _extract_face_baseline

        record = FrameRecord(frame_idx=1, timestamp_ms=100.0)
        _extract_face_baseline(record, None)

        assert record.baseline_n == 0
        assert record.baseline_area_mean == 0.0

    def test_extract_face_baseline_not_converged(self):
        """n < 2 should not populate record."""
        from momentscan.algorithm.batch.types import FrameRecord
        from momentscan.algorithm.batch.extract import _extract_face_baseline

        output = FaceBaselineOutput(main_profile=None)
        obs = Observation(
            source="face.baseline",
            frame_id=1,
            t_ns=100_000_000,
            signals={},
            data=output,
        )
        record = FrameRecord(frame_idx=1, timestamp_ms=100.0)
        _extract_face_baseline(record, obs)

        assert record.baseline_n == 0
