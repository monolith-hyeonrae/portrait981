"""Tests for FaceGateAnalyzer (per-face independent gate)."""

import pytest

from vpx.sdk import Observation
from momentscan.algorithm.analyzers.face_gate import FaceGateAnalyzer, FaceGateConfig


# ── Mock helpers ──

class MockFace:
    def __init__(self, **kw):
        self.face_id = kw.get("face_id", 0)
        self.confidence = kw.get("confidence", 0.9)
        self.area_ratio = kw.get("area_ratio", 0.05)
        self.center_distance = kw.get("center_distance", 0.0)
        self.yaw = kw.get("yaw", 0.0)
        self.pitch = kw.get("pitch", 0.0)
        self.roll = kw.get("roll", 0.0)
        self.bbox = kw.get("bbox", (0.3, 0.2, 0.2, 0.3))
        self.inside_frame = kw.get("inside_frame", True)


class MockFaceOutput:
    def __init__(self, faces, image_size=(640, 480)):
        self.faces = faces
        self.image_size = image_size


class MockClassifiedFace:
    def __init__(self, face, role="main", track_length=10, avg_area=0.05, confidence=0.8):
        self.face = face
        self.role = role
        self.track_length = track_length
        self.avg_area = avg_area
        self.confidence = confidence


class MockClassifierOutput:
    def __init__(self, faces):
        self.faces = faces
        self.main_face = next((f for f in faces if f.role == "main"), None)
        self.passenger_faces = [f for f in faces if f.role == "passenger"]
        self.transient_count = sum(1 for f in faces if f.role == "transient")
        self.noise_count = sum(1 for f in faces if f.role == "noise")


class MockFrame:
    frame_id = 1
    t_src_ns = 100_000_000


class MockFaceQualityResult:
    def __init__(self, face_id=0, head_blur=0.0, head_exposure=0.0,
                 head_contrast=0.0, clipped_ratio=0.0, crushed_ratio=0.0):
        self.face_id = face_id
        self.head_blur = head_blur
        self.head_exposure = head_exposure
        self.head_crop_box = (0, 0, 0, 0)
        self.head_contrast = head_contrast
        self.clipped_ratio = clipped_ratio
        self.crushed_ratio = crushed_ratio


class MockFaceQualityOutput:
    def __init__(self, face_results=None, head_blur=0.0, head_exposure=0.0,
                 head_contrast=0.0, clipped_ratio=0.0, crushed_ratio=0.0):
        self.face_results = face_results or []
        self.head_blur = head_blur
        self.head_exposure = head_exposure
        self.head_contrast = head_contrast
        self.clipped_ratio = clipped_ratio
        self.crushed_ratio = crushed_ratio


class MockHeadPoseEstimate:
    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class MockHeadPoseOutput:
    def __init__(self, estimates):
        self.estimates = estimates


def _face_obs(faces=None, **kw):
    """Create a face.detect Observation."""
    if faces is None:
        faces = [MockFace(**kw)]
    return Observation(
        source="face.detect",
        frame_id=1,
        t_ns=0,
        data=MockFaceOutput(faces),
    )


def _classify_obs(classified_faces):
    """Create a face.classify Observation."""
    output = MockClassifierOutput(classified_faces)
    return Observation(
        source="face.classify",
        frame_id=1,
        t_ns=0,
        signals={
            "main_detected": 1 if output.main_face else 0,
            "passenger_count": len(output.passenger_faces),
        },
        data=output,
    )


def _face_quality_obs(face_results=None, head_blur=0.0, head_exposure=0.0):
    """Create a face.quality Observation."""
    return Observation(
        source="face.quality",
        frame_id=1,
        t_ns=0,
        data=MockFaceQualityOutput(
            face_results=face_results,
            head_blur=head_blur,
            head_exposure=head_exposure,
        ),
    )


def _head_pose_obs(yaw=0.0, pitch=0.0, roll=0.0):
    """Create a head.pose Observation."""
    return Observation(
        source="head.pose",
        frame_id=1,
        t_ns=0,
        data=MockHeadPoseOutput([MockHeadPoseEstimate(yaw=yaw, pitch=pitch, roll=roll)]),
    )


def _quality_obs(blur_score=0.0, brightness=0.0, contrast=0.0):
    """Create a frame.quality Observation."""
    return Observation(
        source="frame.quality",
        frame_id=1,
        t_ns=0,
        signals={"blur_score": blur_score, "brightness": brightness, "contrast": contrast},
    )


# ── Tests ──

class TestFaceGateAnalyzer:
    def test_gate_pass_main_good_frame(self):
        """Main face with all conditions met should pass."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05, yaw=10.0, pitch=5.0)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(face_id=1, head_blur=100.0, head_exposure=128.0)],
                head_blur=100.0,
                head_exposure=128.0,
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is True
        assert obs.data.main_fail_reasons == ()
        assert obs.signals["gate_passed"] is True
        assert len(obs.data.results) == 1
        assert obs.data.results[0].gate_passed is True

    def test_gate_fail_no_faces(self):
        """No classified faces should fail with face_detected."""
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        empty_face_obs = Observation(
            source="face.detect", frame_id=1, t_ns=0,
            data=MockFaceOutput([]),
        )
        empty_classify = Observation(
            source="face.classify", frame_id=1, t_ns=0,
            signals={"main_detected": 0},
            data=MockClassifierOutput([]),
        )
        deps = {
            "face.detect": empty_face_obs,
            "face.classify": empty_classify,
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is False
        assert "face_detected" in obs.data.main_fail_reasons

    def test_gate_noise_transient_auto_rejected(self):
        """Noise and transient faces are auto-rejected."""
        face1 = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        face2 = MockFace(face_id=2, confidence=0.3, area_ratio=0.002)
        face3 = MockFace(face_id=3, confidence=0.5, area_ratio=0.01)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face1, face2, face3]),
            "face.classify": _classify_obs([
                MockClassifiedFace(face1, "main"),
                MockClassifiedFace(face2, "noise"),
                MockClassifiedFace(face3, "transient"),
            ]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        # Main passes, noise and transient are rejected
        assert obs.data.main_gate_passed is True
        assert obs.signals["faces_gated"] == 3
        assert obs.signals["faces_passed"] == 1

        noise_result = next(r for r in obs.data.results if r.face_id == 2)
        assert noise_result.gate_passed is False
        assert "role_rejected" in noise_result.fail_reasons

    def test_gate_passenger_relaxed_thresholds(self):
        """Passenger face should use relaxed thresholds."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        passenger_face = MockFace(face_id=2, confidence=0.8, area_ratio=0.015)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[main_face, passenger_face]),
            "face.classify": _classify_obs([
                MockClassifiedFace(main_face, "main"),
                MockClassifiedFace(passenger_face, "passenger"),
            ]),
            "face.quality": _face_quality_obs(
                face_results=[
                    MockFaceQualityResult(face_id=1, head_blur=100.0, head_exposure=128.0),
                    MockFaceQualityResult(face_id=2, head_blur=25.0, head_exposure=128.0),
                ],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        # Passenger: no confidence/area check, blur=25 > 20 (passenger threshold)
        # Both should pass
        main_r = next(r for r in obs.data.results if r.face_id == 1)
        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert main_r.gate_passed is True
        assert pass_r.gate_passed is True

    def test_gate_passenger_no_pose_check(self):
        """Passenger face should NOT be checked for yaw/pitch."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05, yaw=10.0)
        passenger_face = MockFace(face_id=2, confidence=0.8, area_ratio=0.03, yaw=80.0)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[main_face, passenger_face]),
            "face.classify": _classify_obs([
                MockClassifiedFace(main_face, "main"),
                MockClassifiedFace(passenger_face, "passenger"),
            ]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        # Passenger yaw=80 should NOT cause failure (pose check is main-only)
        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert pass_r.gate_passed is True
        assert "head_yaw" not in pass_r.fail_reasons

    def test_gate_main_extreme_yaw_passes(self):
        """Main face with extreme yaw should still pass (no pose gate)."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05, yaw=75.0)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is True
        assert "head_yaw" not in obs.data.main_fail_reasons

    def test_gate_main_extreme_pitch_passes(self):
        """Main face with extreme pitch should still pass (no pose gate)."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05, pitch=55.0)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is True
        assert "head_pitch" not in obs.data.main_fail_reasons

    def test_gate_multiple_fail_reasons(self):
        """Multiple failing conditions should all be recorded."""
        face = MockFace(face_id=1, confidence=0.3, area_ratio=0.005)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(face_id=1, head_blur=10.0, head_exposure=250.0)],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is False
        reasons = obs.data.main_fail_reasons
        assert "face_confidence" in reasons
        assert "blur" in reasons
        assert "exposure" in reasons

    def test_gate_head_pose_overrides(self):
        """head.pose precise values should override face.detect geometric values in result."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05, yaw=80.0, pitch=0.0)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "head.pose": _head_pose_obs(yaw=20.0, pitch=10.0),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        # head.pose overrides stored yaw/pitch in result
        main_r = obs.data.results[0]
        assert main_r.head_yaw == 20.0
        assert main_r.head_pitch == 10.0
        assert obs.data.main_gate_passed is True

    def test_gate_blur_face_quality_preferred(self):
        """face.quality head_blur should be preferred over frame blur_score."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        # head_blur=20 (below 30 threshold), frame blur_score=100 (above 50)
        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(face_id=1, head_blur=20.0, head_exposure=128.0)],
            ),
            "frame.quality": _quality_obs(blur_score=100.0, brightness=128.0),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert "blur" in obs.data.main_fail_reasons

    def test_gate_blur_frame_fallback(self):
        """Without face.quality, frame blur_score should be used as fallback."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "frame.quality": _quality_obs(blur_score=30.0, brightness=128.0),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert "blur" in obs.data.main_fail_reasons

    def test_gate_unmeasured_pass(self):
        """Zero values (unmeasured) should pass by default."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05, yaw=0.0, pitch=0.0)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is True

    def test_gate_custom_config(self):
        """Custom FaceGateConfig thresholds should be applied."""
        cfg = FaceGateConfig(face_confidence_min=0.5, face_area_ratio_min=0.01)
        face = MockFace(face_id=1, confidence=0.4, area_ratio=0.05)
        analyzer = FaceGateAnalyzer(config=cfg)
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is False
        assert "face_confidence" in obs.data.main_fail_reasons

    def test_gate_name_and_depends(self):
        """Verify analyzer name and dependency declarations."""
        analyzer = FaceGateAnalyzer()
        assert analyzer.name == "face.gate"
        assert "face.detect" in analyzer.depends
        assert "face.classify" in analyzer.depends
        assert "face.quality" in analyzer.optional_depends
        assert "head.pose" in analyzer.optional_depends

    def test_gate_fallback_without_classifier(self):
        """Should work with only face.detect when face.classify is missing."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        # Falls back to treating largest face as main
        assert obs.data.main_gate_passed is True
        assert len(obs.data.results) == 1

    def test_gate_signals(self):
        """Verify signals contain per-face gate summary."""
        face1 = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        face2 = MockFace(face_id=2, confidence=0.8, area_ratio=0.03)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face1, face2]),
            "face.classify": _classify_obs([
                MockClassifiedFace(face1, "main"),
                MockClassifiedFace(face2, "passenger"),
            ]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.signals["faces_gated"] == 2
        assert obs.signals["faces_passed"] == 2
        assert obs.signals["gate_passed"] is True

    def test_gate_contrast_pass(self):
        """Good local contrast should pass exposure gate."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, head_blur=100.0, head_exposure=128.0,
                    head_contrast=0.2, clipped_ratio=0.01, crushed_ratio=0.01,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is True
        assert "exposure" not in obs.data.main_fail_reasons

    def test_gate_contrast_too_low(self):
        """Very low contrast (flat/washed) should fail exposure."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, head_blur=100.0, head_exposure=128.0,
                    head_contrast=0.02, clipped_ratio=0.0, crushed_ratio=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "exposure" in obs.data.main_fail_reasons

    def test_gate_clipped_too_high(self):
        """High clipped ratio should fail exposure."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, head_blur=100.0, head_exposure=200.0,
                    head_contrast=0.15, clipped_ratio=0.5, crushed_ratio=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "exposure" in obs.data.main_fail_reasons

    def test_gate_crushed_too_high(self):
        """High crushed ratio should fail exposure."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, head_blur=100.0, head_exposure=50.0,
                    head_contrast=0.1, clipped_ratio=0.0, crushed_ratio=0.5,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "exposure" in obs.data.main_fail_reasons

    def test_gate_contrast_zero_uses_brightness_fallback(self):
        """When head_contrast is 0 (no mask metrics), fallback to absolute brightness."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, head_blur=100.0, head_exposure=250.0,
                    head_contrast=0.0, clipped_ratio=0.0, crushed_ratio=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        # head_exposure=250 > 220 → fail
        assert obs.data.main_gate_passed is False
        assert "exposure" in obs.data.main_fail_reasons

    def test_gate_result_has_contrast_fields(self):
        """FaceGateResult should include contrast/clipped/crushed."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, head_blur=100.0, head_exposure=128.0,
                    head_contrast=0.2, clipped_ratio=0.05, crushed_ratio=0.01,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        result = obs.data.results[0]
        assert result.head_contrast == 0.2
        assert result.clipped_ratio == 0.05
        assert result.crushed_ratio == 0.01
