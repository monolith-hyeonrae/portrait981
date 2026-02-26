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
    def __init__(self, face_id=0, face_blur=0.0, face_exposure=0.0,
                 face_contrast=0.0, clipped_ratio=0.0, crushed_ratio=0.0,
                 parsing_coverage=0.0, seg_mouth=0.0, seg_face=0.0):
        self.face_id = face_id
        self.face_blur = face_blur
        self.face_exposure = face_exposure
        self.face_contrast = face_contrast
        self.clipped_ratio = clipped_ratio
        self.crushed_ratio = crushed_ratio
        self.parsing_coverage = parsing_coverage
        self.seg_mouth = seg_mouth
        self.seg_face = seg_face


class MockFaceQualityOutput:
    def __init__(self, face_results=None, face_blur=0.0, face_exposure=0.0,
                 face_contrast=0.0, clipped_ratio=0.0, crushed_ratio=0.0):
        self.face_results = face_results or []
        self.face_blur = face_blur
        self.face_exposure = face_exposure
        self.face_contrast = face_contrast
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


def _face_quality_obs(face_results=None, face_blur=0.0, face_exposure=0.0):
    """Create a face.quality Observation."""
    return Observation(
        source="face.quality",
        frame_id=1,
        t_ns=0,
        data=MockFaceQualityOutput(
            face_results=face_results,
            face_blur=face_blur,
            face_exposure=face_exposure,
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
                face_results=[MockFaceQualityResult(face_id=1, face_blur=100.0, face_exposure=128.0)],
                face_blur=100.0,
                face_exposure=128.0,
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is True
        assert obs.data.main_fail_reasons == ()
        assert obs.signals["gate_passed"] is True
        assert len(obs.data.results) == 1
        assert obs.data.results[0].gate_passed is True

    def test_gate_fail_no_faces(self):
        """No classified faces should fail with gate.detect.missing."""
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
        assert "gate.detect.missing" in obs.data.main_fail_reasons

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
        assert "gate.role.rejected" in noise_result.fail_reasons

    def test_gate_passenger_always_passes(self):
        """Passenger face should always gate_passed=True (no hard gate)."""
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
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert pass_r.gate_passed is True
        assert pass_r.fail_reasons == ()

    def test_gate_passenger_suitability_full(self):
        """Passenger with good confidence and parsing → suitability=1.0."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        passenger_face = MockFace(face_id=2, confidence=0.8, area_ratio=0.03)
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
                    MockFaceQualityResult(face_id=1, face_blur=100.0, face_exposure=128.0),
                    MockFaceQualityResult(face_id=2, face_blur=50.0, face_exposure=128.0,
                                          parsing_coverage=0.7),
                ],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert pass_r.gate_passed is True
        # confidence=0.8 >= 0.5 → conf_score=1.0, parsing=0.7 >= 0.5 → parse_score=1.0
        assert pass_r.suitability == pytest.approx(1.0)

    def test_gate_passenger_suitability_partial(self):
        """Passenger with low confidence → suitability proportionally reduced."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        passenger_face = MockFace(face_id=2, confidence=0.25, area_ratio=0.02)
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

        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert pass_r.gate_passed is True
        # confidence=0.25 / 0.5 = 0.5, no parsing → parse_score=1.0
        assert pass_r.suitability == pytest.approx(0.5)

    def test_gate_passenger_suitability_zero(self):
        """Passenger with zero confidence → suitability=0.0."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        passenger_face = MockFace(face_id=2, confidence=0.0, area_ratio=0.01)
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

        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert pass_r.gate_passed is True
        assert pass_r.suitability == pytest.approx(0.0)

    def test_gate_passenger_no_parsing_data(self):
        """Passenger with parsing=0 (not measured) → parse_score=1.0 (no penalty)."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        passenger_face = MockFace(face_id=2, confidence=0.6, area_ratio=0.03)
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
                    MockFaceQualityResult(face_id=1, face_blur=100.0, face_exposure=128.0),
                    MockFaceQualityResult(face_id=2, face_blur=50.0, face_exposure=128.0,
                                          parsing_coverage=0.0),
                ],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        # confidence=0.6 >= 0.5 → conf_score=1.0, parsing=0 → parse_score=1.0
        assert pass_r.suitability == pytest.approx(1.0)

    def test_gate_passenger_ignores_blur_exposure(self):
        """Passenger with bad blur/exposure still passes (no hard gate)."""
        main_face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        passenger_face = MockFace(face_id=2, confidence=0.8, area_ratio=0.03, yaw=80.0)
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
                    MockFaceQualityResult(face_id=1, face_blur=100.0, face_exposure=128.0),
                    MockFaceQualityResult(face_id=2, face_blur=1.0, face_exposure=300.0),
                ],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        pass_r = next(r for r in obs.data.results if r.face_id == 2)
        assert pass_r.gate_passed is True
        assert pass_r.fail_reasons == ()

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
                face_results=[MockFaceQualityResult(face_id=1, face_blur=3.0, face_exposure=250.0)],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is False
        reasons = obs.data.main_fail_reasons
        assert "gate.detect.confidence" in reasons
        assert "gate.blur.face" in reasons
        assert "gate.exposure.brightness" in reasons

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
        """face.quality face_blur should be preferred over frame blur_score."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        # face_blur=3 (below 5 threshold), frame blur_score=100 (above 50)
        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(face_id=1, face_blur=3.0, face_exposure=128.0)],
            ),
            "frame.quality": _quality_obs(blur_score=100.0, brightness=128.0),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert "gate.blur.face" in obs.data.main_fail_reasons

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

        assert "gate.blur.frame" in obs.data.main_fail_reasons

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
        cfg = FaceGateConfig(face_confidence_min=0.5)
        face = MockFace(face_id=1, confidence=0.4, area_ratio=0.05)
        analyzer = FaceGateAnalyzer(config=cfg)
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs.data.main_gate_passed is False
        assert "gate.detect.confidence" in obs.data.main_fail_reasons

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
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    face_contrast=0.2, clipped_ratio=0.01, crushed_ratio=0.01,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is True
        assert all(r not in obs.data.main_fail_reasons
                    for r in ("gate.exposure.contrast", "gate.exposure.white", "gate.exposure.black"))

    def test_gate_contrast_too_low(self):
        """Very low contrast (flat/washed) should fail exposure.contrast."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    face_contrast=0.02, clipped_ratio=0.0, crushed_ratio=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "gate.exposure.contrast" in obs.data.main_fail_reasons

    def test_gate_clipped_too_high(self):
        """High clipped ratio should fail exposure.white."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=200.0,
                    face_contrast=0.15, clipped_ratio=0.5, crushed_ratio=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "gate.exposure.white" in obs.data.main_fail_reasons

    def test_gate_crushed_too_high(self):
        """High crushed ratio should fail exposure.black."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=50.0,
                    face_contrast=0.1, clipped_ratio=0.0, crushed_ratio=0.5,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "gate.exposure.black" in obs.data.main_fail_reasons

    def test_gate_contrast_zero_uses_brightness_fallback(self):
        """When face_contrast is 0 (no mask metrics), fallback to exposure.brightness."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=250.0,
                    face_contrast=0.0, clipped_ratio=0.0, crushed_ratio=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        # face_exposure=250 > 220 → fail
        assert obs.data.main_gate_passed is False
        assert "gate.exposure.brightness" in obs.data.main_fail_reasons

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
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    face_contrast=0.2, clipped_ratio=0.05, crushed_ratio=0.01,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        result = obs.data.results[0]
        assert result.face_contrast == 0.2
        assert result.clipped_ratio == 0.05
        assert result.crushed_ratio == 0.01

    def test_gate_parsing_coverage_pass(self):
        """Good parsing coverage should pass gate."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.6, seg_mouth=0.05, seg_face=0.3,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is True
        assert "gate.parsing.coverage" not in obs.data.main_fail_reasons
        assert obs.data.results[0].parsing_coverage == 0.6

    def test_gate_parsing_coverage_fail(self):
        """Low parsing coverage should fail with parsing.coverage."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.05,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "gate.parsing.coverage" in obs.data.main_fail_reasons

    def test_gate_parsing_coverage_zero_skipped(self):
        """Zero parsing coverage (not measured) should be skipped."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is True
        assert "gate.parsing.coverage" not in obs.data.main_fail_reasons

    def test_gate_parsing_coverage_custom_threshold(self):
        """Custom parsing_coverage_min should be respected."""
        cfg = FaceGateConfig(parsing_coverage_min=0.3)
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer(config=cfg)
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.2,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "gate.parsing.coverage" in obs.data.main_fail_reasons

    def test_gate_mask_detection_seg_mouth_low(self):
        """Low seg_mouth (mask detected) should NOT reject — skips seg_face check instead."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.6, seg_mouth=0.005, seg_face=0.05,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        # Mask detected → seg_face check skipped despite seg_face=0.05 < 0.10
        assert obs.data.main_gate_passed is True
        assert "gate.occlusion.mask" not in obs.data.main_fail_reasons
        assert "gate.exposure.seg_face" not in obs.data.main_fail_reasons

    def test_gate_mask_detection_seg_mouth_ok(self):
        """Normal seg_mouth with good parsing coverage should pass."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.6, seg_mouth=0.05, seg_face=0.3,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is True
        assert "gate.occlusion.mask" not in obs.data.main_fail_reasons

    def test_gate_mask_skipped_when_parsing_low(self):
        """Mask detection should not fire when parsing_coverage is below threshold."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.05, seg_mouth=0.0, seg_face=0.0,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        # parsing_coverage=0.05 < 0.50 → parsing.coverage fails but mask/seg_face checks skipped
        assert "gate.occlusion.mask" not in obs.data.main_fail_reasons
        assert "gate.exposure.seg_face" not in obs.data.main_fail_reasons

    def test_gate_seg_face_collapse(self):
        """Low seg_face with good parsing should fail with exposure.seg_face."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.6, seg_mouth=0.03, seg_face=0.05,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is False
        assert "gate.exposure.seg_face" in obs.data.main_fail_reasons

    def test_gate_seg_face_ok(self):
        """Normal seg_face with good parsing should pass."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.6, seg_mouth=0.05, seg_face=0.25,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs.data.main_gate_passed is True
        assert "gate.exposure.seg_face" not in obs.data.main_fail_reasons

    def test_gate_result_has_seg_fields(self):
        """FaceGateResult should include seg_mouth and seg_face values."""
        face = MockFace(face_id=1, confidence=0.9, area_ratio=0.05)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=100.0, face_exposure=128.0,
                    parsing_coverage=0.4, seg_mouth=0.08, seg_face=0.35,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        result = obs.data.results[0]
        assert result.seg_mouth == 0.08
        assert result.seg_face == 0.35

    def test_gate_all_reasons_have_gate_prefix(self):
        """All fail reasons should use gate.* prefix."""
        face = MockFace(face_id=1, confidence=0.3, area_ratio=0.005)
        analyzer = FaceGateAnalyzer()
        analyzer.initialize()

        deps = {
            "face.detect": _face_obs(faces=[face]),
            "face.classify": _classify_obs([MockClassifiedFace(face, "main")]),
            "face.quality": _face_quality_obs(
                face_results=[MockFaceQualityResult(
                    face_id=1, face_blur=3.0, face_exposure=128.0,
                    face_contrast=0.02, clipped_ratio=0.5, crushed_ratio=0.5,
                    parsing_coverage=0.05,
                )],
            ),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        for reason in obs.data.main_fail_reasons:
            assert reason.startswith("gate."), f"Fail reason '{reason}' missing gate. prefix"

