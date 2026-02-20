"""Tests for ShotQualityAnalyzer.

Tests run WITHOUT GPU or LAION weights — all optional backend calls are
either skipped or mocked.
"""

import numpy as np
import pytest

from vpx.vision_embed.types import ShotQualityOutput
from vpx.vision_embed.crop import CropRatio, face_crop, BBoxSmoother
from vpx.vision_embed.analyzer import ShotQualityAnalyzer
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput
from vpx.sdk import Observation, Capability


# ============================================================
# Mock helpers
# ============================================================


class MockFrame:
    def __init__(self, frame_id=0, t_src_ns=0, w=640, h=480):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((h, w, 3), dtype=np.uint8)


def _make_face_obs(face_count=1, image_size=(640, 480)):
    """Create a mock face.detect Observation."""
    faces = []
    for i in range(face_count):
        faces.append(FaceObservation(
            face_id=i,
            confidence=0.95,
            bbox=(0.3 + i * 0.1, 0.2, 0.15, 0.2),  # normalized
            inside_frame=True,
        ))
    data = FaceDetectOutput(
        faces=faces,
        detected_faces=[],
        image_size=image_size,
    )
    return Observation(
        source="face.detect",
        frame_id=0,
        t_ns=0,
        signals={"face_count": face_count},
        data=data,
    )


# ============================================================
# ShotQualityOutput tests
# ============================================================


class TestShotQualityOutput:
    def test_default_values(self):
        out = ShotQualityOutput()
        assert out.head_crop_box is None
        assert out.image_size is None
        assert out.head_blur == 0.0
        assert out.head_exposure == 0.0
        assert out.head_aesthetic == 0.0

    def test_with_quality_values(self):
        out = ShotQualityOutput(
            head_crop_box=(10, 20, 100, 120),
            head_blur=250.5,
        )
        assert out.head_crop_box == (10, 20, 100, 120)
        assert out.head_blur == 250.5


# ============================================================
# BBoxSmoother tests
# ============================================================


class TestBBoxSmoother:
    def test_first_update_returns_input(self):
        smoother = BBoxSmoother(alpha=0.3)
        result = smoother.update((100, 200, 50, 60))
        assert result == (100, 200, 50, 60)

    def test_ema_smoothing(self):
        smoother = BBoxSmoother(alpha=0.3)
        smoother.update((100, 100, 50, 50))

        # Second update: EMA should blend
        result = smoother.update((200, 100, 50, 50))
        # x: 0.3 * 200 + 0.7 * 100 = 130
        assert result[0] == 130
        # y unchanged: 0.3 * 100 + 0.7 * 100 = 100
        assert result[1] == 100

    def test_ema_converges(self):
        smoother = BBoxSmoother(alpha=0.3)
        smoother.update((100, 100, 50, 50))

        # Keep updating with same value
        for _ in range(50):
            result = smoother.update((200, 200, 80, 80))

        # Should converge to target
        assert abs(result[0] - 200) <= 1
        assert abs(result[1] - 200) <= 1
        assert abs(result[2] - 80) <= 1
        assert abs(result[3] - 80) <= 1

    def test_alpha_1_no_smoothing(self):
        smoother = BBoxSmoother(alpha=1.0)
        smoother.update((100, 100, 50, 50))
        result = smoother.update((200, 200, 80, 80))
        assert result == (200, 200, 80, 80)

    def test_reset(self):
        smoother = BBoxSmoother(alpha=0.3)
        smoother.update((100, 100, 50, 50))
        smoother.update((200, 200, 80, 80))
        smoother.reset()
        # After reset, first update returns input
        result = smoother.update((300, 300, 60, 60))
        assert result == (300, 300, 60, 60)


# ============================================================
# face_crop tests
# ============================================================


class TestFaceCrop:
    def test_output_size(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, box = face_crop(image, (200, 150, 100, 120))
        assert crop.shape == (224, 224, 3)

    def test_custom_output_size(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, box = face_crop(image, (200, 150, 100, 120), output_size=112)
        assert crop.shape == (112, 112, 3)

    def test_expand_increases_crop_area(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, box_small = face_crop(image, (200, 150, 100, 120), expand=1.0)
        _, box_large = face_crop(image, (200, 150, 100, 120), expand=2.0)
        area_small = box_small[2] * box_small[3]
        area_large = box_large[2] * box_large[3]
        assert area_large > area_small

    def test_bbox_clamped_to_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # bbox near edge
        crop, box = face_crop(image, (600, 440, 100, 100), expand=2.0)
        x, y, w, h = box
        assert x >= 0
        assert y >= 0
        assert x + w <= 640
        assert y + h <= 480
        assert crop.shape == (224, 224, 3)

    def test_returns_actual_box(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, box = face_crop(image, (200, 150, 100, 120))
        assert len(box) == 4
        assert all(isinstance(v, int) for v in box)

    def test_portrait_ratio(self):
        """crop_ratio="4:5" → output shape (280, 224, 3)."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, box = face_crop(image, (200, 150, 100, 120), crop_ratio="4:5")
        assert crop.shape == (280, 224, 3)  # H=280, W=224

    def test_portrait_ratio_box_is_4_5(self):
        """crop_ratio="4:5" → actual_box height ≈ width × 5/4 (clamped)."""
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        _, box = face_crop(image, (400, 400, 100, 100), expand=1.5, crop_ratio="4:5")
        x, y, w, h = box
        # h = side × 5/4 × 2 / 2... check ratio h/w ≈ 5/4
        assert h > w


# ============================================================
# ShotQualityAnalyzer tests
# ============================================================


class TestShotQualityAnalyzer:
    def test_name(self):
        analyzer = ShotQualityAnalyzer()
        assert analyzer.name == "shot.quality"

    def test_depends(self):
        assert ShotQualityAnalyzer.depends == ["face.detect"]
        assert ShotQualityAnalyzer.optional_depends == []

    def test_capabilities(self):
        analyzer = ShotQualityAnalyzer()
        caps = analyzer.capabilities
        assert Capability.STATEFUL in caps.flags
        # No GPU requirement for Layer 1 (CV only)
        assert Capability.GPU not in caps.flags

    def test_process_with_face(self):
        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame(frame_id=42, t_src_ns=1000, w=640, h=480)
        # Non-zero image so head_exposure > 0 → has_quality = True
        frame.data = np.ones((480, 640, 3), dtype=np.uint8) * 128
        deps = {"face.detect": _make_face_obs(face_count=1)}

        result = analyzer.process(frame, deps)

        assert result is not None
        assert result.source == "shot.quality"
        assert result.frame_id == 42
        assert result.t_ns == 1000
        assert result.signals["has_quality"] is True

        output: ShotQualityOutput = result.data
        assert output.head_crop_box is not None
        # CV metrics should be computed
        assert output.head_blur >= 0.0
        assert output.head_exposure >= 0.0
        # aesthetic disabled → aesthetic stays 0
        assert output.head_aesthetic == 0.0

        analyzer.cleanup()

    def test_process_no_face_detect_returns_none(self):
        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        result = analyzer.process(frame, deps={})

        assert result is None
        analyzer.cleanup()

    def test_process_no_faces_detected(self):
        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs(face_count=0)}

        result = analyzer.process(frame, deps)
        # Returns observation with zero quality (no faces)
        assert result is not None
        assert result.signals["has_quality"] is False
        assert result.data.head_blur == 0.0

        analyzer.cleanup()

    def test_process_not_initialized_raises(self):
        analyzer = ShotQualityAnalyzer()
        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}

        with pytest.raises(RuntimeError, match="not initialized"):
            analyzer.process(frame, deps)

    def test_context_manager(self):
        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)

        with analyzer:
            frame = MockFrame()
            deps = {"face.detect": _make_face_obs()}
            result = analyzer.process(frame, deps)
            assert result is not None

    def test_reset_clears_smoothers(self):
        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}

        analyzer.process(frame, deps)
        analyzer.reset()
        assert analyzer._head_smoother._state is None

        analyzer.cleanup()

    def test_metrics_populated(self):
        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}
        result = analyzer.process(frame, deps)

        assert "_metrics" in result.metadata
        assert "head_blur" in result.metadata["_metrics"]

        analyzer.cleanup()

    def test_crop_ratio_default_is_4_5(self):
        """Default crop_ratio is "4:5"."""
        analyzer = ShotQualityAnalyzer()
        assert analyzer._crop_ratio == "4:5"

    def test_crop_ratio_1_1(self):
        """crop_ratio="1:1" is accepted and stored."""
        analyzer = ShotQualityAnalyzer(crop_ratio="1:1")
        assert analyzer._crop_ratio == "1:1"

    def test_crop_ratio_1_1_process(self):
        """crop_ratio="1:1" produces quality metrics without error."""
        analyzer = ShotQualityAnalyzer(crop_ratio="1:1", enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame(w=640, h=480)
        # Non-zero image so head_exposure > 0 → has_quality = True
        frame.data = np.ones((480, 640, 3), dtype=np.uint8) * 128
        deps = {"face.detect": _make_face_obs(face_count=1)}
        result = analyzer.process(frame, deps)

        assert result is not None
        assert result.signals["has_quality"] is True
        analyzer.cleanup()

    def test_annotate_returns_marks(self):
        """annotate() returns marks for head/scene crop boxes."""
        from vpx.sdk.marks import BBoxMark

        analyzer = ShotQualityAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs(face_count=1)}
        result = analyzer.process(frame, deps)

        marks = analyzer.annotate(result)
        # At least head crop mark
        bbox_marks = [m for m in marks if isinstance(m, BBoxMark)]
        assert len(bbox_marks) >= 1

        analyzer.cleanup()

    def test_annotate_none_obs_returns_empty(self):
        """annotate(None) returns empty list."""
        analyzer = ShotQualityAnalyzer()
        assert analyzer.annotate(None) == []

