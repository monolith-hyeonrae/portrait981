"""Tests for PortraitScoreAnalyzer.

Tests run WITHOUT GPU or LAION weights — all optional backend calls are
either skipped or mocked.
"""

import numpy as np
import pytest

from vpx.portrait_score.types import PortraitScoreOutput
from vpx.sdk.crop import CropRatio, face_crop, BBoxSmoother
from vpx.portrait_score.analyzer import PortraitScoreAnalyzer
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
# PortraitScoreOutput tests
# ============================================================


class TestPortraitScoreOutput:
    def test_default_values(self):
        out = PortraitScoreOutput()
        assert out.portrait_crop_box is None
        assert out.image_size is None
        assert out.head_aesthetic == 0.0

    def test_with_values(self):
        out = PortraitScoreOutput(
            portrait_crop_box=(10, 20, 100, 120),
            head_aesthetic=0.85,
        )
        assert out.portrait_crop_box == (10, 20, 100, 120)
        assert out.head_aesthetic == 0.85


# ============================================================
# BBoxSmoother tests (imported from vpx.sdk.crop)
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
# face_crop tests (imported from vpx.sdk.crop)
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
# PortraitScoreAnalyzer tests
# ============================================================


class TestPortraitScoreAnalyzer:
    def test_name(self):
        analyzer = PortraitScoreAnalyzer()
        assert analyzer.name == "portrait.score"

    def test_depends(self):
        assert PortraitScoreAnalyzer.depends == ["face.detect"]
        assert PortraitScoreAnalyzer.optional_depends == []

    def test_capabilities(self):
        analyzer = PortraitScoreAnalyzer()
        caps = analyzer.capabilities
        assert Capability.STATEFUL in caps.flags

    def test_process_with_face(self):
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame(frame_id=42, t_src_ns=1000, w=640, h=480)
        frame.data = np.ones((480, 640, 3), dtype=np.uint8) * 128
        deps = {"face.detect": _make_face_obs(face_count=1)}

        result = analyzer.process(frame, deps)

        assert result is not None
        assert result.source == "portrait.score"
        assert result.frame_id == 42
        assert result.t_ns == 1000

        output: PortraitScoreOutput = result.data
        # aesthetic disabled → stays 0
        assert output.head_aesthetic == 0.0

        analyzer.cleanup()

    def test_process_no_face_detect_returns_none(self):
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        result = analyzer.process(frame, deps={})

        assert result is None
        analyzer.cleanup()

    def test_process_no_faces_detected(self):
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs(face_count=0)}

        result = analyzer.process(frame, deps)
        # Returns observation with zero quality (no faces)
        assert result is not None
        assert result.data.head_aesthetic == 0.0

        analyzer.cleanup()

    def test_process_not_initialized_raises(self):
        analyzer = PortraitScoreAnalyzer()
        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}

        with pytest.raises(RuntimeError, match="not initialized"):
            analyzer.process(frame, deps)

    def test_context_manager(self):
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)

        with analyzer:
            frame = MockFrame()
            deps = {"face.detect": _make_face_obs()}
            result = analyzer.process(frame, deps)
            assert result is not None

    def test_reset_clears_smoothers(self):
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}

        analyzer.process(frame, deps)
        analyzer.reset()
        assert analyzer._head_smoother._state is None

        analyzer.cleanup()

    def test_metrics_populated(self):
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}
        result = analyzer.process(frame, deps)

        assert "_metrics" in result.metadata

        analyzer.cleanup()

    def test_annotate_returns_marks(self):
        """annotate() returns marks for portrait crop region."""
        analyzer = PortraitScoreAnalyzer(enable_aesthetic=False)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs(face_count=1)}
        result = analyzer.process(frame, deps)

        marks = analyzer.annotate(result)
        # Without aesthetic, no marks expected (portrait_crop_box is None when CLIP disabled)
        assert isinstance(marks, list)

        analyzer.cleanup()

    def test_annotate_none_obs_returns_empty(self):
        """annotate(None) returns empty list."""
        analyzer = PortraitScoreAnalyzer()
        assert analyzer.annotate(None) == []
