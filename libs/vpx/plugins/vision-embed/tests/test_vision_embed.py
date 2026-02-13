"""Tests for VisionEmbedAnalyzer.

Tests run WITHOUT GPU — all model calls are mocked.
"""

import numpy as np
import pytest

from vpx.vision_embed.types import EmbedOutput
from vpx.vision_embed.crop import face_crop, body_crop, BBoxSmoother
from vpx.vision_embed.analyzer import VisionEmbedAnalyzer
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput
from vpx.sdk import Observation, Capability


# ============================================================
# Mock helpers
# ============================================================


class MockEmbeddingBackend:
    """Mock backend that returns random L2-normalized embeddings."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.embed_calls = 0

    @property
    def embed_dim(self) -> int:
        return self._dim

    def initialize(self, device: str) -> None:
        pass

    def embed(self, image: np.ndarray) -> np.ndarray:
        self.embed_calls += 1
        rng = np.random.RandomState(self.embed_calls)
        vec = rng.randn(self._dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec

    def cleanup(self) -> None:
        pass


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
# EmbedOutput tests
# ============================================================


class TestEmbedOutput:
    def test_default_none_fields(self):
        out = EmbedOutput()
        assert out.e_face is None
        assert out.e_body is None
        assert out.face_crop_box is None
        assert out.body_crop_box is None

    def test_with_embedding(self):
        vec = np.random.randn(384).astype(np.float32)
        out = EmbedOutput(e_face=vec, face_crop_box=(10, 20, 100, 120))
        assert out.e_face is not None
        assert out.e_face.shape == (384,)
        assert out.face_crop_box == (10, 20, 100, 120)


# ============================================================
# Embedding dimension & L2 normalization tests
# ============================================================


class TestEmbeddingProperties:
    def test_embedding_dim_384(self):
        backend = MockEmbeddingBackend(dim=384)
        backend.initialize("cpu")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        vec = backend.embed(img)
        assert vec.shape == (384,)

    def test_l2_normalization(self):
        backend = MockEmbeddingBackend(dim=384)
        backend.initialize("cpu")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        vec = backend.embed(img)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ~1.0, got {norm}"

    def test_different_images_different_embeddings(self):
        backend = MockEmbeddingBackend(dim=384)
        backend.initialize("cpu")
        img1 = np.zeros((224, 224, 3), dtype=np.uint8)
        img2 = np.ones((224, 224, 3), dtype=np.uint8) * 128
        vec1 = backend.embed(img1)
        vec2 = backend.embed(img2)
        # Mock uses call count as seed, so they should differ
        assert not np.allclose(vec1, vec2)


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


# ============================================================
# body_crop tests
# ============================================================


class TestBodyCrop:
    def test_output_size(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Shoulder and hip keypoints (COCO indices 5, 6, 11, 12)
        kps = np.zeros((17, 2), dtype=np.float32)
        kps[5] = [200, 150]   # left shoulder
        kps[6] = [400, 150]   # right shoulder
        kps[11] = [220, 350]  # left hip
        kps[12] = [380, 350]  # right hip
        crop, box = body_crop(image, kps)
        assert crop.shape == (224, 224, 3)

    def test_fallback_with_insufficient_keypoints(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.zeros((17, 2), dtype=np.float32)  # All zeros
        crop, box = body_crop(image, kps)
        assert crop.shape == (224, 224, 3)

    def test_with_confidence_column(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [200, 150, 0.9]
        kps[6] = [400, 150, 0.8]
        kps[11] = [220, 350, 0.7]
        kps[12] = [380, 350, 0.6]
        crop, box = body_crop(image, kps)
        assert crop.shape == (224, 224, 3)


# ============================================================
# VisionEmbedAnalyzer tests
# ============================================================


class TestVisionEmbedAnalyzer:
    def test_name(self):
        analyzer = VisionEmbedAnalyzer()
        assert analyzer.name == "vision.embed"

    def test_depends(self):
        assert VisionEmbedAnalyzer.depends == ["face.detect"]
        assert VisionEmbedAnalyzer.optional_depends == ["body.pose"]

    def test_capabilities(self):
        analyzer = VisionEmbedAnalyzer()
        caps = analyzer.capabilities
        assert Capability.GPU in caps.flags
        assert Capability.STATEFUL in caps.flags
        assert caps.gpu_memory_mb == 512
        assert "torch" in caps.resource_groups

    def test_process_with_face(self):
        backend = MockEmbeddingBackend(dim=384)
        analyzer = VisionEmbedAnalyzer(backend=backend)
        analyzer.initialize()

        frame = MockFrame(frame_id=42, t_src_ns=1000)
        deps = {"face.detect": _make_face_obs(face_count=1)}

        result = analyzer.process(frame, deps)

        assert result is not None
        assert result.source == "vision.embed"
        assert result.frame_id == 42
        assert result.t_ns == 1000
        assert result.signals["has_face_embed"] is True
        assert result.signals["has_body_embed"] is False

        output: EmbedOutput = result.data
        assert output.e_face is not None
        assert output.e_face.shape == (384,)
        assert output.face_crop_box is not None
        assert output.e_body is None

        # Check L2 norm
        norm = np.linalg.norm(output.e_face)
        assert abs(norm - 1.0) < 1e-5

        analyzer.cleanup()

    def test_process_no_face_detect_returns_none(self):
        backend = MockEmbeddingBackend()
        analyzer = VisionEmbedAnalyzer(backend=backend)
        analyzer.initialize()

        frame = MockFrame()
        result = analyzer.process(frame, deps={})

        assert result is None
        analyzer.cleanup()

    def test_process_no_faces_detected(self):
        backend = MockEmbeddingBackend()
        analyzer = VisionEmbedAnalyzer(backend=backend)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs(face_count=0)}

        result = analyzer.process(frame, deps)
        assert result is not None
        assert result.signals["has_face_embed"] is False
        assert result.data.e_face is None

        analyzer.cleanup()

    def test_process_not_initialized_raises(self):
        analyzer = VisionEmbedAnalyzer()
        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}

        with pytest.raises(RuntimeError, match="not initialized"):
            analyzer.process(frame, deps)

    def test_context_manager(self):
        backend = MockEmbeddingBackend()
        analyzer = VisionEmbedAnalyzer(backend=backend)

        with analyzer:
            frame = MockFrame()
            deps = {"face.detect": _make_face_obs()}
            result = analyzer.process(frame, deps)
            assert result is not None

    def test_reset_clears_smoothers(self):
        backend = MockEmbeddingBackend()
        analyzer = VisionEmbedAnalyzer(backend=backend)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}

        # Process once to populate smoother state
        analyzer.process(frame, deps)

        # Reset should clear smoother state
        analyzer.reset()
        assert analyzer._face_smoother._state is None
        assert analyzer._body_smoother._state is None

        analyzer.cleanup()

    def test_metrics_populated(self):
        backend = MockEmbeddingBackend()
        analyzer = VisionEmbedAnalyzer(backend=backend)
        analyzer.initialize()

        frame = MockFrame()
        deps = {"face.detect": _make_face_obs()}
        result = analyzer.process(frame, deps)

        assert "_metrics" in result.metadata
        assert result.metadata["_metrics"]["face_embed_computed"] is True
        assert result.metadata["_metrics"]["body_embed_computed"] is False

        analyzer.cleanup()

    def test_entry_point_registered(self):
        """Verify the entry point name matches the analyzer."""
        analyzer = VisionEmbedAnalyzer()
        assert analyzer.name == "vision.embed"

    def test_embed_dim_consistency(self):
        """Backend embed_dim matches actual output dimension."""
        backend = MockEmbeddingBackend(dim=384)
        assert backend.embed_dim == 384
        backend.initialize("cpu")
        vec = backend.embed(np.zeros((224, 224, 3), dtype=np.uint8))
        assert vec.shape == (backend.embed_dim,)
