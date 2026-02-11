"""Tests for ExpressionAnalyzer._metrics in metadata."""

import numpy as np
from unittest.mock import MagicMock

from vpx.sdk import Observation
from vpx.face_expression.analyzer import ExpressionAnalyzer
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput
from vpx.face_detect.backends.base import DetectedFace
from vpx.face_expression.backends.base import FaceExpression


class MockExpressionBackend:
    def __init__(self, results=None):
        self._results = results or []

    def initialize(self, device: str) -> None:
        pass

    def analyze(self, image, faces):
        return self._results[:len(faces)]

    def cleanup(self) -> None:
        pass


class MockFrame:
    def __init__(self, frame_id=0, t_src_ns=0, w=640, h=480):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((h, w, 3), dtype=np.uint8)


def _make_face_obs(n_faces=1):
    """Create a face.detect Observation with n_faces."""
    faces = [
        FaceObservation(
            face_id=i, confidence=0.9, bbox=(0.3, 0.2, 0.2, 0.3),
            inside_frame=True, yaw=5.0, pitch=3.0, roll=0.0,
            area_ratio=0.06, expression=0.0, signals={},
        )
        for i in range(n_faces)
    ]
    detected = [DetectedFace(bbox=(200, 150, 100, 120), confidence=0.9)] * n_faces
    return Observation(
        source="face.detect", frame_id=0, t_ns=0,
        signals={"face_count": n_faces},
        data=FaceDetectOutput(faces=faces, detected_faces=detected, image_size=(640, 480)),
    )


class TestExpressionMetrics:
    def test_empty_faces_has_metrics(self):
        """No faces -> _metrics with faces_analyzed=0."""
        backend = MockExpressionBackend()
        analyzer = ExpressionAnalyzer(expression_backend=backend)
        analyzer.initialize()

        face_obs = _make_face_obs(0)
        obs = analyzer.process(MockFrame(), deps={"face.detect": face_obs})

        assert obs.metadata is not None
        assert obs.metadata["_metrics"]["faces_analyzed"] == 0

    def test_faces_analyzed_count(self):
        """Faces analyzed -> _metrics with correct count."""
        expr = FaceExpression(
            dominant_emotion="happy",
            emotions={"happy": 0.8, "neutral": 0.1, "angry": 0.1},
            action_units={},
            expression_intensity=0.8,
        )
        backend = MockExpressionBackend(results=[expr, expr])
        analyzer = ExpressionAnalyzer(expression_backend=backend)
        analyzer.initialize()

        face_obs = _make_face_obs(2)
        obs = analyzer.process(MockFrame(), deps={"face.detect": face_obs})

        assert obs.metadata is not None
        assert obs.metadata["_metrics"]["faces_analyzed"] == 2

    def test_metrics_key_always_present(self):
        """_metrics key exists for both empty and non-empty results."""
        expr = FaceExpression(
            dominant_emotion="neutral",
            emotions={"neutral": 0.9},
            action_units={},
            expression_intensity=0.1,
        )
        backend = MockExpressionBackend(results=[expr])
        analyzer = ExpressionAnalyzer(expression_backend=backend)
        analyzer.initialize()

        # Empty
        obs_empty = analyzer.process(MockFrame(), deps={"face.detect": _make_face_obs(0)})
        assert "_metrics" in obs_empty.metadata

        # Non-empty
        obs_full = analyzer.process(MockFrame(), deps={"face.detect": _make_face_obs(1)})
        assert "_metrics" in obs_full.metadata
