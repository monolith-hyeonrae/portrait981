"""Tests for FaceDetectionAnalyzer._metrics in metadata."""

from vpx.sdk import Observation
from vpx.face_detect.analyzer import FaceDetectionAnalyzer
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput
from vpx.face_detect.backends.base import DetectedFace

import numpy as np


class MockFaceBackend:
    """Minimal mock backend for process() testing."""

    def __init__(self, faces=None):
        self._faces = faces or []

    def initialize(self, device: str) -> None:
        pass

    def detect(self, image):
        return self._faces

    def cleanup(self) -> None:
        pass


class MockFrame:
    def __init__(self, frame_id=0, t_src_ns=0, w=640, h=480):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((h, w, 3), dtype=np.uint8)


class TestFaceDetectMetrics:
    def test_empty_detection_has_metrics(self):
        """No faces detected -> _metrics with detection_count=0."""
        backend = MockFaceBackend(faces=[])
        analyzer = FaceDetectionAnalyzer(face_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        assert obs.metadata is not None
        m = obs.metadata["_metrics"]
        assert m["detection_count"] == 0

    def test_detection_metrics_values(self):
        """Faces detected -> _metrics with count, avg_confidence, tracking_active."""
        faces = [
            DetectedFace(bbox=(200, 150, 100, 120), confidence=0.90),
            DetectedFace(bbox=(400, 150, 90, 110), confidence=0.80),
        ]
        backend = MockFaceBackend(faces=faces)
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        assert obs.metadata is not None
        m = obs.metadata["_metrics"]
        assert m["detection_count"] == 2
        assert isinstance(m["avg_confidence"], float)
        assert "tracking_active" in m

    def test_metrics_key_always_present(self):
        """_metrics key exists in metadata for both empty and non-empty results."""
        backend_empty = MockFaceBackend(faces=[])
        backend_full = MockFaceBackend(faces=[
            DetectedFace(bbox=(200, 150, 100, 120), confidence=0.95),
        ])

        for b in [backend_empty, backend_full]:
            analyzer = FaceDetectionAnalyzer(face_backend=b, track_faces=False)
            analyzer.initialize()
            obs = analyzer.process(MockFrame())
            assert "_metrics" in obs.metadata
            analyzer.cleanup()
