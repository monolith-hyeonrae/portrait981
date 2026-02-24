"""Tests for FaceParseAnalyzer."""

import numpy as np
import pytest

from vpx.sdk import Observation
from vpx.sdk.testing import FakeFrame, PluginTestHarness, assert_valid_observation
from vpx.face_parse import FaceParseAnalyzer
from vpx.face_parse.output import FaceParseOutput, FaceParseResult


# ── Mocks ──

class MockDetectedFace:
    def __init__(self, face_id=0, bbox=(100, 100, 200, 200)):
        self.face_id = face_id
        self.bbox = bbox
        self.confidence = 0.9
        self.landmarks = None


class MockFaceDetectOutput:
    def __init__(self, detected_faces):
        self.detected_faces = detected_faces
        self.faces = []
        self.image_size = (640, 480)


class MockParseBackend:
    """Mock backend that returns a simple mask."""

    def __init__(self):
        self.initialized = False
        self.call_count = 0

    def initialize(self, device="cpu"):
        self.initialized = True

    def segment(self, image, detected_faces):
        self.call_count += 1
        results = []
        for face in detected_faces:
            mask = np.ones((512, 512), dtype=np.uint8) * 255
            class_map = np.ones((512, 512), dtype=np.uint8)
            bx, by, bw, bh = face.bbox
            results.append(FaceParseResult(
                face_id=face.face_id,
                face_mask=mask,
                crop_box=(bx, by, bw, bh),
                class_map=class_map,
            ))
        return results

    def cleanup(self):
        self.initialized = False


def _face_obs(detected_faces=None):
    if detected_faces is None:
        detected_faces = [MockDetectedFace()]
    return Observation(
        source="face.detect",
        frame_id=1,
        t_ns=0,
        data=MockFaceDetectOutput(detected_faces),
    )


# ── Tests ──

class TestFaceParseAnalyzer:
    def test_harness(self):
        analyzer = FaceParseAnalyzer()
        analyzer.initialize()
        harness = PluginTestHarness()
        report = harness.check_module(analyzer)
        assert report.valid, report.errors
        analyzer.cleanup()

    def test_name_and_depends(self):
        analyzer = FaceParseAnalyzer()
        assert analyzer.name == "face.parse"
        assert "face.detect" in analyzer.depends

    def test_process_with_mock_backend(self):
        backend = MockParseBackend()
        analyzer = FaceParseAnalyzer(parse_backend=backend)
        analyzer.initialize()

        deps = {"face.detect": _face_obs()}
        frame = FakeFrame.create()
        obs = analyzer.process(frame, deps=deps)

        assert obs is not None
        assert obs.source == "face.parse"
        assert obs.signals["faces_parsed"] == 1
        assert len(obs.data.results) == 1
        assert obs.data.results[0].face_id == 0
        assert backend.call_count == 1
        analyzer.cleanup()

    def test_process_multiple_faces(self):
        faces = [
            MockDetectedFace(face_id=0, bbox=(100, 100, 200, 200)),
            MockDetectedFace(face_id=1, bbox=(300, 100, 150, 150)),
        ]
        backend = MockParseBackend()
        analyzer = FaceParseAnalyzer(parse_backend=backend)
        analyzer.initialize()

        deps = {"face.detect": _face_obs(faces)}
        obs = analyzer.process(FakeFrame.create(), deps=deps)

        assert obs.signals["faces_parsed"] == 2
        assert len(obs.data.results) == 2
        analyzer.cleanup()

    def test_process_no_face_detect(self):
        backend = MockParseBackend()
        analyzer = FaceParseAnalyzer(parse_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(FakeFrame.create(), deps={})
        assert obs is None
        analyzer.cleanup()

    def test_process_empty_faces(self):
        backend = MockParseBackend()
        analyzer = FaceParseAnalyzer(parse_backend=backend)
        analyzer.initialize()

        deps = {"face.detect": _face_obs(detected_faces=[])}
        obs = analyzer.process(FakeFrame.create(), deps=deps)

        assert obs is not None
        assert obs.signals["faces_parsed"] == 0
        analyzer.cleanup()

    def test_process_no_backend(self):
        """Without backend, should produce empty results."""
        analyzer = FaceParseAnalyzer()
        analyzer._initialized = True

        deps = {"face.detect": _face_obs()}
        obs = analyzer.process(FakeFrame.create(), deps=deps)

        assert obs is not None
        assert obs.signals["faces_parsed"] == 0

    def test_cleanup(self):
        backend = MockParseBackend()
        backend.initialize()  # externally provided backends are pre-initialized
        analyzer = FaceParseAnalyzer(parse_backend=backend)
        analyzer.initialize()
        assert backend.initialized

        analyzer.cleanup()
        assert not backend.initialized

    def test_capabilities(self):
        analyzer = FaceParseAnalyzer()
        caps = analyzer.capabilities
        assert caps.gpu_memory_mb == 200

    def test_annotate_empty(self):
        analyzer = FaceParseAnalyzer()
        marks = analyzer.annotate(None)
        assert marks == []


class TestFaceParseOutput:
    def test_default_output(self):
        output = FaceParseOutput()
        assert output.results == []

    def test_result_defaults(self):
        result = FaceParseResult()
        assert result.face_id == 0
        assert result.crop_box == (0, 0, 0, 0)
        assert result.face_mask.shape == (1, 1)
