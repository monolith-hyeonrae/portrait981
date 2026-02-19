"""Tests for vpx-face-au plugin."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vpx.face_au.analyzer import FaceAUAnalyzer
from vpx.face_au.backends.base import AU_NAMES, FaceAUResult
from vpx.face_au.output import FaceAUOutput


# ── Mocks ──


@dataclass
class MockDetectedFace:
    bbox: tuple = (100, 100, 80, 80)
    confidence: float = 0.9
    landmarks: Optional[np.ndarray] = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class MockFaceDetectOutput:
    faces: list = field(default_factory=list)
    detected_faces: list = field(default_factory=list)
    image_size: tuple = (640, 480)


@dataclass
class MockObservation:
    source: str = "face.detect"
    frame_id: int = 0
    t_ns: int = 0
    signals: dict = field(default_factory=dict)
    data: Optional[MockFaceDetectOutput] = None
    metadata: dict = field(default_factory=dict)
    timing: Optional[dict] = None


@dataclass
class MockFrame:
    frame_id: int = 0
    t_src_ns: int = 0
    data: Optional[np.ndarray] = None


class MockAUBackend:
    """Mock AU backend for testing."""

    def __init__(self, results: Optional[List[FaceAUResult]] = None):
        self._results = results or []
        self.initialized = False

    def initialize(self, device="cuda:0"):
        self.initialized = True

    def analyze(self, image, faces):
        return self._results[:len(faces)]

    def cleanup(self):
        self.initialized = False


# ── Tests ──


class TestFaceAUOutput:
    def test_empty_output(self):
        out = FaceAUOutput()
        assert out.au_intensities == []
        assert out.au_presence == []

    def test_with_data(self):
        au = {"AU12": 2.5, "AU25": 0.5}
        pres = {"AU12": True, "AU25": False}
        out = FaceAUOutput(au_intensities=[au], au_presence=[pres])
        assert len(out.au_intensities) == 1
        assert out.au_intensities[0]["AU12"] == 2.5
        assert out.au_presence[0]["AU12"] is True


class TestFaceAUResult:
    def test_default(self):
        r = FaceAUResult()
        assert r.au_intensities == {}
        assert r.au_presence == {}

    def test_with_values(self):
        au = {name: float(i) for i, name in enumerate(AU_NAMES)}
        pres = {name: i >= 1 for i, name in enumerate(AU_NAMES)}
        r = FaceAUResult(au_intensities=au, au_presence=pres)
        assert r.au_intensities["AU1"] == 0.0
        assert r.au_presence["AU1"] is False
        assert r.au_intensities["AU2"] == 1.0
        assert r.au_presence["AU2"] is True


class TestAUNames:
    def test_12_aus(self):
        assert len(AU_NAMES) == 12

    def test_key_aus_present(self):
        assert "AU12" in AU_NAMES  # smile
        assert "AU25" in AU_NAMES  # lips part
        assert "AU26" in AU_NAMES  # jaw drop


class TestFaceAUAnalyzer:
    def test_name(self):
        analyzer = FaceAUAnalyzer()
        assert analyzer.name == "face.au"

    def test_depends(self):
        assert FaceAUAnalyzer.depends == ["face.detect"]

    def test_process_no_deps(self):
        analyzer = FaceAUAnalyzer()
        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        result = analyzer.process(frame, deps=None)
        assert result is None

    def test_process_no_face_detect(self):
        analyzer = FaceAUAnalyzer()
        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        result = analyzer.process(frame, deps={})
        assert result is None

    def test_process_empty_faces(self):
        analyzer = FaceAUAnalyzer()
        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        face_obs = MockObservation(
            data=MockFaceDetectOutput(faces=[], detected_faces=[])
        )
        result = analyzer.process(frame, deps={"face.detect": face_obs})
        assert result is not None
        assert result.source == "face.au"
        assert result.data.au_intensities == []

    def test_process_with_mock_backend(self):
        mock_au = FaceAUResult(
            au_intensities={
                "AU12": 2.5, "AU25": 1.0, "AU26": 0.3,
            },
            au_presence={
                "AU12": True, "AU25": True, "AU26": False,
            },
        )
        backend = MockAUBackend(results=[mock_au])

        analyzer = FaceAUAnalyzer(au_backend=backend)
        analyzer._initialized = True

        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        detected = [MockDetectedFace()]
        face_obs = MockObservation(
            data=MockFaceDetectOutput(
                faces=[MagicMock()],
                detected_faces=detected,
            )
        )
        result = analyzer.process(frame, deps={"face.detect": face_obs})

        assert result is not None
        assert result.source == "face.au"
        assert len(result.data.au_intensities) == 1
        assert result.data.au_intensities[0]["AU12"] == 2.5
        # Signals should include main face AU values
        assert result.signals["au_au12"] == 2.5
        assert result.signals["au_au25"] == 1.0

    def test_capabilities(self):
        analyzer = FaceAUAnalyzer()
        caps = analyzer.capabilities
        assert caps.gpu_memory_mb == 256
        assert "vpx-face-au" in caps.required_extras

    def test_initialize_with_injected_backend(self):
        backend = MockAUBackend()
        analyzer = FaceAUAnalyzer(au_backend=backend)
        analyzer.initialize()
        assert analyzer._initialized

    def test_cleanup(self):
        backend = MockAUBackend()
        backend.initialized = True
        analyzer = FaceAUAnalyzer(au_backend=backend)
        analyzer.cleanup()
        assert not backend.initialized
