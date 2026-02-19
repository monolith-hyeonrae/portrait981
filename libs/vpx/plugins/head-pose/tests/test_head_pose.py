"""Tests for vpx-head-pose plugin."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from vpx.head_pose.analyzer import HeadPoseAnalyzer
from vpx.head_pose.backends.base import HeadPoseBackend
from vpx.head_pose.backends.repnet6d import rotation_matrix_to_euler
from vpx.head_pose.output import HeadPoseOutput
from vpx.head_pose.types import HeadPoseEstimate


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


class MockPoseBackend:
    """Mock pose backend for testing."""

    def __init__(self, results: Optional[List[HeadPoseEstimate]] = None):
        self._results = results or []
        self.initialized = False

    def initialize(self, device="cuda:0"):
        self.initialized = True

    def estimate(self, image, faces):
        return self._results[:len(faces)]

    def cleanup(self):
        self.initialized = False


# ── Tests ──


class TestHeadPoseEstimate:
    def test_default(self):
        est = HeadPoseEstimate()
        assert est.yaw == 0.0
        assert est.pitch == 0.0
        assert est.roll == 0.0

    def test_with_values(self):
        est = HeadPoseEstimate(yaw=30.0, pitch=-10.0, roll=5.0)
        assert est.yaw == 30.0
        assert est.pitch == -10.0
        assert est.roll == 5.0


class TestHeadPoseOutput:
    def test_empty(self):
        out = HeadPoseOutput()
        assert out.estimates == []

    def test_with_estimates(self):
        est = HeadPoseEstimate(yaw=15.0, pitch=5.0, roll=-2.0)
        out = HeadPoseOutput(estimates=[est])
        assert len(out.estimates) == 1
        assert out.estimates[0].yaw == 15.0


class TestRotationMatrixToEuler:
    def test_identity(self):
        """Identity matrix -> (0, 0, 0)."""
        R = np.eye(3)
        yaw, pitch, roll = rotation_matrix_to_euler(R)
        assert abs(yaw) < 1e-6
        assert abs(pitch) < 1e-6
        assert abs(roll) < 1e-6

    def test_yaw_90(self):
        """90 degree yaw rotation around Y axis."""
        angle = np.radians(90)
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        yaw, pitch, roll = rotation_matrix_to_euler(R)
        assert abs(yaw - 90.0) < 1.0

    def test_pitch_30(self):
        """30 degree pitch rotation around X axis."""
        angle = np.radians(30)
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
        yaw, pitch, roll = rotation_matrix_to_euler(R)
        assert abs(yaw) < 1.0
        assert abs(pitch - 30.0) < 1.0


class TestHeadPoseAnalyzer:
    def test_name(self):
        analyzer = HeadPoseAnalyzer()
        assert analyzer.name == "head.pose"

    def test_depends(self):
        assert HeadPoseAnalyzer.depends == ["face.detect"]

    def test_process_no_deps(self):
        analyzer = HeadPoseAnalyzer()
        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        result = analyzer.process(frame, deps=None)
        assert result is None

    def test_process_no_face_detect(self):
        analyzer = HeadPoseAnalyzer()
        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        result = analyzer.process(frame, deps={})
        assert result is None

    def test_process_empty_faces(self):
        analyzer = HeadPoseAnalyzer()
        frame = MockFrame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        face_obs = MockObservation(
            data=MockFaceDetectOutput(faces=[], detected_faces=[])
        )
        result = analyzer.process(frame, deps={"face.detect": face_obs})
        assert result is not None
        assert result.source == "head.pose"
        assert result.data.estimates == []

    def test_process_with_mock_backend(self):
        mock_est = HeadPoseEstimate(yaw=25.0, pitch=-5.0, roll=3.0)
        backend = MockPoseBackend(results=[mock_est])

        analyzer = HeadPoseAnalyzer(pose_backend=backend)
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
        assert result.source == "head.pose"
        assert len(result.data.estimates) == 1
        assert result.data.estimates[0].yaw == 25.0
        assert result.signals["head_yaw"] == 25.0
        assert result.signals["head_pitch"] == -5.0
        assert result.signals["head_roll"] == 3.0

    def test_capabilities(self):
        analyzer = HeadPoseAnalyzer()
        caps = analyzer.capabilities
        assert caps.gpu_memory_mb == 512
        assert "vpx-head-pose" in caps.required_extras

    def test_cleanup(self):
        backend = MockPoseBackend()
        backend.initialized = True
        analyzer = HeadPoseAnalyzer(pose_backend=backend)
        analyzer.cleanup()
        assert not backend.initialized
