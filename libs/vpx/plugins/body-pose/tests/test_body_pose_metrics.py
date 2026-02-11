"""Tests for PoseAnalyzer._metrics in metadata."""

import numpy as np

from vpx.body_pose.analyzer import PoseAnalyzer
from vpx.body_pose.types import KeypointIndex
from vpx.body_pose.backends.base import PoseKeypoints


class MockPoseBackend:
    def __init__(self, poses=None):
        self._poses = poses or []

    def initialize(self, device: str) -> None:
        pass

    def detect(self, image):
        return self._poses

    def cleanup(self) -> None:
        pass


class MockFrame:
    def __init__(self, frame_id=0, t_src_ns=0, w=640, h=480):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((h, w, 3), dtype=np.uint8)


def _make_pose(person_id=0):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.1  # low conf default
    kpts[KeypointIndex.LEFT_SHOULDER] = [100, 200, 0.9]
    kpts[KeypointIndex.RIGHT_SHOULDER] = [300, 200, 0.9]
    kpts[KeypointIndex.LEFT_WRIST] = [100, 300, 0.8]
    kpts[KeypointIndex.RIGHT_WRIST] = [300, 300, 0.8]
    return PoseKeypoints(
        keypoints=kpts,
        keypoint_names=[f"kp_{i}" for i in range(17)],
        person_id=person_id,
        confidence=0.9,
    )


class TestPoseMetrics:
    def test_empty_poses_has_metrics(self):
        """No poses -> _metrics with poses_detected=0."""
        backend = MockPoseBackend(poses=[])
        analyzer = PoseAnalyzer(pose_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        assert obs.metadata is not None
        assert obs.metadata["_metrics"]["poses_detected"] == 0

    def test_poses_detected_count(self):
        """Poses detected -> _metrics with correct count and avg_keypoint_conf."""
        backend = MockPoseBackend(poses=[_make_pose(0), _make_pose(1)])
        analyzer = PoseAnalyzer(pose_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        m = obs.metadata["_metrics"]
        assert m["poses_detected"] == 2
        assert isinstance(m["avg_keypoint_conf"], float)
        assert m["avg_keypoint_conf"] > 0

    def test_metrics_key_always_present(self):
        """_metrics key exists for both empty and non-empty results."""
        # Empty
        backend_empty = MockPoseBackend(poses=[])
        analyzer_empty = PoseAnalyzer(pose_backend=backend_empty)
        analyzer_empty.initialize()
        obs_empty = analyzer_empty.process(MockFrame())
        assert "_metrics" in obs_empty.metadata

        # Non-empty
        backend_full = MockPoseBackend(poses=[_make_pose()])
        analyzer_full = PoseAnalyzer(pose_backend=backend_full)
        analyzer_full.initialize()
        obs_full = analyzer_full.process(MockFrame())
        assert "_metrics" in obs_full.metadata

    def test_existing_metadata_preserved(self):
        """_metrics does not clobber existing metadata keys."""
        backend = MockPoseBackend(poses=[_make_pose()])
        analyzer = PoseAnalyzer(pose_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        # Original metadata keys still present
        assert "wave_detected" in obs.metadata
        assert "poses_detected" in obs.metadata
        # _metrics is nested
        assert "_metrics" in obs.metadata
        assert "poses_detected" in obs.metadata["_metrics"]
