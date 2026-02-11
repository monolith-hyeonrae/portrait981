"""Tests for PoseAnalyzer.annotate()."""

import numpy as np

from vpx.sdk import Observation
from vpx.sdk.marks import KeypointsMark
from vpx.body_pose.analyzer import PoseAnalyzer
from vpx.body_pose.output import PoseOutput
from vpx.body_pose.types import KeypointIndex


def _make_person_keypoints():
    """Create realistic upper body keypoints."""
    kpts = np.zeros((17, 3))
    kpts[KeypointIndex.NOSE] = [320, 100, 0.9]
    kpts[KeypointIndex.LEFT_EYE] = [310, 90, 0.85]
    kpts[KeypointIndex.RIGHT_EYE] = [330, 90, 0.85]
    kpts[KeypointIndex.LEFT_EAR] = [300, 95, 0.7]
    kpts[KeypointIndex.RIGHT_EAR] = [340, 95, 0.7]
    kpts[KeypointIndex.LEFT_SHOULDER] = [280, 180, 0.9]
    kpts[KeypointIndex.RIGHT_SHOULDER] = [360, 180, 0.9]
    kpts[KeypointIndex.LEFT_ELBOW] = [250, 250, 0.8]
    kpts[KeypointIndex.RIGHT_ELBOW] = [390, 250, 0.8]
    kpts[KeypointIndex.LEFT_WRIST] = [230, 320, 0.75]
    kpts[KeypointIndex.RIGHT_WRIST] = [410, 320, 0.75]
    return kpts


class TestPoseAnnotate:
    def test_none_obs_returns_empty(self):
        analyzer = PoseAnalyzer()
        assert analyzer.annotate(None) == []

    def test_none_data_returns_empty(self):
        analyzer = PoseAnalyzer()
        obs = Observation(source="body.pose", frame_id=1, t_ns=0, signals={}, data=None)
        assert analyzer.annotate(obs) == []

    def test_returns_keypoints_marks(self):
        analyzer = PoseAnalyzer()
        kpts = _make_person_keypoints()
        obs = Observation(
            source="body.pose",
            frame_id=1,
            t_ns=0,
            signals={"person_count": 1.0},
            data=PoseOutput(
                keypoints=[{"person_id": 0, "keypoints": kpts.tolist(), "image_size": (640, 480)}],
                person_count=1,
            ),
        )
        marks = analyzer.annotate(obs)

        assert len(marks) == 1
        assert isinstance(marks[0], KeypointsMark)
        assert len(marks[0].points) == 11  # upper body only
        assert marks[0].normalized is False
        assert len(marks[0].connections) == 9  # skeleton connections

    def test_short_keypoints_skipped(self):
        analyzer = PoseAnalyzer()
        obs = Observation(
            source="body.pose",
            frame_id=1,
            t_ns=0,
            signals={"person_count": 1.0},
            data=PoseOutput(
                keypoints=[{"person_id": 0, "keypoints": [[0, 0, 0.5]] * 5}],
                person_count=1,
            ),
        )
        marks = analyzer.annotate(obs)
        assert marks == []

    def test_empty_keypoints_returns_empty(self):
        analyzer = PoseAnalyzer()
        obs = Observation(
            source="body.pose",
            frame_id=1,
            t_ns=0,
            signals={"person_count": 0.0},
            data=PoseOutput(keypoints=[], person_count=0),
        )
        marks = analyzer.annotate(obs)
        assert marks == []

    def test_dict_data_fallback(self):
        """annotate works with serialized dict data (from subprocess)."""
        analyzer = PoseAnalyzer()
        kpts = _make_person_keypoints()
        obs = Observation(
            source="body.pose",
            frame_id=1,
            t_ns=0,
            signals={"person_count": 1.0},
            data={"keypoints": [{"person_id": 0, "keypoints": kpts.tolist()}]},
        )
        marks = analyzer.annotate(obs)
        assert len(marks) == 1
        assert isinstance(marks[0], KeypointsMark)
