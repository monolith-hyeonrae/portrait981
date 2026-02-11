"""Tests for GestureAnalyzer.annotate()."""

import numpy as np

from vpx.sdk import Observation
from vpx.sdk.marks import KeypointsMark, LabelMark
from vpx.hand_gesture.analyzer import GestureAnalyzer
from vpx.hand_gesture.output import GestureOutput


def _make_hand_landmarks(gesture="v_sign", handedness="Right"):
    """Create a hand with 21 landmarks."""
    # Random-ish normalized landmarks
    landmarks = np.random.rand(21, 3).tolist()
    # Set wrist at known position
    landmarks[0] = [0.5, 0.5, 0.9]
    return {
        "handedness": handedness,
        "landmarks": landmarks,
        "confidence": 0.9,
        "gesture": gesture,
        "gesture_confidence": 0.85,
        "image_size": (640, 480),
    }


class TestGestureAnnotate:
    def test_none_obs_returns_empty(self):
        analyzer = GestureAnalyzer()
        assert analyzer.annotate(None) == []

    def test_none_data_returns_empty(self):
        analyzer = GestureAnalyzer()
        obs = Observation(source="hand.gesture", frame_id=1, t_ns=0, signals={}, data=None)
        assert analyzer.annotate(obs) == []

    def test_returns_keypoints_and_label(self):
        analyzer = GestureAnalyzer()
        hand = _make_hand_landmarks(gesture="v_sign", handedness="Right")
        obs = Observation(
            source="hand.gesture",
            frame_id=1,
            t_ns=0,
            signals={"hand_count": 1.0},
            data=GestureOutput(gestures=["v_sign"], hand_landmarks=[hand]),
        )
        marks = analyzer.annotate(obs)

        # 1 KeypointsMark + 1 LabelMark
        assert len(marks) == 2
        assert isinstance(marks[0], KeypointsMark)
        assert isinstance(marks[1], LabelMark)
        assert marks[0].normalized is True
        assert len(marks[0].points) == 21
        assert marks[1].text == "R:v_sign"

    def test_no_gesture_no_label(self):
        analyzer = GestureAnalyzer()
        hand = _make_hand_landmarks(gesture="none", handedness="Left")
        obs = Observation(
            source="hand.gesture",
            frame_id=1,
            t_ns=0,
            signals={"hand_count": 1.0},
            data=GestureOutput(gestures=["none"], hand_landmarks=[hand]),
        )
        marks = analyzer.annotate(obs)

        # Only KeypointsMark, no label for "none" gesture
        assert len(marks) == 1
        assert isinstance(marks[0], KeypointsMark)

    def test_short_landmarks_skipped(self):
        analyzer = GestureAnalyzer()
        hand = {"landmarks": [[0, 0, 0.9]] * 10, "handedness": "Right", "gesture": "none"}
        obs = Observation(
            source="hand.gesture",
            frame_id=1,
            t_ns=0,
            signals={"hand_count": 1.0},
            data=GestureOutput(gestures=[], hand_landmarks=[hand]),
        )
        marks = analyzer.annotate(obs)
        assert marks == []

    def test_empty_hands_returns_empty(self):
        analyzer = GestureAnalyzer()
        obs = Observation(
            source="hand.gesture",
            frame_id=1,
            t_ns=0,
            signals={"hand_count": 0.0},
            data=GestureOutput(gestures=[], hand_landmarks=[]),
        )
        marks = analyzer.annotate(obs)
        assert marks == []

    def test_dict_data_fallback(self):
        """annotate works with serialized dict data (from subprocess)."""
        analyzer = GestureAnalyzer()
        hand = _make_hand_landmarks(gesture="thumbs_up", handedness="Left")
        obs = Observation(
            source="hand.gesture",
            frame_id=1,
            t_ns=0,
            signals={},
            data={"hand_landmarks": [hand]},
        )
        marks = analyzer.annotate(obs)
        assert len(marks) == 2
        assert marks[1].text == "L:thumbs_up"
