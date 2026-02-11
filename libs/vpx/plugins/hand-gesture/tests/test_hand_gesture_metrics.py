"""Tests for GestureAnalyzer._metrics in metadata."""

import numpy as np

from vpx.hand_gesture.analyzer import GestureAnalyzer
from vpx.hand_gesture.types import HandLandmarkIndex
from vpx.hand_gesture.backends.base import HandLandmarks


class MockHandBackend:
    def __init__(self, hands=None):
        self._hands = hands or []

    def initialize(self, device: str) -> None:
        pass

    def detect(self, image):
        return self._hands

    def cleanup(self) -> None:
        pass


class MockFrame:
    def __init__(self, frame_id=0, t_src_ns=0, w=640, h=480):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((h, w, 3), dtype=np.uint8)


def _make_vsign_hand():
    """Create hand landmarks for V-sign (index + middle up)."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[HandLandmarkIndex.WRIST] = [0.5, 0.6, 0.0]
    # Thumb down (right hand: tip.x < ip.x)
    landmarks[HandLandmarkIndex.THUMB_IP] = [0.40, 0.48, 0.0]
    landmarks[HandLandmarkIndex.THUMB_TIP] = [0.38, 0.50, 0.0]
    # Index up (tip.y < pip.y)
    landmarks[HandLandmarkIndex.INDEX_FINGER_PIP] = [0.48, 0.40, 0.0]
    landmarks[HandLandmarkIndex.INDEX_FINGER_TIP] = [0.48, 0.20, 0.0]
    # Middle up
    landmarks[HandLandmarkIndex.MIDDLE_FINGER_PIP] = [0.50, 0.38, 0.0]
    landmarks[HandLandmarkIndex.MIDDLE_FINGER_TIP] = [0.50, 0.18, 0.0]
    # Ring down
    landmarks[HandLandmarkIndex.RING_FINGER_PIP] = [0.52, 0.52, 0.0]
    landmarks[HandLandmarkIndex.RING_FINGER_TIP] = [0.52, 0.56, 0.0]
    # Pinky down
    landmarks[HandLandmarkIndex.PINKY_PIP] = [0.54, 0.54, 0.0]
    landmarks[HandLandmarkIndex.PINKY_TIP] = [0.54, 0.58, 0.0]
    return HandLandmarks(landmarks=landmarks, handedness="Right", confidence=0.9)


class TestGestureMetrics:
    def test_empty_hands_has_metrics(self):
        """No hands -> _metrics with hands_detected=0, gestures_recognized=0."""
        backend = MockHandBackend(hands=[])
        analyzer = GestureAnalyzer(hand_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        m = obs.metadata["_metrics"]
        assert m["hands_detected"] == 0
        assert m["gestures_recognized"] == 0

    def test_gesture_recognized_count(self):
        """Hand with gesture -> _metrics with gestures_recognized=1."""
        backend = MockHandBackend(hands=[_make_vsign_hand()])
        analyzer = GestureAnalyzer(hand_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        m = obs.metadata["_metrics"]
        assert m["hands_detected"] == 1
        assert m["gestures_recognized"] == 1

    def test_metrics_key_always_present(self):
        """_metrics key exists for both empty and non-empty results."""
        # Empty
        backend_empty = MockHandBackend(hands=[])
        analyzer_empty = GestureAnalyzer(hand_backend=backend_empty)
        analyzer_empty.initialize()
        obs_empty = analyzer_empty.process(MockFrame())
        assert "_metrics" in obs_empty.metadata

        # Non-empty
        backend_full = MockHandBackend(hands=[_make_vsign_hand()])
        analyzer_full = GestureAnalyzer(hand_backend=backend_full)
        analyzer_full.initialize()
        obs_full = analyzer_full.process(MockFrame())
        assert "_metrics" in obs_full.metadata

    def test_existing_metadata_preserved(self):
        """_metrics does not clobber existing metadata keys."""
        backend = MockHandBackend(hands=[_make_vsign_hand()])
        analyzer = GestureAnalyzer(hand_backend=backend)
        analyzer.initialize()

        obs = analyzer.process(MockFrame())

        # Original metadata keys still present
        assert "gesture_type" in obs.metadata
        assert "hands_detected" in obs.metadata
        assert "all_gestures" in obs.metadata
        # _metrics is nested
        assert "_metrics" in obs.metadata
