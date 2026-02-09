"""Gesture recognition domain types."""

from enum import Enum


class HandLandmarkIndex:
    """MediaPipe hand landmark indices.

    21 landmarks per hand as defined by MediaPipe Hands.

    Example:
        >>> lms = hand.landmarks
        >>> thumb_tip = lms[HandLandmarkIndex.THUMB_TIP]
        >>> index_tip = lms[HandLandmarkIndex.INDEX_FINGER_TIP]
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class GestureType(Enum):
    """Recognized gesture types.

    Used by GestureAnalyzer to classify hand gestures.

    Example:
        >>> gesture, confidence = analyzer._classify_gesture(hand)
        >>> if gesture == GestureType.V_SIGN:
        ...     print("Peace sign detected!")
    """

    NONE = "none"
    V_SIGN = "v_sign"
    THUMBS_UP = "thumbs_up"
    OK_SIGN = "ok_sign"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINTING = "pointing"


__all__ = ["HandLandmarkIndex", "GestureType"]
