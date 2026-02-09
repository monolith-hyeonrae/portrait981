"""Type definitions for analyzers.

Contains keypoint indices, gesture types, and other constants used
by multiple analyzers.

These types are now defined in their respective vpx packages and
re-exported here for backwards compatibility.
"""

# Re-exported from vpx packages
from vpx.pose.types import KeypointIndex, COCO_KEYPOINT_NAMES  # noqa: F401
from vpx.gesture.types import HandLandmarkIndex, GestureType  # noqa: F401


__all__ = [
    "KeypointIndex",
    "HandLandmarkIndex",
    "GestureType",
    "COCO_KEYPOINT_NAMES",
]
