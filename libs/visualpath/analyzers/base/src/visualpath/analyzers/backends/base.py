"""Backend protocol definitions for swappable ML backends.

These types are now defined in their respective vpx packages and
re-exported here for backwards compatibility.
"""

# Re-exported from vpx packages
from vpx.face_detect.backends.base import (  # noqa: F401
    DetectedFace,
    FaceDetectionBackend,
)
from vpx.face_expression.backends.base import (  # noqa: F401
    FaceExpression,
    ExpressionBackend,
)
from vpx.body_pose.backends.base import (  # noqa: F401
    PoseKeypoints,
    PoseBackend,
)
from vpx.hand_gesture.backends.base import (  # noqa: F401
    HandLandmarks,
    HandLandmarkBackend,
)


__all__ = [
    # Protocols
    "FaceDetectionBackend",
    "ExpressionBackend",
    "PoseBackend",
    "HandLandmarkBackend",
    # Data classes
    "DetectedFace",
    "FaceExpression",
    "PoseKeypoints",
    "HandLandmarks",
]
