import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""Backend implementations for feature analysis.

Each backend can be imported independently to avoid dependency conflicts.

Usage:
    # Import base types (always available)
    from visualpath.analyzers.backends import DetectedFace, FaceExpression

    # Import specific backends (requires corresponding dependencies)
    from vpx.face_detect.backends.insightface import InsightFaceSCRFD
    from vpx.expression.backends.hsemotion import HSEmotionBackend
    from vpx.expression.backends.pyfeat import PyFeatBackend
    from vpx.pose.backends.yolo_pose import YOLOPoseBackend
    from vpx.gesture.backends.mediapipe_hands import MediaPipeHandsBackend
"""

# Base types - always available (no ML dependencies)
from visualpath.analyzers.backends.base import (
    FaceDetectionBackend,
    ExpressionBackend,
    PoseBackend,
    HandLandmarkBackend,
    DetectedFace,
    FaceExpression,
    PoseKeypoints,
    HandLandmarks,
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
    # Lazy imports (import directly from submodule)
    # "InsightFaceSCRFD",     # from vpx.face_detect.backends.insightface
    # "PyFeatBackend",        # from vpx.expression.backends.pyfeat
    # "HSEmotionBackend",     # from vpx.expression.backends.hsemotion
    # "YOLOPoseBackend",      # from vpx.pose.backends.yolo_pose
    # "MediaPipeHandsBackend",# from vpx.gesture.backends.mediapipe_hands
]


def __getattr__(name: str):
    """Lazy import for ML-dependent backends."""
    # Face detection backends
    if name == "InsightFaceSCRFD":
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD
        return InsightFaceSCRFD
    # Expression backends
    elif name == "PyFeatBackend":
        from vpx.expression.backends.pyfeat import PyFeatBackend
        return PyFeatBackend
    elif name == "HSEmotionBackend":
        from vpx.expression.backends.hsemotion import HSEmotionBackend
        return HSEmotionBackend
    # Pose backends
    elif name == "YOLOPoseBackend":
        from vpx.pose.backends.yolo_pose import YOLOPoseBackend
        return YOLOPoseBackend
    # Hand backends
    elif name == "MediaPipeHandsBackend":
        from vpx.gesture.backends.mediapipe_hands import MediaPipeHandsBackend
        return MediaPipeHandsBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
