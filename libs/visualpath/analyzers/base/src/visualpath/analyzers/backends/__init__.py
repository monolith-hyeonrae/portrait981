import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

# Re-exported from vpx packages for backwards compatibility
from vpx.face_detect.backends.base import FaceDetectionBackend, DetectedFace
from vpx.face_expression.backends.base import ExpressionBackend, FaceExpression
from vpx.body_pose.backends.base import PoseBackend, PoseKeypoints
from vpx.hand_gesture.backends.base import HandLandmarkBackend, HandLandmarks

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
