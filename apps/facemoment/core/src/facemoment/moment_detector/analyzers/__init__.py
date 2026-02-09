import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""Analyzers for moment detection.

Each analyzer can be imported independently to avoid dependency conflicts
when running in isolated worker environments.

Usage:
    # Import only what you need (lazy loading)
    from vpx.face import FaceAnalyzer
    from vpx.pose import PoseAnalyzer
    from vpx.gesture import GestureAnalyzer

    # Or import base types (always available)
    from facemoment.moment_detector.analyzers import BaseAnalyzer, Observation

    # Import type definitions (canonical)
    from vpx.pose.types import KeypointIndex
    from vpx.gesture.types import GestureType
"""

# Base types - always available (no ML dependencies)
from visualpath.analyzers.base import (
    Module,
    BaseAnalyzer,  # Alias for Module
    Observation,
    IsolationLevel,
)
from vpx.face_detect.types import FaceObservation
from vpx.pose.types import KeypointIndex, COCO_KEYPOINT_NAMES
from vpx.gesture.types import HandLandmarkIndex, GestureType
from facemoment.moment_detector.analyzers.dummy import DummyAnalyzer
from facemoment.moment_detector.analyzers.quality import QualityAnalyzer
from vpx.face_detect.output import FaceDetectOutput
from vpx.expression.output import ExpressionOutput
from vpx.pose.output import PoseOutput
from vpx.gesture.output import GestureOutput
from facemoment.moment_detector.analyzers.quality import QualityOutput
from facemoment.moment_detector.analyzers.face_classifier import ClassifiedFace, FaceClassifierOutput

__all__ = [
    # Base types (always available)
    "BaseAnalyzer",
    "Observation",
    "FaceObservation",
    "IsolationLevel",
    # Type definitions
    "KeypointIndex",
    "HandLandmarkIndex",
    "GestureType",
    "COCO_KEYPOINT_NAMES",
    # Output types (type-safe deps access)
    "FaceDetectOutput",
    "ExpressionOutput",
    "PoseOutput",
    "GestureOutput",
    "QualityOutput",
    "ClassifiedFace",
    "FaceClassifierOutput",
    # Analyzers (always available)
    "DummyAnalyzer",
    "QualityAnalyzer",
    # Lazy imports (import directly from submodule)
    # "FaceAnalyzer",           # from vpx.face import FaceAnalyzer (composite)
    # "FaceDetectionAnalyzer",  # from vpx.face_detect import FaceDetectionAnalyzer
    # "ExpressionAnalyzer",     # from vpx.expression import ExpressionAnalyzer
    # "PoseAnalyzer",           # from vpx.pose import PoseAnalyzer
    # "GestureAnalyzer",        # from vpx.gesture import GestureAnalyzer
]


def __getattr__(name: str):
    """Lazy import for ML-dependent analyzers."""
    if name == "FaceAnalyzer":
        from vpx.face import FaceAnalyzer
        return FaceAnalyzer
    elif name == "FaceDetectionAnalyzer":
        from vpx.face_detect import FaceDetectionAnalyzer
        return FaceDetectionAnalyzer
    elif name == "ExpressionAnalyzer":
        from vpx.expression import ExpressionAnalyzer
        return ExpressionAnalyzer
    elif name == "FaceClassifierAnalyzer":
        from facemoment.moment_detector.analyzers.face_classifier import FaceClassifierAnalyzer
        return FaceClassifierAnalyzer
    elif name == "PoseAnalyzer":
        from vpx.pose import PoseAnalyzer
        return PoseAnalyzer
    elif name == "GestureAnalyzer":
        from vpx.gesture import GestureAnalyzer
        return GestureAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
