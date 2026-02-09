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

    # Import type definitions
    from visualpath.analyzers.types import KeypointIndex, GestureType
"""

# Base types - always available (no ML dependencies)
from visualpath.analyzers.base import (
    Module,
    BaseAnalyzer,  # Alias for Module
    Observation,
    FaceObservation,
    IsolationLevel,
)
from visualpath.analyzers.types import (
    KeypointIndex,
    HandLandmarkIndex,
    GestureType,
    COCO_KEYPOINT_NAMES,
)
from facemoment.moment_detector.analyzers.dummy import DummyAnalyzer
from facemoment.moment_detector.analyzers.quality import QualityAnalyzer
from visualpath.analyzers.outputs import (
    FaceDetectOutput,
    ExpressionOutput,
    PoseOutput,
    GestureOutput,
    QualityOutput,
    ClassifiedFaceInfo,
    FaceClassifierOutput,
)

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
    "ClassifiedFaceInfo",
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
