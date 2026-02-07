import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""Extractors for moment detection.

Each extractor can be imported independently to avoid dependency conflicts
when running in isolated worker environments.

Usage:
    # Import only what you need (lazy loading)
    from vpx.face import FaceExtractor
    from vpx.pose import PoseExtractor
    from vpx.gesture import GestureExtractor

    # Or import base types (always available)
    from facemoment.moment_detector.extractors import BaseExtractor, Observation

    # Import type definitions
    from visualpath.extractors.types import KeypointIndex, GestureType
"""

# Base types - always available (no ML dependencies)
from visualpath.extractors.base import (
    Module,
    BaseExtractor,  # Alias for Module
    Observation,
    FaceObservation,
    IsolationLevel,
)
from visualpath.extractors.types import (
    KeypointIndex,
    HandLandmarkIndex,
    GestureType,
    COCO_KEYPOINT_NAMES,
)
from facemoment.moment_detector.extractors.dummy import DummyExtractor
from facemoment.moment_detector.extractors.quality import QualityExtractor
from visualpath.extractors.outputs import (
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
    "BaseExtractor",
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
    # Extractors (always available)
    "DummyExtractor",
    "QualityExtractor",
    # Lazy imports (import directly from submodule)
    # "FaceExtractor",           # from vpx.face import FaceExtractor (composite)
    # "FaceDetectionExtractor",  # from vpx.face_detect import FaceDetectionExtractor
    # "ExpressionExtractor",     # from vpx.expression import ExpressionExtractor
    # "PoseExtractor",           # from vpx.pose import PoseExtractor
    # "GestureExtractor",        # from vpx.gesture import GestureExtractor
]


def __getattr__(name: str):
    """Lazy import for ML-dependent extractors."""
    if name == "FaceExtractor":
        from vpx.face import FaceExtractor
        return FaceExtractor
    elif name == "FaceDetectionExtractor":
        from vpx.face_detect import FaceDetectionExtractor
        return FaceDetectionExtractor
    elif name == "ExpressionExtractor":
        from vpx.expression import ExpressionExtractor
        return ExpressionExtractor
    elif name == "FaceClassifierExtractor":
        from facemoment.moment_detector.extractors.face_classifier import FaceClassifierExtractor
        return FaceClassifierExtractor
    elif name == "PoseExtractor":
        from vpx.pose import PoseExtractor
        return PoseExtractor
    elif name == "GestureExtractor":
        from vpx.gesture import GestureExtractor
        return GestureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
