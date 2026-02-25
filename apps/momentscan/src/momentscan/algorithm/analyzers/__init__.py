import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""Analyzers for moment detection.

Each analyzer can be imported independently to avoid dependency conflicts
when running in isolated worker environments.

Usage:
    # Import only what you need (lazy loading)
    from vpx.face_expression import ExpressionAnalyzer
    from vpx.body_pose import PoseAnalyzer
    from vpx.hand_gesture import GestureAnalyzer

    # Or import base types (always available)
    from momentscan.algorithm.analyzers import BaseAnalyzer, Observation

    # Import type definitions (canonical)
    from vpx.body_pose.types import KeypointIndex
    from vpx.hand_gesture.types import GestureType
"""

# Base types - always available (no ML dependencies)
from vpx.sdk import (
    Module,
    BaseAnalyzer,  # Alias for Module
    Observation,
)
from visualpath.core import IsolationLevel
from vpx.face_detect.types import FaceObservation
from vpx.body_pose.types import KeypointIndex, COCO_KEYPOINT_NAMES
from vpx.hand_gesture.types import HandLandmarkIndex, GestureType
from momentscan.algorithm.analyzers.quality import QualityAnalyzer
from vpx.face_detect.output import FaceDetectOutput
from vpx.face_expression.output import ExpressionOutput
from vpx.body_pose.output import PoseOutput
from vpx.hand_gesture.output import GestureOutput
from momentscan.algorithm.analyzers.quality import QualityOutput
from momentscan.algorithm.analyzers.face_classifier import ClassifiedFace, FaceClassifierOutput
from momentscan.algorithm.analyzers.frame_scoring import FrameScoringAnalyzer, ScoreResult
from momentscan.algorithm.analyzers.face_baseline import FaceBaselineAnalyzer, FaceBaselineOutput

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
    "QualityAnalyzer",
    "FrameScoringAnalyzer",
    "ScoreResult",
    "FaceBaselineAnalyzer",
    "FaceBaselineOutput",
    # Lazy imports (import directly from submodule)
    # "FaceDetectionAnalyzer",  # from vpx.face_detect import FaceDetectionAnalyzer
    # "ExpressionAnalyzer",     # from vpx.face_expression import ExpressionAnalyzer
    # "PoseAnalyzer",           # from vpx.body_pose import PoseAnalyzer
    # "GestureAnalyzer",        # from vpx.hand_gesture import GestureAnalyzer
]


def __getattr__(name: str):
    """Lazy import for ML-dependent analyzers."""
    if name == "FaceDetectionAnalyzer":
        from vpx.face_detect import FaceDetectionAnalyzer
        return FaceDetectionAnalyzer
    elif name == "ExpressionAnalyzer":
        from vpx.face_expression import ExpressionAnalyzer
        return ExpressionAnalyzer
    elif name == "FaceClassifierAnalyzer":
        from momentscan.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer
        return FaceClassifierAnalyzer
    elif name == "PoseAnalyzer":
        from vpx.body_pose import PoseAnalyzer
        return PoseAnalyzer
    elif name == "GestureAnalyzer":
        from vpx.hand_gesture import GestureAnalyzer
        return GestureAnalyzer
    elif name == "FaceBaselineAnalyzer":
        from momentscan.algorithm.analyzers.face_baseline import FaceBaselineAnalyzer
        return FaceBaselineAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
