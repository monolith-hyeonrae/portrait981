"""Face baseline module - per-face identity baseline profiling.

Tracks area_ratio and position statistics per face identity
using Welford online algorithm for numerically stable mean/std.

depends: ["face.detect", "face.classify"]
"""

from momentscan.algorithm.analyzers.face_baseline.analyzer import FaceBaselineAnalyzer
from momentscan.algorithm.analyzers.face_baseline.output import (
    FaceBaselineProfile,
    FaceBaselineOutput,
)

__all__ = [
    "FaceBaselineAnalyzer",
    "FaceBaselineProfile",
    "FaceBaselineOutput",
]
