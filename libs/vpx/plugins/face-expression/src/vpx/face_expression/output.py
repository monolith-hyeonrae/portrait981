"""Output type for expression analyzer."""

from dataclasses import dataclass, field
from typing import List

from vpx.face_detect.types import FaceObservation


@dataclass
class ExpressionOutput:
    """Output from ExpressionAnalyzer.

    Attributes:
        faces: List of faces with expression data added.
    """
    faces: List[FaceObservation] = field(default_factory=list)


__all__ = ["ExpressionOutput"]
