"""Face classifier output dataclass."""

from dataclasses import dataclass, field
from typing import Optional, List

from momentscan.algorithm.analyzers.face_classifier.types import ClassifiedFace


@dataclass
class FaceClassifierOutput:
    """Output from FaceClassifierAnalyzer."""
    faces: List[ClassifiedFace] = field(default_factory=list)
    main_face: Optional[ClassifiedFace] = None
    passenger_faces: List[ClassifiedFace] = field(default_factory=list)
    transient_count: int = 0
    noise_count: int = 0
