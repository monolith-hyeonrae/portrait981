"""Face classifier module - classifies detected faces by role.

Classifies faces detected by face.detect into roles:
- main: Primary subject (driver/main person)
- passenger: Secondary subject (co-passenger)
- transient: Temporarily detected face (passing by)
- noise: False detection or low-quality face
"""

from momentscan.algorithm.analyzers.face_classifier.analyzer import FaceClassifierAnalyzer
from momentscan.algorithm.analyzers.face_classifier.types import ClassifiedFace
from momentscan.algorithm.analyzers.face_classifier.output import FaceClassifierOutput

__all__ = [
    "FaceClassifierAnalyzer",
    "ClassifiedFace",
    "FaceClassifierOutput",
]
