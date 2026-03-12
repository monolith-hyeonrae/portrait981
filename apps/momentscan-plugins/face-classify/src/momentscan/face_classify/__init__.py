"""Face classifier module - classifies detected faces by role.

Classifies faces detected by face.detect into roles:
- main: Primary subject (driver/main person)
- passenger: Secondary subject (co-passenger)
- transient: Temporarily detected face (passing by)
- noise: False detection or low-quality face
"""

from momentscan.face_classify.analyzer import FaceClassifierAnalyzer
from momentscan.face_classify.types import ClassifiedFace
from momentscan.face_classify.output import FaceClassifierOutput

__all__ = [
    "FaceClassifierAnalyzer",
    "ClassifiedFace",
    "FaceClassifierOutput",
]
