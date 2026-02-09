"""Output type for face detection analyzer."""

from dataclasses import dataclass, field
from typing import List

from vpx.face_detect.types import FaceObservation
from vpx.face_detect.backends.base import DetectedFace


@dataclass
class FaceDetectOutput:
    """Output from FaceDetectionAnalyzer.

    Attributes:
        faces: List of detected faces with normalized bboxes and head pose.
        detected_faces: Raw DetectedFace objects (for expression analysis).
        image_size: Original image size as (width, height).
    """
    faces: List[FaceObservation] = field(default_factory=list)
    detected_faces: List[DetectedFace] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)


__all__ = ["FaceDetectOutput"]
