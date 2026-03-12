"""Face classifier domain types."""

from dataclasses import dataclass

from vpx.face_detect.types import FaceObservation


@dataclass
class ClassifiedFace:
    """Face with classification info."""
    face: FaceObservation
    role: str  # "main", "passenger", "transient", "noise"
    confidence: float  # Classification confidence
    track_length: int  # Number of consecutive frames tracked
    avg_area: float  # Average area ratio over track
