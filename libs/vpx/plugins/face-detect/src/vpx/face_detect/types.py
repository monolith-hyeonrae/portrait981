"""Face detection domain types."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class FaceObservation:
    """Observation for a single detected face.

    Attributes:
        face_id: Unique identifier for tracking.
        confidence: Detection confidence [0, 1].
        bbox: Bounding box (x, y, width, height) normalized [0, 1].
        inside_frame: Whether face is fully inside frame.
        yaw: Head yaw angle in degrees.
        pitch: Head pitch angle in degrees.
        roll: Head roll angle in degrees.
        area_ratio: Face area as ratio of frame area.
        center_distance: Normalized distance from frame center.
        expression: Expression intensity [0, 1].
        signals: Additional per-face signals.
    """

    face_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x, y, w, h normalized
    inside_frame: bool = True
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    area_ratio: float = 0.0
    center_distance: float = 0.0
    expression: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # ArcFace 512D (L2-normalized)


__all__ = ["FaceObservation"]
