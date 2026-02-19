"""Backend protocol definitions for face AU detection."""

from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import numpy as np

from vpx.face_detect.backends.base import DetectedFace

# DISFA 12 AU names in canonical order
AU_NAMES = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
    "AU12", "AU15", "AU17", "AU20", "AU25", "AU26",
]


@dataclass
class FaceAUResult:
    """Result from AU detection backend for a single face.

    Attributes:
        au_intensities: AU name -> intensity (0-5 scale, DISFA convention).
        au_presence: AU name -> binary presence (intensity >= 1.0).
    """

    au_intensities: Dict[str, float] = field(default_factory=dict)
    au_presence: Dict[str, bool] = field(default_factory=dict)


class FaceAUBackend(Protocol):
    """Protocol for face Action Unit detection backends."""

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models."""
        ...

    def analyze(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[FaceAUResult]:
        """Detect AU intensities for each face."""
        ...

    def cleanup(self) -> None:
        """Release resources."""
        ...


__all__ = ["AU_NAMES", "FaceAUResult", "FaceAUBackend"]
