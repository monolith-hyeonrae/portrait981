"""Backend protocol for face.parse."""

from typing import List, Protocol

import numpy as np

from vpx.face_parse.output import FaceParseResult


class FaceParseBackend(Protocol):
    """Protocol for face parsing backends.

    Implementations should be swappable without changing analyzer logic.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models."""
        ...

    def segment(
        self, image: np.ndarray, detected_faces: list
    ) -> List[FaceParseResult]:
        """Segment face regions from detected faces.

        Args:
            image: BGR image (H, W, 3) uint8.
            detected_faces: List of DetectedFace from face.detect.

        Returns:
            List of FaceParseResult, one per face.
        """
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


__all__ = ["FaceParseBackend"]
