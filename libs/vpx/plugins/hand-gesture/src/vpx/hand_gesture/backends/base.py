"""Backend protocol definitions for hand landmark detection."""

from dataclasses import dataclass
from typing import Protocol, List
import numpy as np


@dataclass
class HandLandmarks:
    """Result from hand landmark detection backend.

    MediaPipe Hands provides 21 landmarks per hand.

    Attributes:
        landmarks: Array of shape (21, 3) with (x, y, z) per landmark.
            Coordinates are normalized [0, 1].
        handedness: "Left" or "Right".
        confidence: Detection confidence [0, 1].
    """

    landmarks: np.ndarray  # Shape: (21, 3) - x, y, z normalized
    handedness: str  # "Left" or "Right"
    confidence: float = 1.0


class HandLandmarkBackend(Protocol):
    """Protocol for hand landmark detection backends.

    Implementations detect hand landmarks for gesture classification.
    Examples: MediaPipe Hands.
    """

    def initialize(self, device: str = "cpu") -> None:
        """Initialize the backend and load models."""
        ...

    def detect(self, image: np.ndarray) -> List[HandLandmarks]:
        """Detect hands and their landmarks in an image."""
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


__all__ = ["HandLandmarks", "HandLandmarkBackend"]
