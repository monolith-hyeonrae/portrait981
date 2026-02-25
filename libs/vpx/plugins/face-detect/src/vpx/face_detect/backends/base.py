"""Backend protocol definitions for face detection."""

from dataclasses import dataclass
from typing import Protocol, List, Optional
import numpy as np


@dataclass
class DetectedFace:
    """Result from face detection backend.

    Attributes:
        bbox: Bounding box (x, y, width, height) in pixels.
        confidence: Detection confidence [0, 1].
        landmarks: Optional facial landmarks (5-point or 68-point).
        yaw: Head yaw angle in degrees (if available).
        pitch: Head pitch angle in degrees (if available).
        roll: Head roll angle in degrees (if available).
    """

    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    confidence: float
    landmarks: Optional[np.ndarray] = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    embedding: Optional[np.ndarray] = None  # ArcFace 512D (L2-normalized)
    face_id: int = 0


class FaceDetectionBackend(Protocol):
    """Protocol for face detection backends.

    Implementations should be swappable without changing analyzer logic.
    Examples: InsightFace SCRFD, YOLOv11-Face, RetinaFace.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models."""
        ...

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image."""
        ...

    def detect_batch(self, images: List[np.ndarray]) -> List[List[DetectedFace]]:
        """Detect faces in a batch of images.

        Default implementation calls detect() sequentially.
        Override for GPU batch inference optimization.

        Args:
            images: List of BGR images as numpy arrays (H, W, 3).

        Returns:
            List of detection results, one per image.
        """
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


__all__ = ["DetectedFace", "FaceDetectionBackend"]
