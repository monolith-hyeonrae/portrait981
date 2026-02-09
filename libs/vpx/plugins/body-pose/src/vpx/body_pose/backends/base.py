"""Backend protocol definitions for pose estimation."""

from dataclasses import dataclass
from typing import Protocol, List, Optional
import numpy as np


@dataclass
class PoseKeypoints:
    """Result from pose estimation backend.

    Attributes:
        keypoints: Array of shape (N, 3) with (x, y, confidence) per keypoint.
        keypoint_names: Names of keypoints in order.
        person_id: Optional person/track ID.
        bbox: Optional person bounding box.
        confidence: Overall detection confidence.
    """

    keypoints: np.ndarray  # Shape: (N, 3) - x, y, conf
    keypoint_names: List[str]
    person_id: Optional[int] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 1.0


class PoseBackend(Protocol):
    """Protocol for pose estimation backends.

    Implementations extract body keypoints for gesture analysis.
    Examples: YOLOv8-Pose, MediaPipe, OpenPose.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models."""
        ...

    def detect(self, image: np.ndarray) -> List[PoseKeypoints]:
        """Detect poses in an image."""
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


__all__ = ["PoseKeypoints", "PoseBackend"]
