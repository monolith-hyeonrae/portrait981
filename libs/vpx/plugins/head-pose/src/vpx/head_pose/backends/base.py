"""Backend protocol definitions for head pose estimation."""

from typing import List, Protocol

import numpy as np

from vpx.face_detect.backends.base import DetectedFace
from vpx.head_pose.types import HeadPoseEstimate


class HeadPoseBackend(Protocol):
    """Protocol for head pose estimation backends."""

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models."""
        ...

    def estimate(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[HeadPoseEstimate]:
        """Estimate head pose for each face."""
        ...

    def cleanup(self) -> None:
        """Release resources."""
        ...


__all__ = ["HeadPoseBackend"]
