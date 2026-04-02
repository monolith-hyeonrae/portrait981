"""Frame data class."""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from visualbase.types import BGRImage


@dataclass(frozen=True, slots=True)
class Frame:
    """Represents a single video frame.

    Attributes:
        data: BGR image array (H, W, 3)
        frame_id: Monotonically increasing frame identifier
        t_src_ns: Source timestamp in nanoseconds
        width: Frame width in pixels
        height: Frame height in pixels
    """

    data: BGRImage
    frame_id: int
    t_src_ns: int
    width: int
    height: int

    @classmethod
    def from_array(
        cls, data: BGRImage, frame_id: int, t_src_ns: int
    ) -> "Frame":
        """Create a Frame from a BGR array."""
        height, width = data.shape[:2]
        return cls(
            data=data,
            frame_id=frame_id,
            t_src_ns=t_src_ns,
            width=width,
            height=height,
        )

    @classmethod
    def from_image(cls, image: BGRImage, frame_id: int = 0) -> "Frame":
        """Create a Frame from a single BGR image (t_src_ns=0)."""
        height, width = image.shape[:2]
        return cls(data=image, frame_id=frame_id, t_src_ns=0, width=width, height=height)

    def crop(self, x1: int, y1: int, x2: int, y2: int) -> BGRImage:
        """Crop a region from the frame. Returns BGR ndarray (not a Frame).

        Args:
            x1, y1: Top-left corner (pixel coordinates).
            x2, y2: Bottom-right corner (pixel coordinates).
        """
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)
        return self.data[y1:y2, x1:x2]

    def resize(self, width: int, height: int) -> BGRImage:
        """Resize the frame. Returns BGR ndarray.

        Args:
            width: Target width.
            height: Target height.
        """
        return cv2.resize(self.data, (width, height))
