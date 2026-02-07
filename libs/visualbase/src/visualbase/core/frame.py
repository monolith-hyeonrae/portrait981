"""Frame data class."""

from dataclasses import dataclass

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
