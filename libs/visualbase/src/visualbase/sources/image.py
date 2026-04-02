"""Single-image source — wraps a BGR ndarray as a 1-frame source."""

from typing import Optional

import numpy as np

from visualbase.core.frame import Frame
from visualbase.sources.base import BaseSource


class ImageSource(BaseSource):
    """Video source from a single BGR image.

    Produces exactly 1 frame, then EOF.

    Args:
        image: BGR ndarray (H, W, 3).
        frame_id: Frame identifier (default 0).
    """

    def __init__(self, image: np.ndarray, frame_id: int = 0):
        self._image = image
        self._frame_id = frame_id
        self._consumed = False
        h, w = image.shape[:2]
        self._width = w
        self._height = h

    def open(self) -> None:
        self._consumed = False

    def read(self) -> Optional[Frame]:
        if self._consumed:
            return None
        self._consumed = True
        return Frame.from_image(self._image, frame_id=self._frame_id)

    def close(self) -> None:
        pass

    def seek(self, t_ns: int) -> bool:
        self._consumed = False
        return True

    @property
    def is_seekable(self) -> bool:
        return True

    @property
    def fps(self) -> float:
        return 1.0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
