"""Frame sampling utilities."""

from collections.abc import Iterator
from typing import Optional

import cv2
import numpy as np

from visualbase.core.frame import Frame
from visualbase.core.timestamp import NS_PER_SECOND
from visualbase.sources.base import BaseSource

# Tolerance for floating-point PTS imprecision from video decoders.
# OpenCV CAP_PROP_POS_MSEC returns a float; int(ms * 1e6) truncation
# can produce values slightly below the mathematical timestamp,
# causing the sampler to skip boundary frames (e.g. 30fps→10fps
# yields every 4th frame instead of every 3rd).  1 ms is safe for
# any source frame rate ≥ 2 fps.
_PTS_TOLERANCE_NS = 1_000_000  # 1 ms


class Sampler:
    """Samples frames from a source at specified fps and resolution.

    Args:
        source: Video source to sample from
        fps: Target frames per second (0 for original fps)
        resolution: Target (width, height) or None for original resolution
    """

    def __init__(
        self,
        source: BaseSource,
        fps: int = 0,
        resolution: Optional[tuple[int, int]] = None,
    ):
        self._source = source
        self._fps = fps
        self._resolution = resolution
        self._frame_interval_ns = (
            NS_PER_SECOND // fps if fps > 0 else 0
        )
        self._next_frame_time_ns: int = 0
        self._output_frame_id: int = 0

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        while True:
            frame = self._source.read()
            if frame is None:
                raise StopIteration

            # FPS sampling: skip frames until we reach the next target time.
            # Tolerance handles floating-point imprecision in PTS from
            # video decoders (OpenCV CAP_PROP_POS_MSEC → pts_to_ns).
            if self._frame_interval_ns > 0:
                if frame.t_src_ns + _PTS_TOLERANCE_NS < self._next_frame_time_ns:
                    continue
                self._next_frame_time_ns = (
                    frame.t_src_ns + self._frame_interval_ns
                )

            # Resize if needed
            output_data = frame.data
            if self._resolution is not None:
                target_w, target_h = self._resolution
                if frame.width != target_w or frame.height != target_h:
                    output_data = cv2.resize(
                        frame.data,
                        (target_w, target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

            output_frame = Frame.from_array(
                data=output_data,
                frame_id=self._output_frame_id,
                t_src_ns=frame.t_src_ns,
            )
            self._output_frame_id += 1
            return output_frame

    def reset(self) -> None:
        """Reset sampler state for reuse."""
        self._next_frame_time_ns = 0
        self._output_frame_id = 0
