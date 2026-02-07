"""Camera-based video source using OpenCV."""

import time
import logging
from typing import Optional

import cv2

from visualbase.core.frame import Frame
from visualbase.sources.base import BaseSource

logger = logging.getLogger(__name__)


class CameraSource(BaseSource):
    """Video source from a USB/local camera.

    Uses monotonic clock for timestamps to ensure consistent timing
    independent of wall clock changes. This is important for real-time
    streaming where analysis may have variable latency.

    Args:
        device_id: Camera device ID (default: 0 for first camera).
        width: Requested frame width (optional, camera default if None).
        height: Requested frame height (optional, camera default if None).
        fps: Requested frames per second (optional, camera default if None).

    Example:
        >>> source = CameraSource(device_id=0)
        >>> with source:
        ...     while True:
        ...         frame = source.read()
        ...         if frame is None:
        ...             break
        ...         process(frame.data)
    """

    def __init__(
        self,
        device_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ):
        self._device_id = device_id
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._fps: float = 30.0  # Default fallback
        self._width: int = 0
        self._height: int = 0

        # Reference time for monotonic timestamps
        self._start_time_ns: int = 0

    def open(self) -> None:
        """Open the camera device.

        Raises:
            IOError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open camera device: {self._device_id}")

        # Apply requested settings
        if self._requested_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        if self._requested_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        if self._requested_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        # Read actual settings from camera
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0  # Fallback if camera doesn't report FPS

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize timing
        self._frame_id = 0
        self._start_time_ns = time.monotonic_ns()

        logger.info(
            f"Camera opened: device={self._device_id}, "
            f"{self._width}x{self._height} @ {self._fps:.1f}fps"
        )

    def read(self) -> Optional[Frame]:
        """Read the next frame from the camera.

        Uses monotonic clock for timestamps to ensure consistent timing
        even if system wall clock changes.

        Returns:
            Frame object with monotonic timestamp, or None if read failed.

        Raises:
            RuntimeError: If camera is not opened.
        """
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        # Capture timestamp before reading
        t_capture_ns = time.monotonic_ns() - self._start_time_ns

        ret, data = self._cap.read()
        if not ret:
            return None

        frame = Frame.from_array(
            data=data,
            frame_id=self._frame_id,
            t_src_ns=t_capture_ns,
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        """Close the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Camera closed: device={self._device_id}")

    def seek(self, t_ns: int) -> bool:
        """Cameras do not support seeking.

        Args:
            t_ns: Target timestamp (ignored).

        Returns:
            Always False - cameras are not seekable.
        """
        return False

    @property
    def is_seekable(self) -> bool:
        """Cameras are not seekable."""
        return False

    @property
    def fps(self) -> float:
        """Camera frames per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def device_id(self) -> int:
        """Camera device ID."""
        return self._device_id

    @property
    def is_opened(self) -> bool:
        """Whether the camera is currently opened."""
        return self._cap is not None and self._cap.isOpened()
