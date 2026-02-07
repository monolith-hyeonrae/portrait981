"""RTSP/IP camera video source using OpenCV."""

import time
import logging
import threading
from queue import Queue, Empty
from typing import Optional

import cv2

from visualbase.core.frame import Frame
from visualbase.sources.base import BaseSource
from visualbase.sources.decoder import configure_capture

logger = logging.getLogger(__name__)


class RTSPSource(BaseSource):
    """Video source from an RTSP/IP camera stream.

    Supports RTSP, HTTP, and other network stream URLs that OpenCV can handle.
    Uses a background thread to continuously read frames, preventing buffer
    buildup and ensuring low-latency access to the latest frames.

    Uses monotonic clock for timestamps to ensure consistent timing.

    Args:
        url: RTSP or stream URL (e.g., "rtsp://192.168.1.100:554/stream").
        buffer_size: Maximum frames to buffer (default: 2, keeps latest frames).
        timeout_sec: Connection timeout in seconds (default: 10).
        reconnect: Whether to auto-reconnect on connection loss (default: True).
        reconnect_delay_sec: Delay between reconnection attempts (default: 5).
        decoder: Video decoder to use ("auto", "nvdec", "vaapi", "cpu").

    Example:
        >>> source = RTSPSource("rtsp://192.168.1.100:554/stream")
        >>> with source:
        ...     while True:
        ...         frame = source.read()
        ...         if frame is None:
        ...             break
        ...         process(frame.data)
    """

    def __init__(
        self,
        url: str,
        buffer_size: int = 2,
        timeout_sec: float = 10.0,
        reconnect: bool = True,
        reconnect_delay_sec: float = 5.0,
        decoder: str = "auto",
    ):
        self._url = url
        self._buffer_size = buffer_size
        self._timeout_sec = timeout_sec
        self._reconnect = reconnect
        self._reconnect_delay_sec = reconnect_delay_sec
        self._decoder = decoder
        self._actual_decoder: str = "cpu"

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._fps: float = 30.0  # Default fallback
        self._width: int = 0
        self._height: int = 0

        # Reference time for monotonic timestamps
        self._start_time_ns: int = 0

        # Background thread for continuous reading
        self._frame_queue: Queue = Queue(maxsize=buffer_size)
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._connected: bool = False

    def open(self) -> None:
        """Open the RTSP stream.

        Starts a background thread that continuously reads frames to prevent
        buffer buildup.

        Raises:
            IOError: If the stream cannot be opened.
        """
        self._connect()

        # Start background reader thread
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name=f"RTSPReader-{self._url[:30]}",
        )
        self._reader_thread.start()

        logger.info(
            f"RTSP stream opened: {self._url}, "
            f"{self._width}x{self._height} @ {self._fps:.1f}fps"
        )

    def _connect(self) -> None:
        """Establish connection to the RTSP stream."""
        # Use decoder configuration for hardware acceleration
        self._cap, self._actual_decoder = configure_capture(
            self._url,
            decoder=self._decoder,
        )

        # Configure for low latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise IOError(f"Failed to open RTSP stream: {self._url}")

        # Read actual settings from stream
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0  # Fallback if stream doesn't report FPS

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize timing
        self._frame_id = 0
        self._start_time_ns = time.monotonic_ns()
        self._connected = True

    def _reader_loop(self) -> None:
        """Background thread that continuously reads frames."""
        consecutive_failures = 0
        max_failures = 10

        while not self._stop_event.is_set():
            if self._cap is None or not self._connected:
                if self._reconnect:
                    time.sleep(self._reconnect_delay_sec)
                    try:
                        self._connect()
                        consecutive_failures = 0
                        logger.info(f"RTSP reconnected: {self._url}")
                    except IOError as e:
                        logger.warning(f"RTSP reconnect failed: {e}")
                        continue
                else:
                    break

            try:
                # Capture timestamp before reading
                t_capture_ns = time.monotonic_ns() - self._start_time_ns

                ret, data = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning(f"RTSP stream lost: {self._url}")
                        self._connected = False
                        if self._cap is not None:
                            self._cap.release()
                            self._cap = None
                    continue

                consecutive_failures = 0

                frame = Frame.from_array(
                    data=data,
                    frame_id=self._frame_id,
                    t_src_ns=t_capture_ns,
                )
                self._frame_id += 1

                # Non-blocking put - drop oldest if full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass

                self._frame_queue.put(frame)

            except Exception as e:
                logger.error(f"RTSP reader error: {e}")
                consecutive_failures += 1

    def read(self) -> Optional[Frame]:
        """Read the next frame from the stream.

        Returns the most recently captured frame from the buffer.

        Returns:
            Frame object with monotonic timestamp, or None if no frame available.

        Raises:
            RuntimeError: If stream is not opened.
        """
        if self._reader_thread is None:
            raise RuntimeError("Stream not opened. Call open() first.")

        try:
            # Use timeout to allow checking for stream end
            return self._frame_queue.get(timeout=1.0)
        except Empty:
            if not self._connected and not self._reconnect:
                return None
            # Return None but stream may still be alive
            return None

    def close(self) -> None:
        """Close the RTSP stream."""
        self._stop_event.set()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._connected = False

        # Clear the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        logger.info(f"RTSP stream closed: {self._url}")

    def seek(self, t_ns: int) -> bool:
        """RTSP streams do not support seeking.

        Args:
            t_ns: Target timestamp (ignored).

        Returns:
            Always False - streams are not seekable.
        """
        return False

    @property
    def is_seekable(self) -> bool:
        """RTSP streams are not seekable."""
        return False

    @property
    def fps(self) -> float:
        """Stream frames per second."""
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
    def url(self) -> str:
        """Stream URL."""
        return self._url

    @property
    def is_connected(self) -> bool:
        """Whether the stream is currently connected."""
        return self._connected

    @property
    def decoder(self) -> str:
        """Actual decoder being used."""
        return self._actual_decoder
