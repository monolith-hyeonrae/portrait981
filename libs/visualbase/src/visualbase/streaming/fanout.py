"""Proxy stream fan-out for distributing video to multiple extractors.

The ProxyFanout class takes frames from a single source and distributes
them to multiple FIFO outputs, each with configurable resolution and FPS.

Architecture:
    Source (4K@30fps)
      │
      ├─→ FIFO 1 (Face): 640x480@10fps
      ├─→ FIFO 2 (Pose): 640x480@10fps
      └─→ FIFO 3 (Quality): 320x240@5fps
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
import logging

import cv2

from visualbase.core.frame import Frame
from visualbase.core.sampler import Sampler
from visualbase.ipc.fifo import FIFOVideoWriter

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Configuration for a proxy output stream.

    Args:
        name: Identifier for this stream (e.g., "face", "pose").
        fifo_path: Path to create the FIFO.
        width: Output width in pixels.
        height: Output height in pixels.
        fps: Target frames per second.
        jpeg_quality: JPEG encoding quality (0-100).
    """

    name: str
    fifo_path: str
    width: int
    height: int
    fps: float
    jpeg_quality: int = 85


class ProxyOutput:
    """A single proxy output stream.

    Handles downsampling and writing to a FIFO.
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._writer = FIFOVideoWriter(config.fifo_path, config.jpeg_quality)
        self._sampler = Sampler(target_fps=config.fps)
        self._connected = False
        self._connect_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start_async(self) -> None:
        """Start connecting to the FIFO in background.

        The FIFO open() blocks until a reader connects.
        """
        self._shutdown.clear()

        def connect():
            try:
                self._writer.open()
                self._connected = True
                logger.info(f"Proxy output '{self.config.name}' connected")
            except Exception as e:
                if not self._shutdown.is_set():
                    logger.error(f"Proxy '{self.config.name}' connect error: {e}")

        self._connect_thread = threading.Thread(target=connect, daemon=True)
        self._connect_thread.start()

    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """Wait for a reader to connect.

        Returns:
            True if connected within timeout.
        """
        if self._connect_thread is None:
            return False
        self._connect_thread.join(timeout)
        return self._connected

    def write(self, frame: Frame) -> bool:
        """Write a frame if connected and FPS sampling allows.

        Args:
            frame: Source frame to write.

        Returns:
            True if frame was written.
        """
        if not self._connected:
            return False

        # Check FPS sampling
        if not self._sampler.should_process(frame.t_src_ns):
            return False

        # Resize if needed
        if frame.width != self.config.width or frame.height != self.config.height:
            resized = cv2.resize(
                frame.data,
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_LINEAR,
            )
            frame = Frame.from_array(
                data=resized,
                frame_id=frame.frame_id,
                t_src_ns=frame.t_src_ns,
            )

        return self._writer.write(frame)

    def stop(self) -> None:
        """Stop the output and clean up."""
        self._shutdown.set()
        self._connected = False
        self._writer.close()

        if self._connect_thread and self._connect_thread.is_alive():
            # Can't interrupt blocking open(), but daemon thread will die
            pass

    @property
    def is_connected(self) -> bool:
        """Check if a reader is connected."""
        return self._connected


class ProxyFanout:
    """Distribute video frames to multiple proxy outputs.

    Takes frames from a stream and distributes them to multiple FIFO
    outputs, each with potentially different resolution and FPS.

    Args:
        configs: List of proxy output configurations.

    Example:
        >>> fanout = ProxyFanout([
        ...     ProxyConfig("face", "/tmp/vid_face.mjpg", 640, 480, 10),
        ...     ProxyConfig("pose", "/tmp/vid_pose.mjpg", 640, 480, 10),
        ...     ProxyConfig("quality", "/tmp/vid_quality.mjpg", 320, 240, 5),
        ... ])
        >>> fanout.start()
        >>> for frame in source.stream():
        ...     fanout.write(frame)
        >>> fanout.stop()
    """

    def __init__(self, configs: List[ProxyConfig]):
        self._configs = configs
        self._outputs: List[ProxyOutput] = []
        self._running = False
        self._stats = {cfg.name: {"written": 0, "dropped": 0} for cfg in configs}
        self._on_frame_callback: Optional[Callable[[Frame], None]] = None

    def start(self, wait_for_readers: bool = False, timeout: float = 10.0) -> None:
        """Start all proxy outputs.

        Args:
            wait_for_readers: If True, wait for all readers to connect.
            timeout: Timeout for waiting (only if wait_for_readers=True).
        """
        self._outputs = [ProxyOutput(cfg) for cfg in self._configs]

        for output in self._outputs:
            output.start_async()

        if wait_for_readers:
            for output in self._outputs:
                if not output.wait_for_connection(timeout):
                    logger.warning(
                        f"Timeout waiting for '{output.config.name}' reader"
                    )

        self._running = True
        logger.info(f"ProxyFanout started with {len(self._outputs)} outputs")

    def write(self, frame: Frame) -> int:
        """Write a frame to all connected outputs.

        Args:
            frame: Frame to distribute.

        Returns:
            Number of outputs that received the frame.
        """
        if not self._running:
            return 0

        # Call frame callback if set (for RingBuffer integration)
        if self._on_frame_callback:
            self._on_frame_callback(frame)

        written = 0
        for output in self._outputs:
            if output.write(frame):
                self._stats[output.config.name]["written"] += 1
                written += 1
            else:
                self._stats[output.config.name]["dropped"] += 1

        return written

    def stop(self) -> None:
        """Stop all outputs and clean up."""
        self._running = False

        for output in self._outputs:
            output.stop()

        self._outputs.clear()
        logger.info("ProxyFanout stopped")

    def set_frame_callback(self, callback: Callable[[Frame], None]) -> None:
        """Set a callback to be called for each frame.

        Useful for integrating with RingBuffer.

        Args:
            callback: Function to call with each frame.
        """
        self._on_frame_callback = callback

    @property
    def is_running(self) -> bool:
        """Check if fanout is running."""
        return self._running

    @property
    def connected_count(self) -> int:
        """Number of connected outputs."""
        return sum(1 for o in self._outputs if o.is_connected)

    def get_stats(self) -> dict:
        """Get write statistics for each output."""
        return dict(self._stats)

    def __enter__(self) -> "ProxyFanout":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
