"""Daemon mode for continuous video streaming via ZMQ.

The daemon reads from a video source and publishes frames via ZMQ PUB/SUB,
allowing dynamic subscriber attachment/detachment without stopping the stream.

Example:
    >>> # Start daemon
    >>> visualbase daemon --source 0 --pub tcp://*:5555

    >>> # Programmatic usage
    >>> from visualbase.daemon import VideoDaemon
    >>> daemon = VideoDaemon(
    ...     source=CameraSource(0),
    ...     pub_address="tcp://*:5555",
    ... )
    >>> daemon.run()  # Blocks until SIGINT/SIGTERM
"""

import logging
import signal
import threading
import time
from typing import Optional

from visualbase.core.frame import Frame
from visualbase.sources.base import BaseSource
from visualbase.ipc.interfaces import VideoWriter
from visualbase.ipc.factory import TransportFactory

logger = logging.getLogger(__name__)


class VideoDaemon:
    """Video streaming daemon using ZMQ PUB/SUB.

    Reads frames from a video source and publishes them via ZMQ,
    allowing multiple subscribers to connect/disconnect dynamically.

    Args:
        source: Video source to read from.
        pub_address: ZMQ PUB address (e.g., "tcp://*:5555").
        fps: Target FPS for publishing. 0 = source FPS.
        hwm: ZMQ high water mark (max queued frames).
        publisher: Optional custom VideoWriter (for testing).
    """

    def __init__(
        self,
        source: BaseSource,
        pub_address: str,
        fps: int = 0,
        hwm: int = 2,
        publisher: Optional[VideoWriter] = None,
    ):
        self._source = source
        self._pub_address = pub_address
        self._target_fps = fps
        self._hwm = hwm

        # Create publisher (or use provided one for testing)
        if publisher is not None:
            self._publisher = publisher
        else:
            self._publisher = TransportFactory.create_video_writer(
                "zmq",
                pub_address,
                hwm=hwm,
            )

        # State
        self._running = False
        self._frame_count = 0
        self._start_time: Optional[float] = None

    @property
    def source(self) -> BaseSource:
        """Get the video source."""
        return self._source

    @property
    def pub_address(self) -> str:
        """Get the ZMQ publish address."""
        return self._pub_address

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Get number of frames published."""
        return self._frame_count

    def get_stats(self) -> dict:
        """Get daemon statistics.

        Returns:
            Dict with frame_count, elapsed_sec, fps.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        return {
            "frame_count": self._frame_count,
            "elapsed_sec": elapsed,
            "fps": fps,
        }

    def run(self) -> None:
        """Run the daemon until stopped.

        Blocks until SIGINT/SIGTERM is received.
        """
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()

        # Setup signal handlers (only in main thread)
        in_main_thread = threading.current_thread() is threading.main_thread()
        original_sigint = None
        original_sigterm = None
        if in_main_thread:
            original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)

        try:
            self._source.open()
            self._publisher.open()

            logger.info(
                f"Daemon started: {self._source.width}x{self._source.height} "
                f"@ {self._source.fps:.1f}fps → {self._pub_address}"
            )

            # Calculate frame interval
            source_fps = self._source.fps or 30
            target_fps = self._target_fps if self._target_fps > 0 else source_fps
            frame_interval = 1.0 / target_fps

            next_frame_time = time.time()

            consecutive_none = 0
            max_consecutive_none = 10  # Stop after 10 consecutive None reads

            while self._running:
                frame = self._source.read()
                if frame is None:
                    # End of source or temporary error
                    consecutive_none += 1
                    if consecutive_none >= max_consecutive_none:
                        # Source appears exhausted
                        break
                    if not self._running:
                        break
                    # For live sources, brief pause before retry
                    time.sleep(0.01)
                    continue

                consecutive_none = 0  # Reset on successful read

                # Publish frame
                if self._publisher.write(frame):
                    self._frame_count += 1

                # FPS limiting: sleep until next frame is due
                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Daemon error: {e}")
            raise
        finally:
            # Restore signal handlers (only if we set them)
            if in_main_thread:
                signal.signal(signal.SIGINT, original_sigint)
                signal.signal(signal.SIGTERM, original_sigterm)

            self._publisher.close()
            self._source.close()
            self._running = False

            logger.info(f"Daemon stopped: {self._frame_count} frames published")

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False

    def _handle_signal(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, stopping...")
        self._running = False
