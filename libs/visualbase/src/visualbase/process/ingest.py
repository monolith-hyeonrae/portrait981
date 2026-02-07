"""Ingest process (A module) for video capture and distribution.

The Ingest process is the "Time Authority" in the A-B*-C architecture:
- Captures video from camera/RTSP source
- Maintains a RingBuffer for clip extraction
- Distributes proxy streams to extractors via FIFOs
- Receives TRIG messages and extracts clips

Supports interface-based dependency injection for swappable transports.

Example (legacy path-based):
    >>> from visualbase.sources.camera import CameraSource
    >>> from visualbase.streaming.fanout import ProxyConfig
    >>>
    >>> process = IngestProcess(
    ...     source=CameraSource(0),
    ...     proxy_configs=[
    ...         ProxyConfig("face", "/tmp/vid_face.mjpg", 640, 480, 10),
    ...         ProxyConfig("pose", "/tmp/vid_pose.mjpg", 640, 480, 10),
    ...     ],
    ...     trig_socket="/tmp/trig.sock",
    ...     clip_output_dir=Path("./clips"),
    ... )
    >>> process.run()

Example (interface-based):
    >>> from visualbase.ipc.factory import TransportFactory
    >>> trig_receiver = TransportFactory.create_message_receiver("uds", "/tmp/trig.sock")
    >>> process = IngestProcess(
    ...     source=CameraSource(0),
    ...     proxy_configs=[...],
    ...     trig_receiver=trig_receiver,
    ...     clip_output_dir=Path("./clips"),
    ... )
    >>> process.run()
"""

import signal
import time
import logging
from pathlib import Path
from typing import Optional, List, Callable
import threading

from visualbase.sources.base import BaseSource
from visualbase.core.ring_buffer import RingBuffer
from visualbase.streaming.fanout import ProxyFanout, ProxyConfig
from visualbase.ipc.interfaces import MessageReceiver
from visualbase.ipc.factory import TransportFactory
from visualbase.ipc.messages import parse_trig_message, TRIGMessage
from visualbase.packaging.trigger import Trigger, TriggerType
from visualbase.packaging.clipper import ClipResult
from visualbase.core.frame import Frame

logger = logging.getLogger(__name__)


class IngestProcess:
    """Video ingest process for A-B*-C architecture.

    Captures video from a source, maintains a ring buffer for clip
    extraction, distributes proxy streams to extractors, and handles
    trigger messages for clip extraction.

    Supports two initialization modes:
    1. Interface-based: Pass MessageReceiver instance directly
    2. Legacy path-based: Pass trig_socket path (auto-creates UDS)

    Args:
        source: Video source (CameraSource, RTSPSource, etc.).
        proxy_configs: List of proxy output configurations.
        trig_receiver: MessageReceiver instance for receiving TRIG messages.
        trig_socket: (Legacy) Path to the UDS socket for receiving TRIG messages.
        message_transport: Transport type for messages ("uds", "zmq"). Default: "uds".
        clip_output_dir: Directory to save extracted clips.
        ring_buffer_retention_sec: Ring buffer retention time in seconds.
        on_clip: Optional callback when a clip is extracted.
        on_frame: Optional callback for each captured frame.
    """

    def __init__(
        self,
        source: BaseSource,
        proxy_configs: List[ProxyConfig],
        clip_output_dir: Path,
        trig_receiver: Optional[MessageReceiver] = None,
        trig_socket: Optional[str] = None,
        message_transport: str = "uds",
        ring_buffer_retention_sec: float = 120.0,
        on_clip: Optional[Callable[[ClipResult], None]] = None,
        on_frame: Optional[Callable[[Frame], None]] = None,
    ):
        self._source = source
        self._proxy_configs = proxy_configs
        self._clip_output_dir = Path(clip_output_dir)
        self._ring_buffer_retention_sec = ring_buffer_retention_sec
        self._on_clip = on_clip
        self._on_frame = on_frame

        # Store transport config
        self._message_transport = message_transport
        self._trig_path = trig_socket

        # Interface-based or legacy path-based initialization
        if trig_receiver is not None:
            self._trig_server: Optional[MessageReceiver] = trig_receiver
            self._trig_server_provided = True
        elif trig_socket is not None:
            self._trig_server = None  # Created in run()
            self._trig_server_provided = False
        else:
            raise ValueError("Either trig_receiver or trig_socket must be provided")

        self._ring_buffer: Optional[RingBuffer] = None
        self._fanout: Optional[ProxyFanout] = None
        self._trig_thread: Optional[threading.Thread] = None

        self._running = False
        self._shutdown = threading.Event()

        # Stats
        self._frames_captured = 0
        self._clips_extracted = 0
        self._triggers_received = 0
        self._errors = 0
        self._start_time: Optional[float] = None

        # Clip output directory
        self._clip_output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the ingest process main loop.

        This method blocks until stop() is called or the process is
        interrupted.
        """
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._start_time = time.monotonic()

        # Initialize components
        self._ring_buffer = RingBuffer(
            retention_sec=self._ring_buffer_retention_sec,
        )

        self._fanout = ProxyFanout(self._proxy_configs)
        self._fanout.set_frame_callback(self._ring_buffer.add_frame)

        # Create TRIG receiver if not provided
        if self._trig_server is None and self._trig_path is not None:
            self._trig_server = TransportFactory.create_message_receiver(
                self._message_transport, self._trig_path
            )

        # Start TRIG server
        if self._trig_server is None:
            logger.error("No TRIG receiver available")
            return
        self._trig_server.start()

        # Start TRIG handler thread
        self._trig_thread = threading.Thread(target=self._trig_handler_loop, daemon=True)
        self._trig_thread.start()

        # Start fanout (waits for readers to connect)
        logger.info("Waiting for extractor connections...")
        self._fanout.start(wait_for_readers=True, timeout=30.0)

        logger.info(f"Ingest process started")
        logger.info(f"  Source: {self._source}")
        logger.info(f"  TRIG receiver: {self._trig_path or 'provided'}")
        logger.info(f"  Clip output: {self._clip_output_dir}")
        logger.info(f"  Proxy outputs: {len(self._proxy_configs)}")

        try:
            # Main capture loop
            self._capture_loop()

        finally:
            self._cleanup()

    def _capture_loop(self) -> None:
        """Main video capture loop."""
        while self._running and not self._shutdown.is_set():
            frame = self._source.read()
            if frame is None:
                if self._source.is_seekable:
                    # End of file
                    logger.info("End of source file")
                    break
                else:
                    # Stream error - wait and retry
                    time.sleep(0.1)
                    continue

            self._frames_captured += 1

            # Distribute to proxy outputs (this also adds to ring buffer)
            self._fanout.write(frame)

            # Call frame callback if set
            if self._on_frame:
                try:
                    self._on_frame(frame)
                except Exception as e:
                    logger.warning(f"Frame callback error: {e}")

    def _trig_handler_loop(self) -> None:
        """Background thread for handling TRIG messages."""
        while self._running and not self._shutdown.is_set():
            try:
                messages = self._trig_server.recv_all(max_messages=10)
                for msg in messages:
                    self._handle_trig_message(msg)

                if not messages:
                    time.sleep(0.01)  # 10ms

            except Exception as e:
                logger.error(f"TRIG handler error: {e}")
                self._errors += 1

    def _handle_trig_message(self, msg: str) -> None:
        """Handle a received TRIG message."""
        trig_msg = parse_trig_message(msg)
        if trig_msg is None:
            logger.warning(f"Failed to parse TRIG message: {msg[:100]}")
            return

        self._triggers_received += 1
        logger.info(
            f"Received TRIG: {trig_msg.label} score={trig_msg.score:.2f} "
            f"reason={trig_msg.reason}"
        )

        # Convert to Trigger and extract clip
        trigger = Trigger.range(
            start_time_ns=trig_msg.t_start_ns,
            end_time_ns=trig_msg.t_end_ns,
            pre_sec=0,  # Already included in t_start_ns
            post_sec=0,  # Already included in t_end_ns
            label=trig_msg.label,
            score=trig_msg.score,
            metadata={
                "reason": trig_msg.reason,
                "faces": trig_msg.faces,
            },
        )

        # Extract clip
        try:
            result = self._ring_buffer.extract_clip(trigger, self._clip_output_dir)
            if result.success:
                self._clips_extracted += 1
                logger.info(f"Clip extracted: {result.output_path}")

                # Call clip callback if set
                if self._on_clip:
                    self._on_clip(result)
            else:
                logger.warning(f"Clip extraction failed: {result.error}")
                self._errors += 1

        except Exception as e:
            logger.error(f"Clip extraction error: {e}")
            self._errors += 1

    def stop(self) -> None:
        """Stop the ingest process."""
        logger.info("Stopping ingest process")
        self._running = False
        self._shutdown.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._fanout:
            self._fanout.stop()
            self._fanout = None

        if self._trig_server:
            self._trig_server.stop()
            if not self._trig_server_provided:
                self._trig_server = None

        if self._ring_buffer:
            self._ring_buffer.close()
            self._ring_buffer = None

        # Log stats
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        fps = self._frames_captured / elapsed if elapsed > 0 else 0
        logger.info(
            f"Ingest process stopped: "
            f"{self._frames_captured} frames ({fps:.1f} fps), "
            f"{self._triggers_received} triggers, "
            f"{self._clips_extracted} clips, "
            f"{self._errors} errors in {elapsed:.1f}s"
        )

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if the process is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get processing statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "frames_captured": self._frames_captured,
            "triggers_received": self._triggers_received,
            "clips_extracted": self._clips_extracted,
            "errors": self._errors,
            "elapsed_sec": elapsed,
            "fps": self._frames_captured / elapsed if elapsed > 0 else 0,
            "buffer_info": self._ring_buffer.info if self._ring_buffer else None,
            "fanout_stats": self._fanout.get_stats() if self._fanout else None,
        }

    def get_buffer_info(self) -> Optional[dict]:
        """Get ring buffer information."""
        if self._ring_buffer:
            info = self._ring_buffer.info
            return {
                "start_ns": info.start_ns,
                "end_ns": info.end_ns,
                "duration_sec": info.duration_sec,
                "segment_count": self._ring_buffer.segment_count,
            }
        return None

    def trigger_manual_clip(
        self,
        pre_sec: float = 5.0,
        post_sec: float = 1.0,
        label: str = "manual",
    ) -> Optional[ClipResult]:
        """Manually trigger a clip extraction.

        Extracts a clip from the ring buffer ending at the current time.

        Args:
            pre_sec: Seconds before current time.
            post_sec: Seconds after current time.
            label: Label for the clip.

        Returns:
            ClipResult if extraction succeeded.
        """
        if not self._ring_buffer:
            return None

        # Get current time from buffer
        info = self._ring_buffer.info
        if info.end_ns == 0:
            return None

        trigger = Trigger.point(
            event_time_ns=info.end_ns,
            pre_sec=pre_sec,
            post_sec=post_sec,
            label=label,
        )

        try:
            result = self._ring_buffer.extract_clip(trigger, self._clip_output_dir)
            if result.success:
                self._clips_extracted += 1
                logger.info(f"Manual clip extracted: {result.output_path}")
            return result
        except Exception as e:
            logger.error(f"Manual clip extraction error: {e}")
            return None
