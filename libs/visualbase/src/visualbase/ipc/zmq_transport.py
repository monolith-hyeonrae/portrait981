"""ZeroMQ-based transport implementations.

Provides ZMQ implementations of the IPC interfaces for high-performance,
zero-copy frame sharing with dynamic connection support.

Video Transport (PUB/SUB pattern):
- ZMQVideoPublisher: Publishes video frames to subscribers
- ZMQVideoSubscriber: Subscribes to video frames from a publisher

Message Transport (PUB/SUB pattern):
- ZMQMessagePublisher: Publishes messages to subscribers
- ZMQMessageSubscriber: Subscribes to messages from a publisher

Example:
    >>> # Publisher side (Ingest)
    >>> pub = ZMQVideoPublisher("tcp://*:5555")
    >>> pub.open()
    >>> for frame in source:
    ...     pub.write(frame)
    >>> pub.close()

    >>> # Subscriber side (Extractor)
    >>> sub = ZMQVideoSubscriber("tcp://localhost:5555")
    >>> sub.open()
    >>> for frame in sub:
    ...     process(frame)
    >>> sub.close()

Requires: pyzmq (install with `uv sync --extra zmq`)
"""

import logging
import struct
import time
from typing import Optional, List

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    zmq = None

import numpy as np

from visualbase.ipc.interfaces import (
    VideoReader,
    VideoWriter,
    MessageReceiver,
    MessageSender,
)
from visualbase.core.frame import Frame

logger = logging.getLogger(__name__)

# Frame header format: frame_id (Q), t_src_ns (Q), width (I), height (I), channels (B)
FRAME_HEADER_FORMAT = "!QQIIB"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)


def _check_zmq():
    """Check if ZMQ is available."""
    if not HAS_ZMQ:
        raise ImportError(
            "pyzmq is required for ZMQ transport. "
            "Install with: uv sync --extra zmq"
        )


class ZMQVideoPublisher(VideoWriter):
    """ZMQ-based video frame publisher using PUB socket.

    Publishes video frames that can be received by multiple subscribers.
    Supports dynamic subscriber connection/disconnection.

    Args:
        address: ZMQ bind address (e.g., "tcp://*:5555").
        hwm: High water mark (max queued messages). Default: 2.
        topic: Topic prefix for filtering. Default: b"frame".

    Example:
        >>> pub = ZMQVideoPublisher("tcp://*:5555")
        >>> pub.open()
        >>> pub.write(frame)
        >>> pub.close()
    """

    def __init__(
        self,
        address: str,
        hwm: int = 2,
        topic: bytes = b"frame",
    ):
        _check_zmq()
        self._address = address
        self._hwm = hwm
        self._topic = topic
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._is_open = False

    def open(self) -> None:
        """Bind the publisher socket."""
        if self._is_open:
            return

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, self._hwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._address)
        self._is_open = True

        # Small delay to allow subscribers to connect
        time.sleep(0.1)
        logger.info(f"ZMQ video publisher bound to {self._address}")

    def write(self, frame: Frame) -> bool:
        """Publish a frame to all subscribers.

        Args:
            frame: Frame to publish.

        Returns:
            True if sent successfully.
        """
        if not self._is_open or self._socket is None:
            return False

        try:
            # Pack header
            header = struct.pack(
                FRAME_HEADER_FORMAT,
                frame.frame_id,
                frame.t_src_ns,
                frame.width,
                frame.height,
                frame.data.shape[2] if len(frame.data.shape) > 2 else 1,
            )

            # Send multipart: [topic, header, data]
            self._socket.send_multipart([
                self._topic,
                header,
                frame.data.tobytes(),
            ], flags=zmq.NOBLOCK)
            return True

        except zmq.Again:
            # No subscribers or buffer full
            return True  # Not an error, just no subscribers
        except zmq.ZMQError as e:
            logger.error(f"ZMQ send error: {e}")
            return False

    def close(self) -> None:
        """Close the publisher socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._is_open = False
        logger.debug("ZMQ video publisher closed")

    @property
    def is_open(self) -> bool:
        """Check if publisher is open."""
        return self._is_open


class ZMQVideoSubscriber(VideoReader):
    """ZMQ-based video frame subscriber using SUB socket.

    Subscribes to video frames from a publisher. Supports dynamic
    connection to running publishers.

    Args:
        address: ZMQ connect address (e.g., "tcp://localhost:5555").
        hwm: High water mark (max queued messages). Default: 2.
        topic: Topic prefix to subscribe to. Default: b"frame".
        timeout_ms: Receive timeout in milliseconds. Default: 1000.

    Example:
        >>> sub = ZMQVideoSubscriber("tcp://localhost:5555")
        >>> if sub.open():
        ...     for frame in sub:
        ...         process(frame)
        >>> sub.close()
    """

    def __init__(
        self,
        address: str,
        hwm: int = 2,
        topic: bytes = b"frame",
        timeout_ms: int = 1000,
    ):
        _check_zmq()
        self._address = address
        self._hwm = hwm
        self._topic = topic
        self._timeout_ms = timeout_ms
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._is_open = False

    def open(self, timeout_sec: Optional[float] = None) -> bool:
        """Connect to the publisher.

        Args:
            timeout_sec: Connection timeout (not used for ZMQ, connects async).

        Returns:
            True if socket created successfully.
        """
        if self._is_open:
            return True

        try:
            self._context = zmq.Context.instance()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.RCVHWM, self._hwm)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
            self._socket.connect(self._address)
            self._is_open = True
            logger.info(f"ZMQ video subscriber connected to {self._address}")
            return True

        except zmq.ZMQError as e:
            logger.error(f"ZMQ connect error: {e}")
            return False

    def read(self) -> Optional[Frame]:
        """Receive the next frame.

        Returns:
            Frame if received, None on timeout or error.
        """
        if not self._is_open or self._socket is None:
            return None

        try:
            # Receive multipart: [topic, header, data]
            parts = self._socket.recv_multipart()
            if len(parts) != 3:
                logger.warning(f"Invalid message parts: {len(parts)}")
                return None

            _, header, data = parts

            # Unpack header
            frame_id, t_src_ns, width, height, channels = struct.unpack(
                FRAME_HEADER_FORMAT, header
            )

            # Reconstruct frame data
            shape = (height, width, channels) if channels > 1 else (height, width)
            frame_data = np.frombuffer(data, dtype=np.uint8).reshape(shape)

            return Frame.from_array(
                data=frame_data.copy(),  # Copy to own the data
                frame_id=frame_id,
                t_src_ns=t_src_ns,
            )

        except zmq.Again:
            # Timeout, no message available
            return None
        except zmq.ZMQError as e:
            logger.error(f"ZMQ receive error: {e}")
            return None
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None

    def close(self) -> None:
        """Close the subscriber socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._is_open = False
        logger.debug("ZMQ video subscriber closed")

    @property
    def is_open(self) -> bool:
        """Check if subscriber is connected."""
        return self._is_open


class ZMQMessagePublisher(MessageSender):
    """ZMQ-based message publisher using PUB socket.

    Publishes messages that can be received by multiple subscribers.

    Args:
        address: ZMQ bind address (e.g., "tcp://*:5556").
        hwm: High water mark. Default: 100.
        topic: Topic prefix. Default: b"msg".
    """

    def __init__(
        self,
        address: str,
        hwm: int = 100,
        topic: bytes = b"msg",
    ):
        _check_zmq()
        self._address = address
        self._hwm = hwm
        self._topic = topic
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._is_connected = False

    def connect(self) -> bool:
        """Bind the publisher socket."""
        if self._is_connected:
            return True

        try:
            self._context = zmq.Context.instance()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.SNDHWM, self._hwm)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(self._address)
            self._is_connected = True
            time.sleep(0.1)  # Allow subscribers to connect
            logger.info(f"ZMQ message publisher bound to {self._address}")
            return True

        except zmq.ZMQError as e:
            logger.error(f"ZMQ bind error: {e}")
            return False

    def send(self, message: str) -> bool:
        """Publish a message.

        Args:
            message: Message string to publish.

        Returns:
            True if sent successfully.
        """
        if not self._is_connected or self._socket is None:
            return False

        try:
            self._socket.send_multipart([
                self._topic,
                message.encode("utf-8"),
            ], flags=zmq.NOBLOCK)
            return True

        except zmq.Again:
            return True  # No subscribers
        except zmq.ZMQError as e:
            logger.error(f"ZMQ send error: {e}")
            return False

    def disconnect(self) -> None:
        """Close the publisher socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if publisher is bound."""
        return self._is_connected


class ZMQMessageSubscriber(MessageReceiver):
    """ZMQ-based message subscriber using SUB socket.

    Subscribes to messages from a publisher.

    Args:
        address: ZMQ connect address (e.g., "tcp://localhost:5556").
        hwm: High water mark. Default: 100.
        topic: Topic prefix to subscribe to. Default: b"msg".
        timeout_ms: Receive timeout in milliseconds. Default: 100.
    """

    def __init__(
        self,
        address: str,
        hwm: int = 100,
        topic: bytes = b"msg",
        timeout_ms: int = 100,
    ):
        _check_zmq()
        self._address = address
        self._hwm = hwm
        self._topic = topic
        self._timeout_ms = timeout_ms
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._is_running = False

    def start(self) -> None:
        """Connect to the publisher."""
        if self._is_running:
            return

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, self._hwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(self._address)
        self._is_running = True
        logger.info(f"ZMQ message subscriber connected to {self._address}")

    def recv(self, timeout: Optional[float] = None) -> Optional[str]:
        """Receive a single message.

        Args:
            timeout: Timeout in seconds (uses socket timeout if None).

        Returns:
            Message string if received, None on timeout.
        """
        if not self._is_running or self._socket is None:
            return None

        # Temporarily adjust timeout if specified
        old_timeout = None
        if timeout is not None:
            old_timeout = self._socket.getsockopt(zmq.RCVTIMEO)
            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))

        try:
            parts = self._socket.recv_multipart()
            if len(parts) >= 2:
                return parts[1].decode("utf-8")
            return None

        except zmq.Again:
            return None
        except zmq.ZMQError as e:
            logger.error(f"ZMQ receive error: {e}")
            return None
        finally:
            if old_timeout is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, old_timeout)

    def recv_all(self, max_messages: int = 100) -> List[str]:
        """Receive all pending messages.

        Args:
            max_messages: Maximum messages to receive.

        Returns:
            List of message strings.
        """
        messages = []
        for _ in range(max_messages):
            msg = self.recv(timeout=0)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def stop(self) -> None:
        """Close the subscriber socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if subscriber is connected."""
        return self._is_running
