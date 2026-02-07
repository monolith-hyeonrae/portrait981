"""Unix Domain Socket (UDS) datagram messaging.

Provides datagram-based messaging for:
- OBS (Observation) messages: B* → C
- TRIG (Trigger) messages: C → A

Uses SOCK_DGRAM for fire-and-forget semantics with message boundaries.
"""

import os
import socket
import threading
import select
from pathlib import Path
from typing import Optional, List
import logging

from visualbase.ipc.interfaces import MessageReceiver, MessageSender

logger = logging.getLogger(__name__)

# Maximum message size (64KB should be enough for any OBS/TRIG message)
MAX_MESSAGE_SIZE = 65536

# Default socket buffer size
SOCKET_BUFFER_SIZE = 1024 * 1024  # 1MB


class UDSServer(MessageReceiver):
    """Unix Domain Socket datagram server for receiving messages.

    Creates a UDS socket at the specified path and receives datagrams.
    Used by C (Fusion) to receive OBS messages and A (Ingest) to receive TRIG.

    Implements the MessageReceiver interface for swappable transport.

    Args:
        path: Path to create the socket.
        buffer_size: Socket receive buffer size.

    Example:
        >>> server = UDSServer("/tmp/obs.sock")
        >>> server.start()
        >>> while True:
        ...     msg = server.recv(timeout=1.0)
        ...     if msg:
        ...         process(msg)
        >>> server.stop()
    """

    def __init__(self, path: str, buffer_size: int = SOCKET_BUFFER_SIZE):
        self._path = Path(path)
        self._buffer_size = buffer_size
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Create and bind the UDS socket."""
        # Remove existing socket file
        if self._path.exists():
            self._path.unlink()

        # Create datagram socket
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, self._buffer_size
        )
        self._socket.bind(str(self._path))
        self._socket.setblocking(False)

        logger.info(f"UDS server started: {self._path}")

    def recv(self, timeout: Optional[float] = None) -> Optional[str]:
        """Receive a message.

        Args:
            timeout: Timeout in seconds. None for non-blocking.

        Returns:
            Message string if received, None on timeout or error.
        """
        if self._socket is None:
            return None

        try:
            with self._lock:
                # Use select for timeout
                if timeout is not None:
                    readable, _, _ = select.select(
                        [self._socket], [], [], timeout
                    )
                    if not readable:
                        return None

                data, _ = self._socket.recvfrom(MAX_MESSAGE_SIZE)
                return data.decode("utf-8")

        except BlockingIOError:
            return None
        except Exception as e:
            logger.error(f"UDS recv error: {e}")
            return None

    def recv_all(self, max_messages: int = 100) -> List[str]:
        """Receive all pending messages.

        Args:
            max_messages: Maximum messages to receive in one call.

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
        """Close the socket and clean up."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        # Remove socket file
        if self._path.exists():
            try:
                self._path.unlink()
            except Exception:
                pass

        logger.info(f"UDS server stopped: {self._path}")

    def __enter__(self) -> "UDSServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._socket is not None

    @property
    def path(self) -> str:
        """Get the socket path."""
        return str(self._path)


class UDSClient(MessageSender):
    """Unix Domain Socket datagram client for sending messages.

    Connects to a UDS socket and sends datagrams.
    Used by B* to send OBS messages and C to send TRIG messages.

    Implements the MessageSender interface for swappable transport.

    Args:
        path: Path to the target socket.

    Example:
        >>> client = UDSClient("/tmp/obs.sock")
        >>> client.connect()
        >>> client.send("OBS src=face frame=123 ...")
        >>> client.disconnect()
    """

    def __init__(self, path: str):
        self._path = Path(path)
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to the UDS socket.

        Returns:
            True if connected successfully.
        """
        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._socket.connect(str(self._path))
            logger.info(f"UDS client connected: {self._path}")
            return True
        except Exception as e:
            logger.error(f"UDS connect error: {e}")
            self._socket = None
            return False

    def send(self, message: str) -> bool:
        """Send a message.

        Args:
            message: Message string to send.

        Returns:
            True if sent successfully.
        """
        if self._socket is None:
            return False

        try:
            with self._lock:
                data = message.encode("utf-8")
                if len(data) > MAX_MESSAGE_SIZE:
                    logger.warning(
                        f"Message too large: {len(data)} > {MAX_MESSAGE_SIZE}"
                    )
                    return False
                self._socket.send(data)
            return True
        except Exception as e:
            logger.error(f"UDS send error: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the socket."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            logger.info(f"UDS client disconnected: {self._path}")

    def __enter__(self) -> "UDSClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._socket is not None

    @property
    def path(self) -> str:
        """Get the target socket path."""
        return str(self._path)
