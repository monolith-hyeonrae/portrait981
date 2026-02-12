"""ZMQ REQ-REP based RPC transport.

Provides synchronous RPC communication using ZMQ REQ/REP sockets.
Each instance owns its own zmq.Context for process isolation safety.

Example:
    Server side (subprocess):
        >>> server = ZMQRPCServer()
        >>> server.bind("ipc:///tmp/worker.sock")
        >>> data = server.recv()
        >>> server.send(b'{"type": "pong"}')
        >>> server.close()

    Client side (main process):
        >>> client = ZMQRPCClient()
        >>> client.connect("ipc:///tmp/worker.sock")
        >>> client.send(b'{"type": "ping"}')
        >>> response = client.recv()
        >>> client.close()

Requires: pyzmq
"""

import logging
from typing import Optional

import zmq

from visualbase.ipc.interfaces import RPCServer, RPCClient

logger = logging.getLogger(__name__)


class ZMQRPCServer(RPCServer):
    """ZMQ REP socket RPC server.

    Binds to an address and processes recv/send cycles.
    Each instance creates its own zmq.Context.

    Args:
        linger_ms: Socket linger time on close (milliseconds).
    """

    def __init__(self, linger_ms: int = 0):
        self._linger_ms = linger_ms
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._is_bound = False

    def bind(self, address: str) -> None:
        """Bind the REP socket to an address."""
        if self._is_bound:
            return

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, self._linger_ms)
        self._socket.bind(address)
        self._is_bound = True
        logger.info(f"RPC server bound to {address}")

    def recv(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        """Receive a request from the client.

        Args:
            timeout_ms: Receive timeout. None for blocking wait.

        Returns:
            Request bytes, or None on timeout.
        """
        if not self._is_bound or self._socket is None:
            return None

        if timeout_ms is not None:
            old_timeout = self._socket.getsockopt(zmq.RCVTIMEO)
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            return self._socket.recv()
        except zmq.Again:
            return None
        finally:
            if timeout_ms is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, old_timeout)

    def send(self, data: bytes) -> None:
        """Send a response to the client."""
        if self._socket is None:
            raise RuntimeError("Server not bound")
        self._socket.send(data)

    def close(self) -> None:
        """Close the socket and terminate the context."""
        if self._socket is not None:
            try:
                self._socket.close(linger=self._linger_ms)
            except Exception:
                pass
            self._socket = None

        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None

        self._is_bound = False

    @property
    def is_bound(self) -> bool:
        return self._is_bound


class ZMQRPCClient(RPCClient):
    """ZMQ REQ socket RPC client.

    Connects to a server and performs send/recv cycles.
    Each instance creates its own zmq.Context.

    Args:
        send_timeout_ms: Default send timeout (milliseconds).
        recv_timeout_ms: Default receive timeout (milliseconds).
        linger_ms: Socket linger time on close (milliseconds).
    """

    def __init__(
        self,
        send_timeout_ms: int = 30000,
        recv_timeout_ms: int = 30000,
        linger_ms: int = 0,
    ):
        self._send_timeout_ms = send_timeout_ms
        self._recv_timeout_ms = recv_timeout_ms
        self._linger_ms = linger_ms
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._is_connected = False

    def connect(self, address: str) -> None:
        """Connect the REQ socket to a server address."""
        if self._is_connected:
            return

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, self._linger_ms)
        self._socket.connect(address)
        self._is_connected = True
        logger.debug(f"RPC client connected to {address}")

    def send(self, data: bytes) -> None:
        """Send a request to the server."""
        if self._socket is None:
            raise RuntimeError("Client not connected")
        self._socket.send(data)

    def recv(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        """Receive a response from the server.

        Args:
            timeout_ms: Override receive timeout. None uses default.

        Returns:
            Response bytes, or None on timeout.
        """
        if not self._is_connected or self._socket is None:
            return None

        if timeout_ms is not None:
            old_timeout = self._socket.getsockopt(zmq.RCVTIMEO)
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            return self._socket.recv()
        except zmq.Again:
            return None
        finally:
            if timeout_ms is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, old_timeout)

    def close(self) -> None:
        """Close the socket and terminate the context."""
        if self._socket is not None:
            try:
                self._socket.close(linger=self._linger_ms)
            except Exception:
                pass
            self._socket = None

        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None

        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected
