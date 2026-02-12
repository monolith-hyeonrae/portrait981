"""Tests for RPC transport (ABC + ZMQ implementation + Factory)."""

import json
import threading
import time

import pytest

zmq = pytest.importorskip("zmq")

from visualbase.ipc.interfaces import RPCServer, RPCClient
from visualbase.ipc.zmq_rpc import ZMQRPCServer, ZMQRPCClient
from visualbase.ipc.factory import TransportFactory
from visualbase.ipc._util import check_zmq_available, generate_ipc_address


# =============================================================================
# Utility tests
# =============================================================================


class TestUtilities:
    """Tests for IPC utility functions."""

    def test_check_zmq_available(self):
        """Test that check_zmq_available returns True when zmq is installed."""
        assert check_zmq_available() is True

    def test_generate_ipc_address_format(self):
        """Test that generate_ipc_address returns correct format."""
        address, filepath = generate_ipc_address()
        assert address.startswith("ipc://")
        assert filepath.endswith(".sock")
        assert "visualbase" in filepath

    def test_generate_ipc_address_custom_prefix(self):
        """Test generate_ipc_address with custom prefix."""
        address, filepath = generate_ipc_address(prefix="test-worker")
        assert "test-worker" in filepath

    def test_generate_ipc_address_unique(self):
        """Test that consecutive calls generate unique addresses."""
        addr1, _ = generate_ipc_address()
        addr2, _ = generate_ipc_address()
        assert addr1 != addr2


# =============================================================================
# ABC conformance tests
# =============================================================================


class TestRPCABCs:
    """Tests that ZMQ implementations conform to ABC interfaces."""

    def test_server_is_rpc_server(self):
        """Test ZMQRPCServer is an RPCServer."""
        server = ZMQRPCServer()
        assert isinstance(server, RPCServer)

    def test_client_is_rpc_client(self):
        """Test ZMQRPCClient is an RPCClient."""
        client = ZMQRPCClient()
        assert isinstance(client, RPCClient)


# =============================================================================
# ZMQ RPC communication tests
# =============================================================================


class TestZMQRPC:
    """Tests for ZMQ RPC communication."""

    def test_server_bind_close(self):
        """Test server can bind and close."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)
        assert server.is_bound
        server.close()
        assert not server.is_bound

    def test_client_connect_close(self):
        """Test client can connect and close."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        # Need a server for the client to connect to
        server = ZMQRPCServer()
        server.bind(address)

        client = ZMQRPCClient()
        client.connect(address)
        assert client.is_connected
        client.close()
        assert not client.is_connected

        server.close()

    def test_single_request_reply(self):
        """Test single request-reply cycle."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)

        client = ZMQRPCClient(recv_timeout_ms=5000)
        client.connect(address)
        time.sleep(0.05)

        # Run server in background thread
        received = {}

        def server_handler():
            data = server.recv(timeout_ms=5000)
            if data:
                received["request"] = json.loads(data)
                server.send(json.dumps({"type": "pong"}).encode())

        t = threading.Thread(target=server_handler)
        t.start()

        # Client sends and receives
        client.send(json.dumps({"type": "ping"}).encode())
        response = client.recv(timeout_ms=5000)

        t.join(timeout=5.0)

        assert response is not None
        assert json.loads(response) == {"type": "pong"}
        assert received["request"] == {"type": "ping"}

        client.close()
        server.close()

    def test_multiple_request_reply(self):
        """Test multiple request-reply cycles."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)

        client = ZMQRPCClient(recv_timeout_ms=5000)
        client.connect(address)
        time.sleep(0.05)

        def server_handler():
            for _ in range(3):
                data = server.recv(timeout_ms=5000)
                if data is None:
                    break
                msg = json.loads(data)
                server.send(json.dumps({"echo": msg["value"]}).encode())

        t = threading.Thread(target=server_handler)
        t.start()

        for i in range(3):
            client.send(json.dumps({"value": i}).encode())
            response = client.recv(timeout_ms=5000)
            assert response is not None
            assert json.loads(response) == {"echo": i}

        t.join(timeout=5.0)

        client.close()
        server.close()

    def test_client_recv_timeout(self):
        """Test client recv returns None on timeout."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)

        client = ZMQRPCClient(recv_timeout_ms=100)
        client.connect(address)
        time.sleep(0.05)

        # Send a request but server doesn't reply
        client.send(b'{"type": "ping"}')
        response = client.recv(timeout_ms=100)
        assert response is None

        # Drain the request on server side to clean up
        server.recv(timeout_ms=100)

        client.close()
        server.close()

    def test_server_recv_timeout(self):
        """Test server recv returns None on timeout."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)

        result = server.recv(timeout_ms=100)
        assert result is None

        server.close()

    def test_server_context_manager(self):
        """Test server as context manager."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        with ZMQRPCServer() as server:
            server.bind(address)
            assert server.is_bound
        assert not server.is_bound

    def test_client_context_manager(self):
        """Test client as context manager."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)

        with ZMQRPCClient() as client:
            client.connect(address)
            assert client.is_connected
        assert not client.is_connected

        server.close()

    def test_double_bind_is_noop(self):
        """Test that binding twice is safe."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)
        server.bind(address)  # Should be no-op
        assert server.is_bound
        server.close()

    def test_double_connect_is_noop(self):
        """Test that connecting twice is safe."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)

        client = ZMQRPCClient()
        client.connect(address)
        client.connect(address)  # Should be no-op
        assert client.is_connected

        client.close()
        server.close()

    def test_double_close_is_safe(self):
        """Test that closing twice is safe."""
        address, _ = generate_ipc_address(prefix="test-rpc")
        server = ZMQRPCServer()
        server.bind(address)
        server.close()
        server.close()  # Should be safe

        client = ZMQRPCClient()
        client.connect(address)
        client.close()
        client.close()  # Should be safe


# =============================================================================
# Factory tests
# =============================================================================


class TestTransportFactoryRPC:
    """Tests for TransportFactory RPC methods."""

    def test_create_rpc_server(self):
        """Test creating RPC server via factory."""
        server = TransportFactory.create_rpc_server("zmq")
        assert isinstance(server, ZMQRPCServer)

    def test_create_rpc_client(self):
        """Test creating RPC client via factory."""
        client = TransportFactory.create_rpc_client("zmq")
        assert isinstance(client, ZMQRPCClient)

    def test_create_rpc_server_with_kwargs(self):
        """Test creating RPC server with kwargs."""
        server = TransportFactory.create_rpc_server("zmq", linger_ms=100)
        assert isinstance(server, ZMQRPCServer)
        assert server._linger_ms == 100

    def test_create_rpc_client_with_kwargs(self):
        """Test creating RPC client with kwargs."""
        client = TransportFactory.create_rpc_client(
            "zmq",
            send_timeout_ms=5000,
            recv_timeout_ms=10000,
        )
        assert isinstance(client, ZMQRPCClient)
        assert client._send_timeout_ms == 5000
        assert client._recv_timeout_ms == 10000

    def test_unknown_rpc_server_raises(self):
        """Test that unknown server type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown RPC server"):
            TransportFactory.create_rpc_server("unknown")

    def test_unknown_rpc_client_raises(self):
        """Test that unknown client type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown RPC client"):
            TransportFactory.create_rpc_client("unknown")

    def test_rpc_listed_in_transports(self):
        """Test RPC is listed in available transports."""
        rpc_transports = TransportFactory.list_rpc_transports()
        assert "zmq" in rpc_transports["servers"]
        assert "zmq" in rpc_transports["clients"]
