"""Tests for ZMQ transport implementations."""

import threading
import time

import pytest
import numpy as np

# Skip all tests if pyzmq is not installed
zmq = pytest.importorskip("zmq")

from visualbase.core.frame import Frame
from visualbase.ipc.zmq_transport import (
    ZMQVideoPublisher,
    ZMQVideoSubscriber,
    ZMQMessagePublisher,
    ZMQMessageSubscriber,
)
from visualbase.ipc.factory import TransportFactory


class TestZMQVideoTransport:
    """Tests for ZMQ video transport (PUB/SUB)."""

    def test_publisher_open_close(self):
        """Test publisher can open and close."""
        pub = ZMQVideoPublisher("tcp://127.0.0.1:15555")
        pub.open()
        assert pub.is_open
        pub.close()
        assert not pub.is_open

    def test_subscriber_open_close(self):
        """Test subscriber can open and close."""
        sub = ZMQVideoSubscriber("tcp://127.0.0.1:15556")
        assert sub.open()
        assert sub.is_open
        sub.close()
        assert not sub.is_open

    def test_pub_sub_single_frame(self):
        """Test publishing and receiving a single frame."""
        address = "tcp://127.0.0.1:15557"

        # Create publisher
        pub = ZMQVideoPublisher(address)
        pub.open()

        # Create subscriber
        sub = ZMQVideoSubscriber(address, timeout_ms=2000)
        sub.open()

        # Small delay for connection
        time.sleep(0.2)

        # Create and send frame
        data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data=data, frame_id=42, t_src_ns=1_000_000_000)

        assert pub.write(frame)

        # Receive frame
        received = sub.read()

        assert received is not None
        assert received.frame_id == 42
        assert received.t_src_ns == 1_000_000_000
        assert received.width == 640
        assert received.height == 480
        assert np.array_equal(received.data, data)

        # Cleanup
        sub.close()
        pub.close()

    def test_pub_sub_multiple_frames(self):
        """Test publishing and receiving multiple frames."""
        address = "tcp://127.0.0.1:15558"

        # Use higher HWM to avoid dropping frames in test
        pub = ZMQVideoPublisher(address, hwm=10)
        pub.open()

        sub = ZMQVideoSubscriber(address, hwm=10, timeout_ms=2000)
        sub.open()
        time.sleep(0.2)

        # Send multiple frames with small delay to ensure delivery
        sent_frames = []
        for i in range(5):
            data = np.full((100, 100, 3), i, dtype=np.uint8)
            frame = Frame.from_array(data=data, frame_id=i, t_src_ns=i * 100_000_000)
            sent_frames.append(frame)
            pub.write(frame)
            time.sleep(0.01)  # Small delay between sends

        # Small delay for all frames to arrive
        time.sleep(0.1)

        # Receive all frames
        received_frames = []
        for _ in range(5):
            frame = sub.read()
            if frame:
                received_frames.append(frame)

        assert len(received_frames) == 5
        for i, frame in enumerate(received_frames):
            assert frame.frame_id == i

        sub.close()
        pub.close()

    def test_subscriber_timeout(self):
        """Test subscriber returns None on timeout."""
        sub = ZMQVideoSubscriber("tcp://127.0.0.1:15559", timeout_ms=100)
        sub.open()

        # No publisher, should timeout
        frame = sub.read()
        assert frame is None

        sub.close()

    def test_grayscale_frame(self):
        """Test publishing/receiving grayscale frames."""
        address = "tcp://127.0.0.1:15560"

        pub = ZMQVideoPublisher(address)
        pub.open()

        sub = ZMQVideoSubscriber(address, timeout_ms=2000)
        sub.open()
        time.sleep(0.2)

        # Grayscale frame (2D array)
        data = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        frame = Frame.from_array(data=data, frame_id=1, t_src_ns=0)

        pub.write(frame)
        received = sub.read()

        assert received is not None
        assert received.data.shape == (480, 640)
        assert np.array_equal(received.data, data)

        sub.close()
        pub.close()


class TestZMQMessageTransport:
    """Tests for ZMQ message transport (PUB/SUB)."""

    def test_publisher_connect_disconnect(self):
        """Test message publisher can connect and disconnect."""
        pub = ZMQMessagePublisher("tcp://127.0.0.1:15561")
        assert pub.connect()
        assert pub.is_connected
        pub.disconnect()
        assert not pub.is_connected

    def test_subscriber_start_stop(self):
        """Test message subscriber can start and stop."""
        sub = ZMQMessageSubscriber("tcp://127.0.0.1:15562")
        sub.start()
        assert sub.is_running
        sub.stop()
        assert not sub.is_running

    def test_pub_sub_single_message(self):
        """Test publishing and receiving a single message."""
        address = "tcp://127.0.0.1:15563"

        pub = ZMQMessagePublisher(address)
        pub.connect()

        sub = ZMQMessageSubscriber(address, timeout_ms=2000)
        sub.start()
        time.sleep(0.2)

        # Send message
        assert pub.send("Hello, ZMQ!")

        # Receive message
        msg = sub.recv(timeout=2.0)
        assert msg == "Hello, ZMQ!"

        sub.stop()
        pub.disconnect()

    def test_pub_sub_multiple_messages(self):
        """Test publishing and receiving multiple messages."""
        address = "tcp://127.0.0.1:15564"

        pub = ZMQMessagePublisher(address)
        pub.connect()

        sub = ZMQMessageSubscriber(address, timeout_ms=2000)
        sub.start()
        time.sleep(0.2)

        # Send messages
        for i in range(5):
            pub.send(f"Message {i}")

        # Small delay for messages to arrive
        time.sleep(0.1)

        # Receive all
        messages = sub.recv_all(max_messages=10)
        assert len(messages) == 5
        assert messages[0] == "Message 0"
        assert messages[4] == "Message 4"

        sub.stop()
        pub.disconnect()

    def test_subscriber_timeout(self):
        """Test subscriber returns None on timeout."""
        sub = ZMQMessageSubscriber("tcp://127.0.0.1:15565", timeout_ms=100)
        sub.start()

        msg = sub.recv(timeout=0.1)
        assert msg is None

        sub.stop()

    def test_recv_all_empty(self):
        """Test recv_all returns empty list when no messages."""
        sub = ZMQMessageSubscriber("tcp://127.0.0.1:15566", timeout_ms=10)
        sub.start()

        messages = sub.recv_all()
        assert messages == []

        sub.stop()


class TestTransportFactoryZMQ:
    """Tests for TransportFactory with ZMQ transports."""

    def test_zmq_video_reader_creation(self):
        """Test creating ZMQ video reader via factory."""
        reader = TransportFactory.create_video_reader("zmq", "tcp://127.0.0.1:15567")
        assert isinstance(reader, ZMQVideoSubscriber)

    def test_zmq_video_writer_creation(self):
        """Test creating ZMQ video writer via factory."""
        writer = TransportFactory.create_video_writer("zmq", "tcp://127.0.0.1:15568")
        assert isinstance(writer, ZMQVideoPublisher)

    def test_zmq_message_receiver_creation(self):
        """Test creating ZMQ message receiver via factory."""
        receiver = TransportFactory.create_message_receiver("zmq", "tcp://127.0.0.1:15569")
        assert isinstance(receiver, ZMQMessageSubscriber)

    def test_zmq_message_sender_creation(self):
        """Test creating ZMQ message sender via factory."""
        sender = TransportFactory.create_message_sender("zmq", "tcp://127.0.0.1:15570")
        assert isinstance(sender, ZMQMessagePublisher)

    def test_zmq_listed_in_transports(self):
        """Test ZMQ is listed in available transports."""
        video_transports = TransportFactory.list_video_transports()
        assert "zmq" in video_transports["readers"]
        assert "zmq" in video_transports["writers"]

        message_transports = TransportFactory.list_message_transports()
        assert "zmq" in message_transports["receivers"]
        assert "zmq" in message_transports["senders"]


class TestZMQMultipleSubscribers:
    """Tests for ZMQ with multiple subscribers (1:N pattern)."""

    def test_one_publisher_multiple_subscribers(self):
        """Test one publisher can send to multiple subscribers."""
        address = "tcp://127.0.0.1:15571"

        pub = ZMQVideoPublisher(address)
        pub.open()

        # Create multiple subscribers
        subs = []
        for _ in range(3):
            sub = ZMQVideoSubscriber(address, timeout_ms=2000)
            sub.open()
            subs.append(sub)

        time.sleep(0.3)  # Allow all to connect

        # Send frame
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = Frame.from_array(data=data, frame_id=99, t_src_ns=0)
        pub.write(frame)

        # All subscribers should receive
        for sub in subs:
            received = sub.read()
            assert received is not None
            assert received.frame_id == 99

        # Cleanup
        for sub in subs:
            sub.close()
        pub.close()


class TestZMQContextManager:
    """Tests for context manager support."""

    def test_publisher_context_manager(self):
        """Test publisher as context manager."""
        with ZMQVideoPublisher("tcp://127.0.0.1:15572") as pub:
            assert pub.is_open
        assert not pub.is_open

    def test_subscriber_context_manager(self):
        """Test subscriber as context manager."""
        with ZMQVideoSubscriber("tcp://127.0.0.1:15573") as sub:
            assert sub.is_open
        assert not sub.is_open
