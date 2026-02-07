"""Tests for IPC modules (FIFO, UDS, messages, interfaces, factory)."""

import os
import tempfile
import threading
import time
from abc import ABC
from pathlib import Path

import pytest
import numpy as np

from visualbase.core.frame import Frame
from visualbase.ipc.fifo import FIFOVideoWriter, FIFOVideoReader, HEADER_SIZE
from visualbase.ipc.uds import UDSServer, UDSClient
from visualbase.ipc.interfaces import (
    VideoReader,
    VideoWriter,
    MessageReceiver,
    MessageSender,
)
from visualbase.ipc.factory import TransportFactory
from visualbase.ipc.messages import (
    FaceData,
    PoseData,
    QualityData,
    FaceOBS,
    PoseOBS,
    QualityOBS,
    OBSMessage,
    TRIGMessage,
    parse_obs_message,
    parse_trig_message,
)


class TestFaceData:
    """Tests for FaceData serialization."""

    def test_to_string(self):
        face = FaceData(
            id=0, conf=0.95, x=0.1, y=0.2, w=0.3, h=0.4, expr=0.8
        )
        s = face.to_string()
        assert "id:0" in s
        assert "conf:0.950" in s
        assert "expr:0.800" in s

    def test_from_string(self):
        s = "id:0,conf:0.95,x:0.1,y:0.2,w:0.3,h:0.4,expr:0.8"
        face = FaceData.from_string(s)
        assert face.id == 0
        assert face.conf == 0.95
        assert face.x == 0.1
        assert face.expr == 0.8

    def test_roundtrip(self):
        original = FaceData(
            id=1, conf=0.92, x=0.15, y=0.25, w=0.35, h=0.45,
            expr=0.7, yaw=5.0, pitch=2.0
        )
        s = original.to_string()
        parsed = FaceData.from_string(s)
        assert parsed.id == original.id
        assert parsed.conf == pytest.approx(original.conf, rel=1e-2)
        assert parsed.expr == pytest.approx(original.expr, rel=1e-2)


class TestPoseData:
    """Tests for PoseData serialization."""

    def test_to_string(self):
        pose = PoseData(id=0, conf=0.9, hand_raised=True, hand_wave=False)
        s = pose.to_string()
        assert "id:0" in s
        assert "hr:1" in s
        assert "hw:0" in s

    def test_from_string(self):
        s = "id:0,conf:0.9,hr:1,hw:0"
        pose = PoseData.from_string(s)
        assert pose.id == 0
        assert pose.hand_raised is True
        assert pose.hand_wave is False


class TestQualityData:
    """Tests for QualityData serialization."""

    def test_to_string(self):
        quality = QualityData(blur=100.0, brightness=128.0, contrast=0.5, gate_open=True)
        s = quality.to_string()
        assert "blur:100.0" in s
        assert "bright:128.0" in s
        assert "gate:1" in s

    def test_from_string(self):
        s = "blur:100.0,bright:128.0,contrast:0.500,gate:1"
        quality = QualityData.from_string(s)
        assert quality.blur == 100.0
        assert quality.brightness == 128.0
        assert quality.gate_open is True


class TestFaceOBS:
    """Tests for FaceOBS message."""

    def test_to_message(self):
        obs = FaceOBS(
            frame_id=1234,
            t_ns=1234567890,
            faces=[
                FaceData(id=0, conf=0.95, x=0.1, y=0.2, w=0.3, h=0.4, expr=0.8),
            ],
        )
        msg = obs.to_message()
        assert msg.startswith("OBS src=face")
        assert "frame=1234" in msg
        assert "faces=1" in msg
        assert "f0=" in msg


class TestPoseOBS:
    """Tests for PoseOBS message."""

    def test_to_message(self):
        obs = PoseOBS(
            frame_id=1234,
            t_ns=1234567890,
            poses=[
                PoseData(id=0, conf=0.9, hand_raised=True, hand_wave=False),
            ],
        )
        msg = obs.to_message()
        assert msg.startswith("OBS src=pose")
        assert "frame=1234" in msg
        assert "poses=1" in msg
        assert "p0=" in msg


class TestQualityOBS:
    """Tests for QualityOBS message."""

    def test_to_message(self):
        obs = QualityOBS(
            frame_id=1234,
            t_ns=1234567890,
            quality=QualityData(blur=100.0, brightness=128.0, contrast=0.5),
        )
        msg = obs.to_message()
        assert msg.startswith("OBS src=quality")
        assert "frame=1234" in msg
        assert "blur:100.0" in msg


class TestParseOBSMessage:
    """Tests for parse_obs_message."""

    def test_parse_face_obs(self):
        msg = "OBS src=face frame=1234 t_ns=1234567890 faces=1 f0=id:0,conf:0.95,x:0.1,y:0.2,w:0.3,h:0.4,expr:0.8"
        obs = parse_obs_message(msg)
        assert obs is not None
        assert obs.src == "face"
        assert obs.frame_id == 1234
        assert len(obs.faces) == 1
        assert obs.faces[0].id == 0
        assert obs.faces[0].expr == 0.8

    def test_parse_pose_obs(self):
        msg = "OBS src=pose frame=1234 t_ns=1234567890 poses=1 p0=id:0,conf:0.9,hr:1,hw:0"
        obs = parse_obs_message(msg)
        assert obs is not None
        assert obs.src == "pose"
        assert len(obs.poses) == 1
        assert obs.poses[0].hand_raised is True

    def test_parse_invalid(self):
        assert parse_obs_message("INVALID") is None
        assert parse_obs_message("") is None


class TestTRIGMessage:
    """Tests for TRIGMessage."""

    def test_to_message(self):
        trig = TRIGMessage(
            label="PORTRAIT_HIGHLIGHT",
            t_start_ns=1234567890,
            t_end_ns=1239567890,
            faces=2,
            score=0.85,
            reason="expr_spike",
        )
        msg = trig.to_message()
        assert msg.startswith("TRIG")
        assert "label=PORTRAIT_HIGHLIGHT" in msg
        assert "score=0.850" in msg
        assert "reason=expr_spike" in msg


class TestParseTRIGMessage:
    """Tests for parse_trig_message."""

    def test_parse_trig(self):
        msg = "TRIG label=PORTRAIT_HIGHLIGHT t_start_ns=1234567890 t_end_ns=1239567890 faces=2 score=0.85 reason=expr_spike"
        trig = parse_trig_message(msg)
        assert trig is not None
        assert trig.label == "PORTRAIT_HIGHLIGHT"
        assert trig.t_start_ns == 1234567890
        assert trig.t_end_ns == 1239567890
        assert trig.faces == 2
        assert trig.score == 0.85
        assert trig.reason == "expr_spike"

    def test_parse_invalid(self):
        assert parse_trig_message("INVALID") is None


class TestUDS:
    """Tests for UDS server and client."""

    def test_server_start_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")
            server = UDSServer(sock_path)
            server.start()
            assert server.is_running
            assert Path(sock_path).exists()
            server.stop()
            assert not server.is_running

    def test_client_connect_disconnect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")
            server = UDSServer(sock_path)
            server.start()

            client = UDSClient(sock_path)
            assert client.connect()
            assert client.is_connected
            client.disconnect()
            assert not client.is_connected

            server.stop()

    def test_send_receive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")

            with UDSServer(sock_path) as server:
                with UDSClient(sock_path) as client:
                    # Send message
                    assert client.send("Hello, World!")

                    # Receive message
                    msg = server.recv(timeout=1.0)
                    assert msg == "Hello, World!"

    def test_recv_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")

            with UDSServer(sock_path) as server:
                with UDSClient(sock_path) as client:
                    # Send multiple messages
                    client.send("msg1")
                    client.send("msg2")
                    client.send("msg3")

                    # Give messages time to arrive
                    time.sleep(0.1)

                    # Receive all
                    messages = server.recv_all()
                    assert len(messages) == 3
                    assert "msg1" in messages
                    assert "msg2" in messages
                    assert "msg3" in messages


class TestFIFO:
    """Tests for FIFO video streaming."""

    def _create_test_frame(self, frame_id: int = 1) -> Frame:
        """Create a test frame."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        data[100:200, 100:200] = (255, 0, 0)  # Red square
        return Frame.from_array(data, frame_id=frame_id, t_src_ns=frame_id * 1_000_000_000)

    def test_writer_reader_roundtrip(self):
        """Test writing and reading frames through FIFO."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo_path = os.path.join(tmpdir, "test.mjpg")

            # Start writer in background (blocks until reader connects)
            writer = FIFOVideoWriter(fifo_path)
            writer_thread = threading.Thread(target=writer.open)
            writer_thread.start()

            # Give writer time to create FIFO
            time.sleep(0.1)

            # Open reader (unblocks writer)
            reader = FIFOVideoReader(fifo_path)
            assert reader.open()

            # Wait for writer to finish connecting
            writer_thread.join(timeout=1.0)

            # Write a frame
            original = self._create_test_frame(1)
            assert writer.write(original)

            # Read the frame
            received = reader.read()
            assert received is not None
            assert received.frame_id == 1
            assert received.t_src_ns == 1_000_000_000
            assert received.width == 640
            assert received.height == 480

            # Clean up
            writer.close()
            reader.close()

    def test_multiple_frames(self):
        """Test writing and reading multiple frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo_path = os.path.join(tmpdir, "test.mjpg")

            writer = FIFOVideoWriter(fifo_path)
            writer_thread = threading.Thread(target=writer.open)
            writer_thread.start()

            time.sleep(0.1)

            reader = FIFOVideoReader(fifo_path)
            reader.open()
            writer_thread.join(timeout=1.0)

            # Write multiple frames
            for i in range(5):
                frame = self._create_test_frame(i)
                writer.write(frame)

            # Read them back
            for i in range(5):
                received = reader.read()
                assert received is not None
                assert received.frame_id == i

            writer.close()
            reader.close()


class TestInterfaces:
    """Tests for IPC interfaces (ABCs)."""

    def test_video_reader_is_abc(self):
        """VideoReader is an abstract base class."""
        assert issubclass(VideoReader, ABC)
        # Cannot instantiate ABC
        with pytest.raises(TypeError):
            VideoReader()

    def test_video_writer_is_abc(self):
        """VideoWriter is an abstract base class."""
        assert issubclass(VideoWriter, ABC)
        with pytest.raises(TypeError):
            VideoWriter()

    def test_message_receiver_is_abc(self):
        """MessageReceiver is an abstract base class."""
        assert issubclass(MessageReceiver, ABC)
        with pytest.raises(TypeError):
            MessageReceiver()

    def test_message_sender_is_abc(self):
        """MessageSender is an abstract base class."""
        assert issubclass(MessageSender, ABC)
        with pytest.raises(TypeError):
            MessageSender()

    def test_fifo_reader_implements_video_reader(self):
        """FIFOVideoReader implements VideoReader interface."""
        assert issubclass(FIFOVideoReader, VideoReader)
        # Check required methods exist
        reader = FIFOVideoReader("/tmp/test.mjpg")
        assert hasattr(reader, "open")
        assert hasattr(reader, "read")
        assert hasattr(reader, "close")
        assert hasattr(reader, "is_open")

    def test_fifo_writer_implements_video_writer(self):
        """FIFOVideoWriter implements VideoWriter interface."""
        assert issubclass(FIFOVideoWriter, VideoWriter)
        writer = FIFOVideoWriter("/tmp/test.mjpg")
        assert hasattr(writer, "open")
        assert hasattr(writer, "write")
        assert hasattr(writer, "close")
        assert hasattr(writer, "is_open")

    def test_uds_server_implements_message_receiver(self):
        """UDSServer implements MessageReceiver interface."""
        assert issubclass(UDSServer, MessageReceiver)
        server = UDSServer("/tmp/test.sock")
        assert hasattr(server, "start")
        assert hasattr(server, "recv")
        assert hasattr(server, "recv_all")
        assert hasattr(server, "stop")
        assert hasattr(server, "is_running")

    def test_uds_client_implements_message_sender(self):
        """UDSClient implements MessageSender interface."""
        assert issubclass(UDSClient, MessageSender)
        client = UDSClient("/tmp/test.sock")
        assert hasattr(client, "connect")
        assert hasattr(client, "send")
        assert hasattr(client, "disconnect")
        assert hasattr(client, "is_connected")


class TestTransportFactory:
    """Tests for TransportFactory."""

    def test_create_video_reader_fifo(self):
        """Create FIFO video reader via factory."""
        reader = TransportFactory.create_video_reader("fifo", "/tmp/test.mjpg")
        assert isinstance(reader, FIFOVideoReader)
        assert isinstance(reader, VideoReader)

    def test_create_video_writer_fifo(self):
        """Create FIFO video writer via factory."""
        writer = TransportFactory.create_video_writer("fifo", "/tmp/test.mjpg")
        assert isinstance(writer, FIFOVideoWriter)
        assert isinstance(writer, VideoWriter)

    def test_create_video_writer_with_kwargs(self):
        """Create video writer with custom kwargs."""
        writer = TransportFactory.create_video_writer(
            "fifo", "/tmp/test.mjpg", jpeg_quality=95
        )
        assert isinstance(writer, FIFOVideoWriter)
        assert writer._jpeg_quality == 95

    def test_create_message_receiver_uds(self):
        """Create UDS message receiver via factory."""
        receiver = TransportFactory.create_message_receiver("uds", "/tmp/test.sock")
        assert isinstance(receiver, UDSServer)
        assert isinstance(receiver, MessageReceiver)

    def test_create_message_sender_uds(self):
        """Create UDS message sender via factory."""
        sender = TransportFactory.create_message_sender("uds", "/tmp/test.sock")
        assert isinstance(sender, UDSClient)
        assert isinstance(sender, MessageSender)

    def test_invalid_video_reader_type(self):
        """Raise error for invalid video reader type."""
        with pytest.raises(ValueError) as exc_info:
            TransportFactory.create_video_reader("invalid", "/tmp/test")
        assert "Unknown video reader transport" in str(exc_info.value)

    def test_invalid_video_writer_type(self):
        """Raise error for invalid video writer type."""
        with pytest.raises(ValueError) as exc_info:
            TransportFactory.create_video_writer("invalid", "/tmp/test")
        assert "Unknown video writer transport" in str(exc_info.value)

    def test_invalid_message_receiver_type(self):
        """Raise error for invalid message receiver type."""
        with pytest.raises(ValueError) as exc_info:
            TransportFactory.create_message_receiver("invalid", "/tmp/test")
        assert "Unknown message receiver transport" in str(exc_info.value)

    def test_invalid_message_sender_type(self):
        """Raise error for invalid message sender type."""
        with pytest.raises(ValueError) as exc_info:
            TransportFactory.create_message_sender("invalid", "/tmp/test")
        assert "Unknown message sender transport" in str(exc_info.value)

    def test_list_video_transports(self):
        """List available video transports."""
        transports = TransportFactory.list_video_transports()
        assert "readers" in transports
        assert "writers" in transports
        assert "fifo" in transports["readers"]
        assert "fifo" in transports["writers"]

    def test_list_message_transports(self):
        """List available message transports."""
        transports = TransportFactory.list_message_transports()
        assert "receivers" in transports
        assert "senders" in transports
        assert "uds" in transports["receivers"]
        assert "uds" in transports["senders"]

    def test_register_custom_video_reader(self):
        """Register and use custom video reader."""

        class DummyVideoReader(VideoReader):
            def __init__(self, path):
                self.path = path
                self._open = False

            def open(self, timeout_sec=None):
                self._open = True
                return True

            def read(self):
                return None

            def close(self):
                self._open = False

            @property
            def is_open(self):
                return self._open

        TransportFactory.register_video_reader("dummy", DummyVideoReader)
        reader = TransportFactory.create_video_reader("dummy", "/tmp/test")
        assert isinstance(reader, DummyVideoReader)
        assert reader.path == "/tmp/test"

    def test_register_custom_message_sender(self):
        """Register and use custom message sender."""

        class DummyMessageSender(MessageSender):
            def __init__(self, path):
                self.path = path
                self._connected = False

            def connect(self):
                self._connected = True
                return True

            def send(self, message):
                return True

            def disconnect(self):
                self._connected = False

            @property
            def is_connected(self):
                return self._connected

        TransportFactory.register_message_sender("dummy", DummyMessageSender)
        sender = TransportFactory.create_message_sender("dummy", "/tmp/test")
        assert isinstance(sender, DummyMessageSender)


class TestInterfaceContextManagers:
    """Tests for interface context manager support."""

    def test_video_reader_context_manager(self):
        """VideoReader supports context manager via ABC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo_path = os.path.join(tmpdir, "test.mjpg")

            # Setup writer
            writer = FIFOVideoWriter(fifo_path)
            writer_thread = threading.Thread(target=writer.open)
            writer_thread.start()

            time.sleep(0.1)

            # Use reader as context manager (via ABC __enter__/__exit__)
            reader: VideoReader = FIFOVideoReader(fifo_path)
            with reader:
                assert reader.is_open
            assert not reader.is_open

            writer.close()
            writer_thread.join(timeout=1.0)

    def test_message_receiver_context_manager(self):
        """MessageReceiver supports context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")

            server: MessageReceiver = UDSServer(sock_path)
            with server:
                assert server.is_running
            assert not server.is_running

    def test_message_sender_context_manager(self):
        """MessageSender supports context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")

            with UDSServer(sock_path):
                sender: MessageSender = UDSClient(sock_path)
                with sender:
                    assert sender.is_connected
                assert not sender.is_connected
