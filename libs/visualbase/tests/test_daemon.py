"""Tests for VideoDaemon."""

import threading
import time

import pytest
import numpy as np

from visualbase.daemon import VideoDaemon
from visualbase.core.frame import Frame


class MockVideoSource:
    """Mock video source for testing."""

    def __init__(self, frame_count: int = 100, fps: float = 30.0):
        self._frame_count = frame_count
        self._fps = fps
        self._width = 640
        self._height = 480
        self._current = 0
        self._is_open = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> bool:
        self._is_open = True
        self._current = 0
        return True

    def close(self) -> None:
        self._is_open = False

    def read(self):
        if not self._is_open or self._current >= self._frame_count:
            return None
        data = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame = Frame.from_array(
            data=data,
            frame_id=self._current,
            t_src_ns=int(self._current * 1e9 / self._fps),
        )
        self._current += 1
        return frame


class MockVideoWriter:
    """Mock video writer for testing."""

    def __init__(self):
        self._is_open = False
        self._frames = []

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def write(self, frame: Frame) -> bool:
        if not self._is_open:
            return False
        self._frames.append(frame)
        return True


class TestVideoDaemon:
    """Tests for VideoDaemon class."""

    def test_daemon_creation(self):
        """Test daemon can be created."""
        source = MockVideoSource()
        writer = MockVideoWriter()
        daemon = VideoDaemon(
            source=source,
            pub_address="tcp://*:15580",
            publisher=writer,
        )

        assert daemon.source is source
        assert daemon.pub_address == "tcp://*:15580"
        assert not daemon.is_running
        assert daemon.frame_count == 0

    def test_daemon_run_and_stop(self):
        """Test daemon can run and be stopped."""
        source = MockVideoSource(frame_count=1000)
        writer = MockVideoWriter()
        daemon = VideoDaemon(
            source=source,
            pub_address="tcp://*:15581",
            fps=100,  # High FPS to process quickly
            publisher=writer,
        )

        # Run daemon in background thread
        def run_daemon():
            daemon.run()

        thread = threading.Thread(target=run_daemon)
        thread.start()

        # Wait a bit then stop
        time.sleep(0.1)
        daemon.stop()
        thread.join(timeout=2.0)

        assert not daemon.is_running
        assert daemon.frame_count > 0
        assert len(writer._frames) > 0

    def test_daemon_stats(self):
        """Test daemon statistics."""
        source = MockVideoSource(frame_count=50)
        writer = MockVideoWriter()
        daemon = VideoDaemon(
            source=source,
            pub_address="tcp://*:15582",
            fps=1000,  # Very high FPS to not limit
            publisher=writer,
        )

        # Run daemon in background thread
        def run_daemon():
            daemon.run()

        thread = threading.Thread(target=run_daemon)
        thread.start()

        # Wait for all frames to be processed
        time.sleep(0.2)
        thread.join(timeout=2.0)

        stats = daemon.get_stats()
        assert stats["frame_count"] == 50
        assert stats["elapsed_sec"] > 0
        assert stats["fps"] > 0

    def test_daemon_fps_limiting(self):
        """Test daemon respects FPS limit."""
        source = MockVideoSource(frame_count=1000)
        writer = MockVideoWriter()
        daemon = VideoDaemon(
            source=source,
            pub_address="tcp://*:15583",
            fps=50,  # Limit to 50 FPS
            publisher=writer,
        )

        # Run for a short time
        def run_daemon():
            daemon.run()

        thread = threading.Thread(target=run_daemon)
        thread.start()

        time.sleep(0.15)  # Run for 150ms
        daemon.stop()
        thread.join(timeout=2.0)

        # At 50 FPS, 150ms should yield ~7-8 frames
        # Allow some variance due to timing
        assert 3 <= daemon.frame_count <= 15


class TestDaemonWithZMQ:
    """Tests for daemon with actual ZMQ transport."""

    @pytest.fixture
    def zmq_available(self):
        """Check if ZMQ is available."""
        try:
            import zmq
            return True
        except ImportError:
            pytest.skip("pyzmq not installed")

    def test_daemon_publishes_frames(self, zmq_available):
        """Test daemon can publish frames via ZMQ."""
        from visualbase.ipc.zmq_transport import ZMQVideoSubscriber

        address = "tcp://127.0.0.1:15584"
        source = MockVideoSource(frame_count=100)

        daemon = VideoDaemon(
            source=source,
            pub_address=address,
            fps=100,
        )

        # Create subscriber
        sub = ZMQVideoSubscriber(address, timeout_ms=1000)
        sub.open()

        # Run daemon in background
        def run_daemon():
            daemon.run()

        thread = threading.Thread(target=run_daemon)
        thread.start()

        # Wait for connection
        time.sleep(0.2)

        # Try to receive a frame
        received_frames = []
        for _ in range(5):
            frame = sub.read()
            if frame:
                received_frames.append(frame)

        # Stop daemon
        daemon.stop()
        thread.join(timeout=2.0)
        sub.close()

        # Should have received some frames
        assert len(received_frames) > 0

    def test_multiple_subscribers(self, zmq_available):
        """Test daemon can serve multiple subscribers."""
        from visualbase.ipc.zmq_transport import ZMQVideoSubscriber

        address = "tcp://127.0.0.1:15585"
        source = MockVideoSource(frame_count=100)

        daemon = VideoDaemon(
            source=source,
            pub_address=address,
            fps=50,
        )

        # Create multiple subscribers
        subs = []
        for _ in range(3):
            sub = ZMQVideoSubscriber(address, timeout_ms=1000)
            sub.open()
            subs.append(sub)

        # Run daemon
        def run_daemon():
            daemon.run()

        thread = threading.Thread(target=run_daemon)
        thread.start()

        time.sleep(0.3)

        # All subscribers should receive frames
        received_counts = []
        for sub in subs:
            frames = []
            for _ in range(3):
                frame = sub.read()
                if frame:
                    frames.append(frame)
            received_counts.append(len(frames))

        daemon.stop()
        thread.join(timeout=2.0)

        for sub in subs:
            sub.close()

        # Each subscriber should have received frames
        for count in received_counts:
            assert count > 0
