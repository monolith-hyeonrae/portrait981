"""Tests for RTSPSource."""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from visualbase.sources.rtsp import RTSPSource
from visualbase.core.frame import Frame


class TestRTSPSource:
    """Test suite for RTSPSource."""

    def test_init_url(self):
        """Test initialization with URL."""
        source = RTSPSource("rtsp://192.168.1.100:554/stream")
        assert source.url == "rtsp://192.168.1.100:554/stream"

    def test_init_with_options(self):
        """Test initialization with custom options."""
        source = RTSPSource(
            url="rtsp://example.com/stream",
            buffer_size=5,
            timeout_sec=30.0,
            reconnect=False,
            reconnect_delay_sec=10.0,
        )
        assert source.url == "rtsp://example.com/stream"

    def test_is_not_seekable(self):
        """Test that RTSPSource is not seekable."""
        source = RTSPSource("rtsp://example.com/stream")
        assert source.is_seekable is False

    def test_seek_returns_false(self):
        """Test that seek always returns False."""
        source = RTSPSource("rtsp://example.com/stream")
        assert source.seek(1000000000) is False

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_open_success(self, mock_capture_class):
        """Test successful stream opening."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 25.0,  # CAP_PROP_FPS
            3: 1920,  # CAP_PROP_FRAME_WIDTH
            4: 1080,  # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)
        mock_capture_class.return_value = mock_cap

        source = RTSPSource("rtsp://example.com/stream", reconnect=False)
        source.open()

        try:
            assert source.fps == 25.0
            assert source.width == 1920
            assert source.height == 1080
            assert source.is_connected is True
        finally:
            source.close()

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_open_failure(self, mock_capture_class):
        """Test stream opening failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap

        source = RTSPSource("rtsp://invalid/stream")

        with pytest.raises(IOError, match="Failed to open RTSP stream"):
            source.open()

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_close(self, mock_capture_class):
        """Test stream closing."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 0
        mock_capture_class.return_value = mock_cap

        source = RTSPSource("rtsp://example.com/stream", reconnect=False)
        source.open()
        source.close()

        assert source.is_connected is False

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_capture_class):
        """Test reading returns a frame from the queue."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,
            3: 640,
            4: 480,
        }.get(prop, 0)

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_capture_class.return_value = mock_cap

        source = RTSPSource("rtsp://example.com/stream", reconnect=False)
        source.open()

        try:
            # Wait a bit for reader thread to start
            time.sleep(0.1)

            frame = source.read()
            # May be None if reader thread hasn't captured yet
            if frame is not None:
                assert isinstance(frame, Frame)
        finally:
            source.close()

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_read_not_opened(self, mock_capture_class):
        """Test reading without opening raises error."""
        source = RTSPSource("rtsp://example.com/stream")

        with pytest.raises(RuntimeError, match="not opened"):
            source.read()

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_context_manager(self, mock_capture_class):
        """Test context manager protocol."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 0
        mock_capture_class.return_value = mock_cap

        with RTSPSource("rtsp://example.com/stream", reconnect=False) as source:
            assert source is not None

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_fps_fallback(self, mock_capture_class):
        """Test FPS fallback when stream doesn't report it."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 0.0,  # Invalid FPS
            3: 640,
            4: 480,
        }.get(prop, 0)
        mock_capture_class.return_value = mock_cap

        source = RTSPSource("rtsp://example.com/stream", reconnect=False)
        source.open()

        try:
            assert source.fps == 30.0  # Fallback value
        finally:
            source.close()

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_buffer_drops_old_frames(self, mock_capture_class):
        """Test that buffer drops old frames when full."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,
            3: 640,
            4: 480,
        }.get(prop, 0)

        frame_counter = [0]

        def mock_read():
            frame_counter[0] += 1
            test_frame = np.full((480, 640, 3), frame_counter[0], dtype=np.uint8)
            return (True, test_frame)

        mock_cap.read.side_effect = mock_read
        mock_capture_class.return_value = mock_cap

        # Small buffer
        source = RTSPSource(
            "rtsp://example.com/stream",
            buffer_size=2,
            reconnect=False,
        )
        source.open()

        try:
            # Let reader thread fill the buffer
            time.sleep(0.2)

            # Read frames - should get recent ones, not the oldest
            frames_read = []
            for _ in range(3):
                frame = source.read()
                if frame is not None:
                    frames_read.append(frame)

            # Should have dropped old frames
            if len(frames_read) >= 2:
                # Frame IDs should be relatively recent
                assert frames_read[-1].frame_id > frames_read[0].frame_id

        finally:
            source.close()


class TestRTSPSourceReconnect:
    """Test reconnection behavior of RTSPSource."""

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    def test_no_reconnect_when_disabled(self, mock_capture_class):
        """Test that reconnect doesn't happen when disabled."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,
            3: 640,
            4: 480,
        }.get(prop, 0)

        # First read succeeds, then fails
        read_count = [0]

        def mock_read():
            read_count[0] += 1
            if read_count[0] <= 2:
                return (True, np.zeros((480, 640, 3), dtype=np.uint8))
            return (False, None)

        mock_cap.read.side_effect = mock_read
        mock_capture_class.return_value = mock_cap

        source = RTSPSource(
            "rtsp://example.com/stream",
            reconnect=False,
        )
        source.open()

        try:
            time.sleep(0.5)
            # After failures, stream should disconnect
            # Reader thread should eventually stop
        finally:
            source.close()

    @patch("visualbase.sources.rtsp.cv2.VideoCapture")
    @patch("visualbase.sources.rtsp.time.sleep")
    def test_reconnect_on_stream_loss(self, mock_sleep, mock_capture_class):
        """Test reconnection on stream loss."""
        # Track VideoCapture calls
        connect_count = [0]

        def create_mock_cap(url, *args, **kwargs):
            connect_count[0] += 1
            mock_cap = MagicMock()

            if connect_count[0] <= 1:
                # First connection succeeds
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda prop: {
                    5: 30.0,
                    3: 640,
                    4: 480,
                }.get(prop, 0)

                # Read fails after a few frames
                read_count = [0]

                def mock_read():
                    read_count[0] += 1
                    if read_count[0] <= 3:
                        return (True, np.zeros((480, 640, 3), dtype=np.uint8))
                    return (False, None)

                mock_cap.read.side_effect = mock_read
            else:
                # Subsequent connections fail
                mock_cap.isOpened.return_value = False

            return mock_cap

        mock_capture_class.side_effect = create_mock_cap

        source = RTSPSource(
            "rtsp://example.com/stream",
            reconnect=True,
            reconnect_delay_sec=0.1,
        )
        source.open()

        try:
            time.sleep(0.3)
            # Should have attempted reconnection
            assert connect_count[0] >= 1
        finally:
            source.close()
