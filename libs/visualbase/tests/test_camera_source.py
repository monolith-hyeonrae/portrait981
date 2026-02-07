"""Tests for CameraSource."""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from visualbase.sources.camera import CameraSource
from visualbase.core.frame import Frame


class TestCameraSource:
    """Test suite for CameraSource."""

    def test_init_default_device(self):
        """Test default device ID is 0."""
        source = CameraSource()
        assert source.device_id == 0

    def test_init_custom_device(self):
        """Test custom device ID."""
        source = CameraSource(device_id=1)
        assert source.device_id == 1

    def test_init_with_settings(self):
        """Test initialization with custom settings."""
        source = CameraSource(
            device_id=0,
            width=1280,
            height=720,
            fps=30.0,
        )
        assert source.device_id == 0

    def test_is_not_seekable(self):
        """Test that CameraSource is not seekable."""
        source = CameraSource()
        assert source.is_seekable is False

    def test_seek_returns_false(self):
        """Test that seek always returns False."""
        source = CameraSource()
        assert source.seek(1000000000) is False

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_open_success(self, mock_capture_class):
        """Test successful camera opening."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            3: 640,   # CAP_PROP_FRAME_WIDTH
            4: 480,   # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)
        mock_capture_class.return_value = mock_cap

        source = CameraSource()
        source.open()

        assert source.fps == 30.0
        assert source.width == 640
        assert source.height == 480
        assert source.is_opened is True

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_open_failure(self, mock_capture_class):
        """Test camera opening failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap

        source = CameraSource()

        with pytest.raises(IOError, match="Failed to open camera"):
            source.open()

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_read_success(self, mock_capture_class):
        """Test successful frame reading."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            3: 640,   # CAP_PROP_FRAME_WIDTH
            4: 480,   # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)

        # Create mock frame data
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_capture_class.return_value = mock_cap

        source = CameraSource()
        source.open()

        frame = source.read()

        assert frame is not None
        assert isinstance(frame, Frame)
        assert frame.frame_id == 0
        assert frame.width == 640
        assert frame.height == 480
        assert frame.t_src_ns >= 0

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_read_failure(self, mock_capture_class):
        """Test frame reading failure returns None."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,
            3: 640,
            4: 480,
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)
        mock_capture_class.return_value = mock_cap

        source = CameraSource()
        source.open()

        frame = source.read()

        assert frame is None

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_read_not_opened(self, mock_capture_class):
        """Test reading without opening raises error."""
        source = CameraSource()

        with pytest.raises(RuntimeError, match="not opened"):
            source.read()

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_close(self, mock_capture_class):
        """Test camera closing."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 0
        mock_capture_class.return_value = mock_cap

        source = CameraSource()
        source.open()
        source.close()

        mock_cap.release.assert_called_once()

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_context_manager(self, mock_capture_class):
        """Test context manager protocol."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 0
        mock_capture_class.return_value = mock_cap

        with CameraSource() as source:
            assert source is not None

        mock_cap.release.assert_called_once()

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_monotonic_timestamps(self, mock_capture_class):
        """Test that timestamps are monotonically increasing."""
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

        source = CameraSource()
        source.open()

        frame1 = source.read()
        time.sleep(0.01)  # Small delay
        frame2 = source.read()

        assert frame1.t_src_ns < frame2.t_src_ns
        assert frame1.frame_id == 0
        assert frame2.frame_id == 1

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_fps_fallback(self, mock_capture_class):
        """Test FPS fallback when camera doesn't report it."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 0.0,  # Invalid FPS
            3: 640,
            4: 480,
        }.get(prop, 0)
        mock_capture_class.return_value = mock_cap

        source = CameraSource()
        source.open()

        assert source.fps == 30.0  # Fallback value

    @patch("visualbase.sources.camera.cv2.VideoCapture")
    def test_requested_settings_applied(self, mock_capture_class):
        """Test that requested settings are applied to camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 60.0,
            3: 1280,
            4: 720,
        }.get(prop, 0)
        mock_capture_class.return_value = mock_cap

        source = CameraSource(width=1280, height=720, fps=60.0)
        source.open()

        # Verify settings were applied
        mock_cap.set.assert_any_call(3, 1280)  # CAP_PROP_FRAME_WIDTH
        mock_cap.set.assert_any_call(4, 720)   # CAP_PROP_FRAME_HEIGHT
        mock_cap.set.assert_any_call(5, 60.0)  # CAP_PROP_FPS
