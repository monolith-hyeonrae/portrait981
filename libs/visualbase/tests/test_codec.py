"""Tests for Frame codec (encode/decode)."""

import base64

import cv2
import numpy as np
import pytest

from visualbase.core.frame import Frame
from visualbase.ipc.codec import encode_frame, decode_frame


class TestEncodeFrame:
    """Tests for encode_frame."""

    def test_encode_basic(self):
        """Test basic frame encoding."""
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=42, t_src_ns=123456789)

        encoded = encode_frame(frame, jpeg_quality=95)

        assert encoded["frame_id"] == 42
        assert encoded["t_src_ns"] == 123456789
        assert encoded["width"] == 100
        assert encoded["height"] == 100
        assert "data_b64" in encoded

        # Verify valid base64
        decoded_bytes = base64.b64decode(encoded["data_b64"])
        assert len(decoded_bytes) > 0

    def test_encode_different_quality(self):
        """Test encoding with different JPEG quality levels."""
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=1, t_src_ns=0)

        low_q = encode_frame(frame, jpeg_quality=10)
        high_q = encode_frame(frame, jpeg_quality=100)

        # Lower quality should produce smaller data
        low_size = len(low_q["data_b64"])
        high_size = len(high_q["data_b64"])
        assert low_size < high_size


class TestDecodeFrame:
    """Tests for decode_frame."""

    def test_decode_basic(self):
        """Test basic frame decoding."""
        data = np.zeros((50, 50, 3), dtype=np.uint8)
        data[10:40, 10:40, 2] = 255  # Red square (BGR)
        frame = Frame.from_array(data, frame_id=7, t_src_ns=999)

        encoded = encode_frame(frame, jpeg_quality=100)
        decoded = decode_frame(encoded)

        assert decoded.frame_id == 7
        assert decoded.t_src_ns == 999
        assert decoded.width == 50
        assert decoded.height == 50
        # JPEG is lossy, check approximate values
        assert decoded.data[20, 20, 2] > 200

    def test_decode_invalid_data(self):
        """Test decoding with invalid base64 data."""
        bad_data = {
            "frame_id": 1,
            "t_src_ns": 0,
            "width": 10,
            "height": 10,
            "data_b64": base64.b64encode(b"not_a_jpeg").decode("ascii"),
        }

        with pytest.raises(ValueError, match="Failed to decode"):
            decode_frame(bad_data)


class TestCodecRoundTrip:
    """Tests for encode/decode round-trip."""

    def test_round_trip_preserves_metadata(self):
        """Test that round-trip preserves frame metadata."""
        data = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=42, t_src_ns=1_000_000_000)

        encoded = encode_frame(frame, jpeg_quality=95)
        decoded = decode_frame(encoded)

        assert decoded.frame_id == 42
        assert decoded.t_src_ns == 1_000_000_000
        assert decoded.width == 300
        assert decoded.height == 200

    def test_round_trip_preserves_image_approximately(self):
        """Test that round-trip preserves image data approximately (JPEG is lossy)."""
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        data[:50, :, 0] = 255  # Top half blue
        data[50:, :, 2] = 255  # Bottom half red
        frame = Frame.from_array(data, frame_id=1, t_src_ns=0)

        encoded = encode_frame(frame, jpeg_quality=100)
        decoded = decode_frame(encoded)

        # Check approximate color preservation
        assert decoded.data[25, 50, 0] > 200  # Blue channel in top half
        assert decoded.data[75, 50, 2] > 200  # Red channel in bottom half

    def test_round_trip_large_frame(self):
        """Test round-trip with a larger frame."""
        data = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=100, t_src_ns=5_000_000_000)

        encoded = encode_frame(frame, jpeg_quality=80)
        decoded = decode_frame(encoded)

        assert decoded.frame_id == 100
        assert decoded.width == 1920
        assert decoded.height == 1080
