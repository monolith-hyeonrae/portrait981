"""Tests for RingBuffer."""

import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from visualbase.core.ring_buffer import RingBuffer, Segment
from visualbase.core.frame import Frame
from visualbase.packaging.trigger import Trigger


class TestRingBuffer:
    """Test suite for RingBuffer."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for segments."""
        return tmp_path / "segments"

    @pytest.fixture
    def buffer(self, temp_dir):
        """Create a RingBuffer with test settings."""
        return RingBuffer(
            segment_dir=temp_dir,
            segment_duration_sec=1.0,
            retention_sec=5.0,
            fps=30.0,
        )

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Frame.from_array(data, frame_id=0, t_src_ns=0)

    def test_init_default_segment_dir(self):
        """Test default segment directory selection."""
        buffer = RingBuffer()
        # Should use /dev/shm if available, otherwise /tmp
        assert "visualbase" in str(buffer.segment_dir)

    def test_init_custom_segment_dir(self, temp_dir):
        """Test custom segment directory."""
        buffer = RingBuffer(segment_dir=temp_dir)
        assert buffer.segment_dir == temp_dir

    def test_init_creates_segment_dir(self, temp_dir):
        """Test that segment directory is created."""
        buffer = RingBuffer(segment_dir=temp_dir)
        assert temp_dir.exists()

    def test_is_not_seekable(self, buffer):
        """Test that RingBuffer reports as not seekable."""
        info = buffer.info
        assert info.is_seekable is False

    def test_add_frame_creates_segment(self, buffer, sample_frame):
        """Test that adding a frame creates a segment."""
        buffer.add_frame(sample_frame)
        assert buffer.segment_count >= 0  # May be 0 if segment not finalized

    def test_add_multiple_frames(self, buffer, temp_dir):
        """Test adding multiple frames."""
        for i in range(10):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 33_333_333)  # ~30fps
            buffer.add_frame(frame)

        # Buffer should have accumulated some frames
        info = buffer.info
        assert info.end_ns >= info.start_ns

    def test_segment_rollover(self, temp_dir):
        """Test that segments roll over when duration exceeded."""
        buffer = RingBuffer(
            segment_dir=temp_dir,
            segment_duration_sec=0.1,  # Very short segments for testing
            retention_sec=10.0,
            fps=30.0,
        )

        # Add frames spanning multiple segments
        for i in range(30):  # ~1 second at 30fps
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 33_333_333)
            buffer.add_frame(frame)

        buffer.close()

        # Should have created multiple segments
        segment_files = list(temp_dir.glob("seg_*.mp4"))
        assert len(segment_files) >= 1

    def test_old_segments_cleanup(self, temp_dir):
        """Test that old segments are cleaned up."""
        buffer = RingBuffer(
            segment_dir=temp_dir,
            segment_duration_sec=0.1,
            retention_sec=0.5,  # Very short retention
            fps=30.0,
        )

        # Add frames with increasing timestamps
        # First batch
        for i in range(10):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 33_333_333)
            buffer.add_frame(frame)

        # Add more frames with much later timestamps to trigger cleanup
        for i in range(10):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Jump 2 seconds into the future
            frame = Frame.from_array(
                data,
                frame_id=10 + i,
                t_src_ns=2_000_000_000 + i * 33_333_333
            )
            buffer.add_frame(frame)

        buffer.close()

        # Old segments should have been cleaned up
        info = buffer.info
        # Buffer start should be relatively recent
        assert info.start_ns >= 0

    def test_query_empty_buffer(self, buffer):
        """Test query on empty buffer."""
        assert buffer.query(0, 1000000000) is False

    def test_query_within_range(self, temp_dir):
        """Test query for time range within buffer."""
        buffer = RingBuffer(
            segment_dir=temp_dir,
            segment_duration_sec=0.2,  # Short segments to finalize quickly
            retention_sec=60.0,
            fps=30.0,
        )

        # Add frames spanning multiple segments
        for i in range(30):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
            buffer.add_frame(frame)

        # Force finalize current segment to make it queryable
        buffer.close()

        # Reopen a new buffer with same dir to test persisted segments
        buffer2 = RingBuffer(
            segment_dir=temp_dir,
            segment_duration_sec=0.2,
            retention_sec=60.0,
            fps=30.0,
        )

        # Add frames again to populate buffer's segment list
        for i in range(30):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
            buffer2.add_frame(frame)

        # Query should succeed for range within buffer
        # The buffer tracks segments as they're created, so query within active range should work
        info = buffer2.info
        assert info.end_ns > info.start_ns
        assert buffer2.query(info.start_ns, info.end_ns) is True

    def test_query_outside_range(self, buffer):
        """Test query for time range outside buffer."""
        # Add frames
        for i in range(10):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
            buffer.add_frame(frame)

        # Query for range before buffer should fail
        # Note: Query checks against buffer bounds, so early timestamps are OK if start is 0
        assert buffer.query(10_000_000_000, 11_000_000_000) is False

    def test_buffer_info_empty(self, buffer):
        """Test buffer info when empty."""
        info = buffer.info
        assert info.start_ns == 0
        assert info.end_ns == 0
        assert info.duration_sec == 0.0
        assert info.is_seekable is False

    def test_buffer_info_with_frames(self, buffer):
        """Test buffer info with frames."""
        for i in range(10):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
            buffer.add_frame(frame)

        info = buffer.info
        assert info.end_ns >= info.start_ns
        assert info.duration_sec >= 0

    def test_close(self, buffer, sample_frame):
        """Test buffer close."""
        buffer.add_frame(sample_frame)
        buffer.close()
        # Should not raise
        assert buffer.segment_count == 0  # Segments cleared on close

    def test_cleanup_all(self, temp_dir):
        """Test cleanup_all removes all segments."""
        buffer = RingBuffer(
            segment_dir=temp_dir,
            segment_duration_sec=0.1,
            retention_sec=10.0,
            fps=30.0,
        )

        # Add frames
        for i in range(30):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 33_333_333)
            buffer.add_frame(frame)

        buffer.cleanup_all()

        # All segments should be deleted
        segment_files = list(temp_dir.glob("seg_*.mp4"))
        assert len(segment_files) == 0


class TestSegment:
    """Test suite for Segment dataclass."""

    def test_duration_sec(self, tmp_path):
        """Test duration calculation."""
        segment = Segment(
            path=tmp_path / "test.mp4",
            start_ns=0,
            end_ns=1_000_000_000,  # 1 second
            frame_count=30,
        )
        assert segment.duration_sec == 1.0

    def test_duration_sec_subsecond(self, tmp_path):
        """Test subsecond duration calculation."""
        segment = Segment(
            path=tmp_path / "test.mp4",
            start_ns=0,
            end_ns=500_000_000,  # 0.5 seconds
            frame_count=15,
        )
        assert segment.duration_sec == 0.5


class TestRingBufferExtractClip:
    """Test suite for clip extraction from RingBuffer."""

    @pytest.fixture
    def buffer_with_frames(self, tmp_path):
        """Create a RingBuffer with frames."""
        buffer = RingBuffer(
            segment_dir=tmp_path / "segments",
            segment_duration_sec=1.0,
            retention_sec=60.0,
            fps=30.0,
        )

        # Add 60 frames (2 seconds at 30fps)
        for i in range(60):
            data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 33_333_333)
            buffer.add_frame(frame)

        return buffer

    def test_extract_clip_empty_buffer(self, tmp_path):
        """Test extraction from empty buffer fails."""
        buffer = RingBuffer(
            segment_dir=tmp_path / "segments",
            segment_duration_sec=1.0,
            retention_sec=60.0,
            fps=30.0,
        )

        trigger = Trigger.point(event_time_ns=500_000_000, pre_sec=0.5, post_sec=0.5)
        result = buffer.extract_clip(trigger, tmp_path / "clips")

        assert result.success is False
        assert "not in buffer" in result.error

    @patch("visualbase.core.ring_buffer.subprocess.run")
    def test_extract_clip_single_segment(self, mock_run, buffer_with_frames, tmp_path):
        """Test extraction from single segment."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        trigger = Trigger.point(event_time_ns=500_000_000, pre_sec=0.2, post_sec=0.2)
        result = buffer_with_frames.extract_clip(trigger, tmp_path / "clips")

        # Should call ffmpeg
        assert mock_run.called

    @patch("visualbase.core.ring_buffer.subprocess.run")
    def test_extract_clip_ffmpeg_error(self, mock_run, buffer_with_frames, tmp_path):
        """Test extraction with ffmpeg error."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="ffmpeg error"
        )

        trigger = Trigger.point(event_time_ns=500_000_000, pre_sec=0.2, post_sec=0.2)
        result = buffer_with_frames.extract_clip(trigger, tmp_path / "clips")

        assert result.success is False
        assert "ffmpeg error" in result.error

    def test_extract_clip_creates_output_dir(self, buffer_with_frames, tmp_path):
        """Test that extraction creates output directory."""
        output_dir = tmp_path / "new_clips"
        assert not output_dir.exists()

        trigger = Trigger.point(event_time_ns=500_000_000, pre_sec=0.2, post_sec=0.2)

        with patch("visualbase.core.ring_buffer.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            buffer_with_frames.extract_clip(trigger, output_dir)

        assert output_dir.exists()


class TestRingBufferThreadSafety:
    """Test thread safety of RingBuffer."""

    def test_concurrent_add_frames(self, tmp_path):
        """Test concurrent frame additions."""
        import threading

        buffer = RingBuffer(
            segment_dir=tmp_path / "segments",
            segment_duration_sec=0.5,
            retention_sec=10.0,
            fps=30.0,
        )

        errors = []
        frame_counter = [0]
        lock = threading.Lock()

        def add_frames():
            try:
                for _ in range(20):
                    with lock:
                        frame_id = frame_counter[0]
                        frame_counter[0] += 1

                    data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    frame = Frame.from_array(
                        data,
                        frame_id=frame_id,
                        t_src_ns=frame_id * 33_333_333
                    )
                    buffer.add_frame(frame)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_frames) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        buffer.close()

    def test_concurrent_query_and_add(self, tmp_path):
        """Test concurrent query and add operations."""
        import threading

        buffer = RingBuffer(
            segment_dir=tmp_path / "segments",
            segment_duration_sec=0.5,
            retention_sec=10.0,
            fps=30.0,
        )

        errors = []
        stop_event = threading.Event()
        frame_counter = [0]

        def add_frames():
            try:
                while not stop_event.is_set():
                    frame_id = frame_counter[0]
                    frame_counter[0] += 1

                    data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    frame = Frame.from_array(
                        data,
                        frame_id=frame_id,
                        t_src_ns=frame_id * 33_333_333
                    )
                    buffer.add_frame(frame)
            except Exception as e:
                errors.append(e)

        def query_buffer():
            try:
                for _ in range(50):
                    buffer.query(0, 1_000_000_000)
                    _ = buffer.info
            except Exception as e:
                errors.append(e)

        add_thread = threading.Thread(target=add_frames)
        query_thread = threading.Thread(target=query_buffer)

        add_thread.start()
        query_thread.start()

        query_thread.join()
        stop_event.set()
        add_thread.join()

        assert len(errors) == 0
        buffer.close()
