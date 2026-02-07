"""Tests for FileSource."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from visualbase import VisualBase, FileSource, Frame


def create_test_video(path: Path, num_frames: int = 30, fps: int = 30) -> None:
    """Create a test video file."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with varying color based on frame number
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 8) % 256  # Blue channel
        frame[:, :, 1] = (i * 4) % 256  # Green channel
        frame[:, :, 2] = (i * 2) % 256  # Red channel
        writer.write(frame)

    writer.release()


class TestFileSource:
    def test_open_nonexistent_file(self):
        source = FileSource("/nonexistent/path/video.mp4")
        with pytest.raises(IOError, match="not found"):
            source.open()

    def test_open_and_close(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path)

        source = FileSource(video_path)
        source.open()

        assert source.fps == 30.0
        assert source.width == 320
        assert source.height == 240
        assert source.is_seekable

        source.close()

    def test_context_manager(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path)

        with FileSource(video_path) as source:
            assert source.fps == 30.0

    def test_read_frames(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        num_frames = 10
        create_test_video(video_path, num_frames=num_frames)

        with FileSource(video_path) as source:
            frames = []
            while True:
                frame = source.read()
                if frame is None:
                    break
                frames.append(frame)

            assert len(frames) == num_frames

            # Check frame properties
            for i, frame in enumerate(frames):
                assert isinstance(frame, Frame)
                assert frame.frame_id == i
                assert frame.width == 320
                assert frame.height == 240
                assert frame.data.shape == (240, 320, 3)

    def test_seek(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=60, fps=30)

        with FileSource(video_path) as source:
            # Seek to 1 second (1_000_000_000 ns)
            success = source.seek(1_000_000_000)
            assert success

            frame = source.read()
            assert frame is not None
            # After seeking to 1 second at 30fps, we should be around frame 30
            assert frame.t_src_ns >= 900_000_000  # Allow some tolerance


class TestVisualBase:
    def test_connect_disconnect(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path)

        vb = VisualBase()
        assert not vb.is_connected

        vb.connect(FileSource(video_path))
        assert vb.is_connected
        assert vb.source is not None

        vb.disconnect()
        assert not vb.is_connected
        assert vb.source is None

    def test_context_manager(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path)

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))
            assert vb.is_connected

    def test_get_stream_without_connect(self):
        vb = VisualBase()
        with pytest.raises(RuntimeError, match="No source connected"):
            list(vb.get_stream())

    def test_get_stream_original_fps(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        num_frames = 30
        create_test_video(video_path, num_frames=num_frames)

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))
            frames = list(vb.get_stream())

            assert len(frames) == num_frames

    def test_get_stream_reduced_fps(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        # 30 frames at 30fps = 1 second of video
        create_test_video(video_path, num_frames=30, fps=30)

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))
            # Request 10 fps from 30fps source = ~1/3 of frames
            frames = list(vb.get_stream(fps=10))

            # Should get approximately 10 frames (1 second * 10 fps)
            assert 8 <= len(frames) <= 12

    def test_get_stream_with_resolution(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=5)

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))
            frames = list(vb.get_stream(resolution=(160, 120)))

            for frame in frames:
                assert frame.width == 160
                assert frame.height == 120
                assert frame.data.shape == (120, 160, 3)

    def test_get_stream_fps_and_resolution(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=30, fps=30)

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))
            frames = list(vb.get_stream(fps=10, resolution=(160, 120)))

            # Check fps reduction
            assert 8 <= len(frames) <= 12

            # Check resolution
            for frame in frames:
                assert frame.width == 160
                assert frame.height == 120
