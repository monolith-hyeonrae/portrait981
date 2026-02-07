"""Shared test helpers for facemoment tests."""

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np


def create_test_video(path: Path, num_frames: int = 30, fps: int = 30) -> None:
    """Create a test video file.

    Args:
        path: Output path for the video file.
        num_frames: Number of frames to generate.
        fps: Frame rate of the video.
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 3) % 256
        frame[:, :, 1] = (i * 2) % 256
        frame[:, :, 2] = (i * 1) % 256
        cv2.putText(
            frame, f"F{i}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        writer.write(frame)

    writer.release()


def create_mock_frame(frame_id: int = 0, t_ns: int = 0):
    """Create a mock Frame object.

    Args:
        frame_id: Frame identifier.
        t_ns: Timestamp in nanoseconds.

    Returns:
        Mock frame with frame_id, t_src_ns, and data attributes.
    """
    frame = Mock()
    frame.frame_id = frame_id
    frame.t_src_ns = t_ns
    frame.data = np.zeros((240, 320, 3), dtype=np.uint8)
    return frame
