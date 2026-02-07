"""Tests for Trigger and Clip extraction."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from visualbase import VisualBase, FileSource, Trigger, TriggerType, ClipResult


def create_test_video(path: Path, num_frames: int = 90, fps: int = 30) -> None:
    """Create a test video file (3 seconds at 30fps)."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 3) % 256
        frame[:, :, 1] = (i * 2) % 256
        frame[:, :, 2] = (i * 1) % 256
        # Add frame number text
        cv2.putText(
            frame,
            f"F{i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    writer.release()


class TestTrigger:
    def test_point_trigger_creation(self):
        trig = Trigger.point(
            event_time_ns=5_000_000_000,  # 5 seconds
            pre_sec=2.0,
            post_sec=1.0,
        )
        assert trig.type == TriggerType.POINT
        assert trig.event_time_ns == 5_000_000_000
        assert trig.pre_sec == 2.0
        assert trig.post_sec == 1.0

    def test_point_trigger_clip_bounds(self):
        trig = Trigger.point(
            event_time_ns=5_000_000_000,  # 5 seconds
            pre_sec=2.0,
            post_sec=1.0,
        )
        # Clip should be [3s, 6s]
        assert trig.clip_start_ns == 3_000_000_000
        assert trig.clip_end_ns == 6_000_000_000
        assert trig.clip_start_sec == 3.0
        assert trig.clip_end_sec == 6.0
        assert trig.clip_duration_sec == 3.0

    def test_range_trigger_creation(self):
        trig = Trigger.range(
            start_time_ns=5_000_000_000,
            end_time_ns=10_000_000_000,
            pre_sec=2.0,
            post_sec=1.0,
        )
        assert trig.type == TriggerType.RANGE
        assert trig.start_time_ns == 5_000_000_000
        assert trig.end_time_ns == 10_000_000_000

    def test_range_trigger_clip_bounds(self):
        trig = Trigger.range(
            start_time_ns=5_000_000_000,  # 5 seconds
            end_time_ns=10_000_000_000,  # 10 seconds
            pre_sec=2.0,
            post_sec=1.0,
        )
        # Clip should be [3s, 11s]
        assert trig.clip_start_ns == 3_000_000_000
        assert trig.clip_end_ns == 11_000_000_000
        assert trig.clip_duration_sec == 8.0

    def test_point_trigger_requires_event_time(self):
        with pytest.raises(ValueError, match="event_time_ns"):
            Trigger(type=TriggerType.POINT)

    def test_range_trigger_requires_both_times(self):
        with pytest.raises(ValueError, match="start_time_ns and end_time_ns"):
            Trigger(type=TriggerType.RANGE, start_time_ns=1000)

    def test_range_trigger_validates_order(self):
        with pytest.raises(ValueError, match="start_time_ns must be"):
            Trigger.range(
                start_time_ns=10_000_000_000,
                end_time_ns=5_000_000_000,  # end before start
            )

    def test_trigger_with_metadata(self):
        trig = Trigger.point(
            event_time_ns=5_000_000_000,
            label="smile",
            score=0.85,
            metadata={"face_id": 1, "reason": "expression_spike"},
        )
        assert trig.label == "smile"
        assert trig.score == 0.85
        assert trig.metadata["face_id"] == 1

    def test_clip_start_clamps_to_zero(self):
        # Event at 1s with 3s pre should clamp to 0
        trig = Trigger.point(event_time_ns=1_000_000_000, pre_sec=3.0)
        assert trig.clip_start_ns == 0


class TestClipExtraction:
    def test_extract_point_clip(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=90, fps=30)  # 3 seconds

        with VisualBase(clip_output_dir=output_dir) as vb:
            vb.connect(FileSource(video_path))

            # Extract clip around 1.5s mark
            trig = Trigger.point(
                event_time_ns=1_500_000_000,  # 1.5s
                pre_sec=0.5,
                post_sec=0.5,
                label="test_clip",
            )

            result = vb.trigger(trig)

            assert result.success, f"Extraction failed: {result.error}"
            assert result.output_path is not None
            assert result.output_path.exists()
            assert result.duration_sec > 0

    def test_extract_range_clip(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=90, fps=30)  # 3 seconds

        with VisualBase(clip_output_dir=output_dir) as vb:
            vb.connect(FileSource(video_path))

            # Extract clip from 0.5s to 1.5s
            trig = Trigger.range(
                start_time_ns=500_000_000,  # 0.5s
                end_time_ns=1_500_000_000,  # 1.5s
                pre_sec=0.2,
                post_sec=0.2,
                label="range_test",
            )

            result = vb.trigger(trig)

            assert result.success, f"Extraction failed: {result.error}"
            assert result.output_path.exists()

    def test_buffer_info(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=90, fps=30)  # 3 seconds

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))

            info = vb.get_buffer_info()

            assert info.start_ns == 0
            assert info.duration_sec == pytest.approx(3.0, rel=0.1)
            assert info.is_seekable is True

    def test_query_buffer(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=90, fps=30)  # 3 seconds

        with VisualBase() as vb:
            vb.connect(FileSource(video_path))

            # Valid range
            assert vb.query_buffer(0, 1_000_000_000) is True

            # Range beyond video
            assert vb.query_buffer(0, 10_000_000_000) is False

            # Negative start
            assert vb.query_buffer(-1, 1_000_000_000) is False

    def test_trigger_without_connection(self):
        vb = VisualBase()
        trig = Trigger.point(event_time_ns=1_000_000_000)

        with pytest.raises(RuntimeError, match="No buffer available"):
            vb.trigger(trig)

    def test_custom_output_dir(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        default_dir = tmp_path / "default_clips"
        custom_dir = tmp_path / "custom_clips"
        create_test_video(video_path, num_frames=90, fps=30)

        with VisualBase(clip_output_dir=default_dir) as vb:
            vb.connect(FileSource(video_path))

            trig = Trigger.point(event_time_ns=1_000_000_000, pre_sec=0.3, post_sec=0.3)

            # Use custom output dir
            result = vb.trigger(trig, output_dir=custom_dir)

            assert result.success
            assert custom_dir in result.output_path.parents or result.output_path.parent == custom_dir
