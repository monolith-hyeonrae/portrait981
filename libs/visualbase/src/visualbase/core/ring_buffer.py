"""Ring buffer for streaming video sources.

Stores video segments in a memory-mapped directory (tmpfs) for efficient
24/7 operation with bounded memory usage.
"""

import os
import time
import logging
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Deque

import cv2
import numpy as np

from visualbase.core.buffer import BaseBuffer, BufferInfo
from visualbase.core.frame import Frame
from visualbase.core.timestamp import ns_to_seconds, seconds_to_ns
from visualbase.packaging.clipper import ClipResult
from visualbase.packaging.trigger import Trigger

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A video segment in the ring buffer.

    Attributes:
        path: Path to the segment file.
        start_ns: Start timestamp in nanoseconds.
        end_ns: End timestamp in nanoseconds.
        frame_count: Number of frames in the segment.
    """

    path: Path
    start_ns: int
    end_ns: int
    frame_count: int

    @property
    def duration_sec(self) -> float:
        """Segment duration in seconds."""
        return ns_to_seconds(self.end_ns - self.start_ns)


class RingBuffer(BaseBuffer):
    """Ring buffer for streaming video sources using segment files.

    Stores video as sequential segment files in a memory-mapped directory
    (typically /dev/shm or tmpfs). Old segments are automatically deleted
    when the retention limit is reached.

    The buffer stores raw frames and writes them to segments using OpenCV's
    VideoWriter. This approach is simpler than MPEG-TS segments and works
    well for local/memory storage.

    Args:
        segment_dir: Directory for segment files (default: /dev/shm/visualbase/).
        segment_duration_sec: Target duration for each segment in seconds (default: 2.0).
        retention_sec: Total retention time in seconds (default: 120.0).
        fps: Frame rate for encoding segments (default: 30.0).
        codec: FourCC codec for segments (default: "mp4v").

    Example:
        >>> buffer = RingBuffer(retention_sec=60.0)
        >>> for frame in camera_stream:
        ...     buffer.add_frame(frame)
        ...     if should_trigger:
        ...         trigger = Trigger.point(event_time_ns=frame.t_src_ns, pre_sec=5.0, post_sec=2.0)
        ...         result = buffer.extract_clip(trigger, output_dir)
    """

    def __init__(
        self,
        segment_dir: Optional[Path] = None,
        segment_duration_sec: float = 2.0,
        retention_sec: float = 120.0,
        fps: float = 30.0,
        codec: str = "mp4v",
    ):
        # Use /dev/shm if available (Linux tmpfs), otherwise fallback to /tmp
        if segment_dir is None:
            if Path("/dev/shm").exists():
                segment_dir = Path("/dev/shm/visualbase")
            else:
                segment_dir = Path("/tmp/visualbase")

        self._segment_dir = Path(segment_dir)
        self._segment_duration_sec = segment_duration_sec
        self._retention_sec = retention_sec
        self._fps = fps
        self._codec = codec

        # Segment management
        self._segments: Deque[Segment] = deque()
        self._current_segment: Optional[_SegmentWriter] = None
        self._segment_counter: int = 0

        # Thread safety
        self._lock = threading.Lock()

        # Frame dimensions (set on first frame)
        self._width: int = 0
        self._height: int = 0

        # Ensure segment directory exists
        self._segment_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the ring buffer.

        Creates new segments as needed and cleans up old segments that
        exceed the retention time.

        Args:
            frame: Frame to add to the buffer.
        """
        with self._lock:
            # Initialize dimensions from first frame
            if self._width == 0:
                self._width = frame.width
                self._height = frame.height

            # Create new segment if needed
            if self._current_segment is None:
                self._start_new_segment(frame.t_src_ns)

            # Check if current segment is full
            if self._current_segment.duration_sec >= self._segment_duration_sec:
                self._finalize_current_segment()
                self._start_new_segment(frame.t_src_ns)

            # Write frame to current segment
            self._current_segment.write(frame)

            # Clean up old segments
            self._cleanup_old_segments(frame.t_src_ns)

    def _start_new_segment(self, start_ns: int) -> None:
        """Start a new segment file."""
        self._segment_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"seg_{timestamp}_{self._segment_counter:06d}.mp4"
        path = self._segment_dir / filename

        self._current_segment = _SegmentWriter(
            path=path,
            start_ns=start_ns,
            fps=self._fps,
            width=self._width,
            height=self._height,
            codec=self._codec,
        )
        logger.debug(f"Started new segment: {path}")

    def _finalize_current_segment(self) -> None:
        """Finalize and close the current segment."""
        if self._current_segment is None:
            return

        segment = self._current_segment.finalize()
        if segment is not None and segment.frame_count > 0:
            self._segments.append(segment)
            logger.debug(
                f"Finalized segment: {segment.path.name}, "
                f"frames={segment.frame_count}, duration={segment.duration_sec:.2f}s"
            )

        self._current_segment = None

    def _cleanup_old_segments(self, current_ns: int) -> None:
        """Remove segments that exceed retention time."""
        cutoff_ns = current_ns - seconds_to_ns(self._retention_sec)

        while self._segments and self._segments[0].end_ns < cutoff_ns:
            old_segment = self._segments.popleft()
            try:
                old_segment.path.unlink(missing_ok=True)
                logger.debug(f"Deleted old segment: {old_segment.path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete segment {old_segment.path}: {e}")

    def query(self, start_ns: int, end_ns: int) -> bool:
        """Check if the requested time range is available in the buffer.

        Args:
            start_ns: Start timestamp in nanoseconds.
            end_ns: End timestamp in nanoseconds.

        Returns:
            True if the range is fully available in the buffer.
        """
        with self._lock:
            if not self._segments:
                return False

            # Check if range is within buffer bounds
            buffer_start = self._segments[0].start_ns
            buffer_end = self._segments[-1].end_ns

            # Include current segment if it exists
            if self._current_segment is not None:
                buffer_end = max(buffer_end, self._current_segment.end_ns)

            return start_ns >= buffer_start and end_ns <= buffer_end

    def extract_clip(self, trigger: Trigger, output_dir: Path) -> ClipResult:
        """Extract a clip from the ring buffer based on trigger.

        Concatenates relevant segments and extracts the requested time range
        using ffmpeg.

        Args:
            trigger: Trigger defining clip boundaries.
            output_dir: Directory to save the extracted clip.

        Returns:
            ClipResult with extraction status and output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Finalize current segment to ensure all frames are written
            if self._current_segment is not None:
                self._finalize_current_segment()

            # Find segments that overlap with the trigger time range
            clip_start_ns = trigger.clip_start_ns
            clip_end_ns = trigger.clip_end_ns

            relevant_segments = []
            for segment in self._segments:
                # Check for overlap
                if segment.start_ns < clip_end_ns and segment.end_ns > clip_start_ns:
                    relevant_segments.append(segment)

            if not relevant_segments:
                return ClipResult(
                    success=False,
                    output_path=None,
                    trigger=trigger,
                    error="Requested time range not in buffer",
                )

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = trigger.label or "clip"
        score_str = f"_{trigger.score:.2f}" if trigger.score > 0 else ""
        output_filename = f"{timestamp}_{label}{score_str}.mp4"
        output_path = output_dir / output_filename

        # Extract using ffmpeg
        try:
            if len(relevant_segments) == 1:
                # Single segment - direct extraction
                result = self._extract_from_single_segment(
                    relevant_segments[0], trigger, output_path
                )
            else:
                # Multiple segments - concatenate then extract
                result = self._extract_from_multiple_segments(
                    relevant_segments, trigger, output_path
                )

            return result

        except Exception as e:
            logger.error(f"Clip extraction failed: {e}")
            return ClipResult(
                success=False,
                output_path=None,
                trigger=trigger,
                error=str(e),
            )

    def _extract_from_single_segment(
        self, segment: Segment, trigger: Trigger, output_path: Path
    ) -> ClipResult:
        """Extract clip from a single segment."""
        # Calculate relative start time within segment
        segment_start_sec = ns_to_seconds(segment.start_ns)
        clip_start_sec = max(0, trigger.clip_start_sec - segment_start_sec)
        duration_sec = trigger.clip_duration_sec

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(clip_start_sec),
            "-i", str(segment.path),
            "-t", str(duration_sec),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return ClipResult(
                success=False,
                output_path=None,
                trigger=trigger,
                error=f"ffmpeg error: {result.stderr}",
            )

        actual_duration = self._get_duration(output_path)
        return ClipResult(
            success=True,
            output_path=output_path,
            trigger=trigger,
            duration_sec=actual_duration,
        )

    def _extract_from_multiple_segments(
        self, segments: list[Segment], trigger: Trigger, output_path: Path
    ) -> ClipResult:
        """Extract clip by concatenating multiple segments."""
        # Create concat list file
        concat_list_path = self._segment_dir / f"concat_{os.getpid()}.txt"

        try:
            with open(concat_list_path, "w") as f:
                for segment in segments:
                    # Use absolute path with proper escaping
                    f.write(f"file '{segment.path.absolute()}'\n")

            # Calculate timing relative to first segment
            first_segment_start_sec = ns_to_seconds(segments[0].start_ns)
            clip_start_sec = max(0, trigger.clip_start_sec - first_segment_start_sec)
            duration_sec = trigger.clip_duration_sec

            # Concatenate and extract
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-ss", str(clip_start_sec),
                "-t", str(duration_sec),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return ClipResult(
                    success=False,
                    output_path=None,
                    trigger=trigger,
                    error=f"ffmpeg concat error: {result.stderr}",
                )

            actual_duration = self._get_duration(output_path)
            return ClipResult(
                success=True,
                output_path=output_path,
                trigger=trigger,
                duration_sec=actual_duration,
            )

        finally:
            # Clean up concat list
            concat_list_path.unlink(missing_ok=True)

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            import json
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0.0

    @property
    def info(self) -> BufferInfo:
        """Get buffer state information."""
        with self._lock:
            if not self._segments:
                return BufferInfo(
                    start_ns=0,
                    end_ns=0,
                    duration_sec=0.0,
                    is_seekable=False,
                )

            start_ns = self._segments[0].start_ns
            end_ns = self._segments[-1].end_ns

            # Include current segment
            if self._current_segment is not None:
                end_ns = max(end_ns, self._current_segment.end_ns)

            duration_sec = ns_to_seconds(end_ns - start_ns)

            return BufferInfo(
                start_ns=start_ns,
                end_ns=end_ns,
                duration_sec=duration_sec,
                is_seekable=False,
            )

    @property
    def segment_count(self) -> int:
        """Number of segments currently in the buffer."""
        with self._lock:
            return len(self._segments)

    @property
    def segment_dir(self) -> Path:
        """Directory containing segment files."""
        return self._segment_dir

    def close(self) -> None:
        """Close the buffer and clean up resources."""
        with self._lock:
            # Finalize current segment
            if self._current_segment is not None:
                self._finalize_current_segment()

            # Clear segments (but don't delete files - they may still be needed)
            self._segments.clear()

    def cleanup_all(self) -> None:
        """Clean up all segment files and close the buffer."""
        with self._lock:
            if self._current_segment is not None:
                self._finalize_current_segment()

            # Delete all segment files
            for segment in self._segments:
                try:
                    segment.path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete segment {segment.path}: {e}")

            self._segments.clear()

            # Clean up any remaining files in the segment directory
            try:
                for f in self._segment_dir.glob("seg_*.mp4"):
                    f.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean segment directory: {e}")


class _SegmentWriter:
    """Internal class for writing frames to a segment file."""

    def __init__(
        self,
        path: Path,
        start_ns: int,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ):
        self._path = path
        self._start_ns = start_ns
        self._fps = fps
        self._width = width
        self._height = height

        self._frame_count = 0
        self._end_ns = start_ns

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(path), fourcc, fps, (width, height)
        )

        if not self._writer.isOpened():
            raise IOError(f"Failed to create video writer: {path}")

    def write(self, frame: Frame) -> None:
        """Write a frame to the segment."""
        self._writer.write(frame.data)
        self._frame_count += 1
        self._end_ns = frame.t_src_ns

    def finalize(self) -> Optional[Segment]:
        """Finalize the segment and return its metadata."""
        self._writer.release()

        if self._frame_count == 0:
            # Delete empty segment
            self._path.unlink(missing_ok=True)
            return None

        return Segment(
            path=self._path,
            start_ns=self._start_ns,
            end_ns=self._end_ns,
            frame_count=self._frame_count,
        )

    @property
    def duration_sec(self) -> float:
        """Current segment duration in seconds."""
        return ns_to_seconds(self._end_ns - self._start_ns)

    @property
    def end_ns(self) -> int:
        """End timestamp of the current segment."""
        return self._end_ns
