"""Buffer management for video sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from visualbase.packaging.trigger import Trigger
from visualbase.packaging.clipper import Clipper, ClipResult


@dataclass
class BufferInfo:
    """Information about buffer state.

    Attributes:
        start_ns: Earliest available timestamp in nanoseconds.
        end_ns: Latest available timestamp in nanoseconds.
        duration_sec: Total buffered duration in seconds.
        is_seekable: Whether the buffer supports random access.
    """

    start_ns: int
    end_ns: int
    duration_sec: float
    is_seekable: bool


class BaseBuffer(ABC):
    """Abstract base class for video buffers."""

    @abstractmethod
    def query(self, start_ns: int, end_ns: int) -> bool:
        """Check if the requested time range is available.

        Args:
            start_ns: Start timestamp in nanoseconds.
            end_ns: End timestamp in nanoseconds.

        Returns:
            True if the range is fully available.
        """
        ...

    @abstractmethod
    def extract_clip(self, trigger: Trigger, output_dir: Path) -> ClipResult:
        """Extract a clip based on trigger.

        Args:
            trigger: Trigger defining clip boundaries.
            output_dir: Directory to save the clip.

        Returns:
            ClipResult with extraction status.
        """
        ...

    @property
    @abstractmethod
    def info(self) -> BufferInfo:
        """Get buffer state information."""
        ...


class FileBuffer(BaseBuffer):
    """Buffer backed by a video file.

    In file mode, the original file serves as the buffer.
    Seeking is used to access any part of the video.

    Args:
        source_path: Path to the video file.
        duration_ns: Total duration of the video in nanoseconds.
    """

    def __init__(self, source_path: Path, duration_ns: int):
        self._source_path = Path(source_path)
        self._duration_ns = duration_ns
        self._clipper: Optional[Clipper] = None

    def query(self, start_ns: int, end_ns: int) -> bool:
        """Check if time range is within file bounds."""
        if start_ns < 0:
            return False
        if end_ns > self._duration_ns:
            return False
        return True

    def extract_clip(self, trigger: Trigger, output_dir: Path) -> ClipResult:
        """Extract clip using ffmpeg."""
        # Create clipper if needed
        if self._clipper is None or self._clipper.output_dir != output_dir:
            self._clipper = Clipper(output_dir=output_dir)

        # Check bounds
        if not self.query(trigger.clip_start_ns, trigger.clip_end_ns):
            # Adjust trigger to fit within bounds
            adjusted_start = max(0, trigger.clip_start_ns)
            adjusted_end = min(self._duration_ns, trigger.clip_end_ns)

            if adjusted_start >= adjusted_end:
                return ClipResult(
                    success=False,
                    output_path=None,
                    trigger=trigger,
                    error="Requested time range is outside video bounds",
                )

        return self._clipper.extract(self._source_path, trigger)

    @property
    def info(self) -> BufferInfo:
        """File buffer info."""
        from visualbase.core.timestamp import ns_to_seconds

        return BufferInfo(
            start_ns=0,
            end_ns=self._duration_ns,
            duration_sec=ns_to_seconds(self._duration_ns),
            is_seekable=True,
        )

    @property
    def source_path(self) -> Path:
        """Path to the source file."""
        return self._source_path
