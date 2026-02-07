"""VisualBase main API."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union

from visualbase.core.buffer import BaseBuffer, FileBuffer, BufferInfo
from visualbase.core.frame import Frame
from visualbase.core.ring_buffer import RingBuffer
from visualbase.core.sampler import Sampler
from visualbase.packaging.clipper import ClipResult
from visualbase.packaging.trigger import Trigger
from visualbase.sources.base import BaseSource
from visualbase.sources.file import FileSource


class VisualBase:
    """Main interface for video frame streaming and clip extraction.

    Example:
        >>> vb = VisualBase(clip_output_dir=Path("/data/clips"))
        >>> vb.connect(FileSource("video.mp4"))
        >>>
        >>> # Stream frames for analysis
        >>> for frame in vb.get_stream(fps=10, resolution=(640, 480)):
        ...     process(frame.data)
        >>>
        >>> # Extract clip on trigger
        >>> trigger = Trigger.point(event_time_ns=5_000_000_000, pre_sec=2.0, post_sec=2.0)
        >>> result = vb.trigger(trigger)
        >>> if result.success:
        ...     print(f"Clip saved: {result.output_path}")
        >>>
        >>> vb.disconnect()
    """

    def __init__(self, clip_output_dir: Optional[Union[str, Path]] = None):
        """Initialize VisualBase.

        Args:
            clip_output_dir: Directory for extracted clips. Defaults to "./clips".
        """
        self._source: Optional[BaseSource] = None
        self._buffer: Optional[BaseBuffer] = None
        self._clip_output_dir = Path(clip_output_dir) if clip_output_dir else Path("./clips")

    def connect(
        self,
        source: BaseSource,
        ring_buffer_retention_sec: float = 120.0,
    ) -> None:
        """Connect to a video source.

        Automatically selects the appropriate buffer type:
        - FileBuffer for seekable sources (FileSource)
        - RingBuffer for streaming sources (CameraSource, RTSPSource)

        Args:
            source: Video source to connect to.
            ring_buffer_retention_sec: Retention time for RingBuffer in seconds.
                Only used for non-seekable sources. Default: 120 seconds.

        Raises:
            IOError: If the source cannot be opened.
        """
        if self._source is not None:
            self.disconnect()

        source.open()
        self._source = source

        # Initialize buffer based on source seekability
        if source.is_seekable:
            # Seekable source (FileSource) - use FileBuffer
            if isinstance(source, FileSource):
                self._buffer = FileBuffer(
                    source_path=source.path,
                    duration_ns=source.duration_ns,
                )
            else:
                # Other seekable sources - buffer will need to be set up manually
                self._buffer = None
        else:
            # Non-seekable source (CameraSource, RTSPSource) - use RingBuffer
            self._buffer = RingBuffer(
                retention_sec=ring_buffer_retention_sec,
                fps=source.fps,
            )

    def disconnect(self) -> None:
        """Disconnect from the current video source."""
        if self._buffer is not None:
            # Close RingBuffer if it's a streaming buffer
            if isinstance(self._buffer, RingBuffer):
                self._buffer.close()
            self._buffer = None

        if self._source is not None:
            self._source.close()
            self._source = None

    def get_stream(
        self,
        fps: int = 0,
        resolution: Optional[tuple[int, int]] = None,
        buffer_frames: bool = True,
    ) -> Iterator[Frame]:
        """Get a frame stream from the connected source.

        For streaming sources (camera, RTSP), this also buffers frames
        automatically so they're available for clip extraction.

        Args:
            fps: Target frames per second. 0 for original fps.
            resolution: Target (width, height). None for original resolution.
            buffer_frames: Whether to add frames to the buffer (default: True).
                Only applies to RingBuffer (streaming sources).

        Returns:
            Iterator yielding Frame objects.

        Raises:
            RuntimeError: If no source is connected.
        """
        if self._source is None:
            raise RuntimeError("No source connected. Call connect() first.")

        sampler = Sampler(self._source, fps=fps, resolution=resolution)

        # Wrap with buffering for streaming sources
        if buffer_frames and isinstance(self._buffer, RingBuffer):
            return self._buffered_stream(sampler)
        else:
            return sampler

    def _buffered_stream(self, sampler: Sampler) -> Iterator[Frame]:
        """Wrap a sampler to buffer frames for RingBuffer."""
        for frame in sampler:
            if isinstance(self._buffer, RingBuffer):
                self._buffer.add_frame(frame)
            yield frame

    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the buffer.

        Only applicable for streaming sources with RingBuffer.
        For FileSource, this is a no-op since the file serves as the buffer.

        Args:
            frame: Frame to add to the buffer.
        """
        if isinstance(self._buffer, RingBuffer):
            self._buffer.add_frame(frame)

    @property
    def source(self) -> Optional[BaseSource]:
        """Currently connected source."""
        return self._source

    @property
    def buffer(self) -> Optional[BaseBuffer]:
        """Currently active buffer."""
        return self._buffer

    @property
    def is_connected(self) -> bool:
        """Whether a source is currently connected."""
        return self._source is not None

    def query_buffer(self, start_ns: int, end_ns: int) -> bool:
        """Check if a time range is available in the buffer.

        Args:
            start_ns: Start timestamp in nanoseconds.
            end_ns: End timestamp in nanoseconds.

        Returns:
            True if the range is available.

        Raises:
            RuntimeError: If no source is connected or buffer unavailable.
        """
        if self._buffer is None:
            raise RuntimeError("No buffer available. Connect to a source first.")
        return self._buffer.query(start_ns, end_ns)

    def get_buffer_info(self) -> BufferInfo:
        """Get buffer state information.

        Returns:
            BufferInfo with current buffer state.

        Raises:
            RuntimeError: If no buffer is available.
        """
        if self._buffer is None:
            raise RuntimeError("No buffer available. Connect to a source first.")
        return self._buffer.info

    def trigger(
        self,
        trig: Trigger,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ClipResult:
        """Extract a clip based on trigger.

        Args:
            trig: Trigger defining clip boundaries.
            output_dir: Override output directory for this clip.

        Returns:
            ClipResult with extraction status and output path.

        Raises:
            RuntimeError: If no source is connected or buffer unavailable.
        """
        if self._buffer is None:
            raise RuntimeError("No buffer available. Connect to a source first.")

        out_dir = Path(output_dir) if output_dir else self._clip_output_dir
        return self._buffer.extract_clip(trig, out_dir)

    @property
    def clip_output_dir(self) -> Path:
        """Default output directory for clips."""
        return self._clip_output_dir

    @clip_output_dir.setter
    def clip_output_dir(self, value: Union[str, Path]) -> None:
        """Set default output directory for clips."""
        self._clip_output_dir = Path(value)

    def __enter__(self) -> "VisualBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
