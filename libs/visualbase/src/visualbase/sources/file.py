"""File-based video source using OpenCV."""

from pathlib import Path
from typing import Optional, Union

import cv2

from visualbase.core.frame import Frame
from visualbase.core.timestamp import pts_to_ns, ns_to_pts
from visualbase.sources.base import BaseSource
from visualbase.sources.decoder import configure_capture


class FileSource(BaseSource):
    """Video source from a local file (MP4, etc.).

    Args:
        path: Path to the video file.
        decoder: Video decoder to use ("auto", "nvdec", "vaapi", "cpu").
                 Default is "auto" which tries hardware acceleration first.
    """

    def __init__(self, path: Union[str, Path], decoder: str = "auto"):
        self._path = Path(path)
        self._decoder = decoder
        self._actual_decoder: str = "cpu"
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._duration_ns: int = 0

    def open(self) -> None:
        """Open the video file."""
        if not self._path.exists():
            raise IOError(f"Video file not found: {self._path}")

        self._cap, self._actual_decoder = configure_capture(
            str(self._path),
            decoder=self._decoder,
        )
        if not self._cap.isOpened():
            raise IOError(f"Failed to open video file: {self._path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_id = 0

        # Calculate duration in nanoseconds
        if self._fps > 0 and self._frame_count > 0:
            duration_sec = self._frame_count / self._fps
            self._duration_ns = int(duration_sec * 1_000_000_000)

    def read(self) -> Optional[Frame]:
        """Read the next frame from the video file."""
        if self._cap is None:
            raise RuntimeError("Source not opened. Call open() first.")

        ret, data = self._cap.read()
        if not ret:
            return None

        pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        t_src_ns = pts_to_ns(pts_ms)

        frame = Frame.from_array(
            data=data,
            frame_id=self._frame_id,
            t_src_ns=t_src_ns,
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        """Close the video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def seek(self, t_ns: int) -> bool:
        """Seek to a specific timestamp."""
        if self._cap is None:
            raise RuntimeError("Source not opened. Call open() first.")

        pts_ms = ns_to_pts(t_ns)
        return self._cap.set(cv2.CAP_PROP_POS_MSEC, pts_ms)

    @property
    def is_seekable(self) -> bool:
        """File sources are always seekable."""
        return True

    @property
    def fps(self) -> float:
        """Original frames per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Original frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Original frame height."""
        return self._height

    @property
    def path(self) -> Path:
        """Path to the video file."""
        return self._path

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        return self._frame_count

    @property
    def duration_ns(self) -> int:
        """Total duration in nanoseconds."""
        return self._duration_ns

    @property
    def duration_sec(self) -> float:
        """Total duration in seconds."""
        return self._duration_ns / 1_000_000_000

    @property
    def decoder(self) -> str:
        """Actual decoder being used (after open)."""
        return self._actual_decoder
