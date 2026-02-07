"""FIFO-based video streaming for A→B* proxy streams.

Uses named pipes (FIFOs) to distribute video frames from the Ingest process (A)
to multiple Extractor processes (B*). Each extractor gets its own FIFO with
potentially different resolution/fps.

MJPEG format is used for simplicity:
- Each frame is a JPEG with length prefix
- Format: [4-byte length][8-byte frame_id][8-byte t_ns][JPEG data]
"""

import os
import stat
import struct
import threading
from pathlib import Path
from typing import Optional
import logging

import cv2
import numpy as np

from visualbase.core.frame import Frame
from visualbase.ipc.interfaces import VideoReader, VideoWriter

logger = logging.getLogger(__name__)

# Header format: uint32 length + uint64 frame_id + uint64 t_ns = 20 bytes
HEADER_FORMAT = "<IQQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class FIFOVideoWriter(VideoWriter):
    """Write video frames to a FIFO (named pipe).

    Creates a FIFO at the specified path and writes MJPEG-encoded frames.
    The writer blocks until a reader connects.

    Implements the VideoWriter interface for swappable transport.

    Args:
        path: Path to create the FIFO.
        jpeg_quality: JPEG encoding quality (0-100).

    Example:
        >>> writer = FIFOVideoWriter("/tmp/vid_face.mjpg")
        >>> writer.open()  # blocks until reader connects
        >>> writer.write(frame)
        >>> writer.close()
    """

    def __init__(self, path: str, jpeg_quality: int = 85):
        self._path = Path(path)
        self._jpeg_quality = jpeg_quality
        self._fd: Optional[int] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """Create and open the FIFO for writing.

        This call blocks until a reader opens the other end.
        """
        # Remove existing file if present
        if self._path.exists():
            self._path.unlink()

        # Create FIFO
        os.mkfifo(str(self._path))
        logger.info(f"Created FIFO: {self._path}")

        # Open for writing (blocks until reader connects)
        self._fd = os.open(str(self._path), os.O_WRONLY)
        logger.info(f"FIFO opened for writing: {self._path}")

    def write(self, frame: Frame) -> bool:
        """Write a frame to the FIFO.

        Args:
            frame: Frame to write.

        Returns:
            True if write succeeded, False on error.
        """
        if self._fd is None:
            return False

        try:
            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            success, jpeg_data = cv2.imencode(".jpg", frame.data, encode_params)
            if not success:
                logger.warning(f"Failed to encode frame {frame.frame_id}")
                return False

            jpeg_bytes = jpeg_data.tobytes()

            # Build header
            header = struct.pack(
                HEADER_FORMAT,
                len(jpeg_bytes),
                frame.frame_id,
                frame.t_src_ns,
            )

            # Write atomically
            with self._lock:
                os.write(self._fd, header + jpeg_bytes)

            return True

        except BrokenPipeError:
            logger.warning("FIFO reader disconnected")
            return False
        except Exception as e:
            logger.error(f"FIFO write error: {e}")
            return False

    def close(self) -> None:
        """Close the FIFO."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

        # Remove the FIFO file
        if self._path.exists():
            try:
                self._path.unlink()
            except Exception:
                pass

    def __enter__(self) -> "FIFOVideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        """Check if the FIFO is open."""
        return self._fd is not None


class FIFOVideoReader(VideoReader):
    """Read video frames from a FIFO (named pipe).

    Opens an existing FIFO and reads MJPEG-encoded frames.

    Implements the VideoReader interface for swappable transport.

    Args:
        path: Path to the existing FIFO.

    Example:
        >>> reader = FIFOVideoReader("/tmp/vid_face.mjpg")
        >>> reader.open()
        >>> frame = reader.read()
        >>> if frame:
        ...     process(frame)
        >>> reader.close()
    """

    def __init__(self, path: str):
        self._path = Path(path)
        self._fd: Optional[int] = None
        self._lock = threading.Lock()

    def open(self, timeout_sec: Optional[float] = None) -> bool:
        """Open the FIFO for reading.

        Args:
            timeout_sec: Optional timeout. If None, blocks indefinitely.

        Returns:
            True if opened successfully.
        """
        if not self._path.exists():
            logger.error(f"FIFO does not exist: {self._path}")
            return False

        # Check if it's actually a FIFO
        if not stat.S_ISFIFO(os.stat(str(self._path)).st_mode):
            logger.error(f"Not a FIFO: {self._path}")
            return False

        # Open for reading (this unblocks the writer)
        try:
            self._fd = os.open(str(self._path), os.O_RDONLY)
            logger.info(f"FIFO opened for reading: {self._path}")
            return True
        except Exception as e:
            logger.error(f"Failed to open FIFO: {e}")
            return False

    def read(self) -> Optional[Frame]:
        """Read a frame from the FIFO.

        Returns:
            Frame if read succeeded, None on EOF or error.
        """
        if self._fd is None:
            return None

        try:
            with self._lock:
                # Read header
                header_data = self._read_exact(HEADER_SIZE)
                if header_data is None:
                    return None

                jpeg_len, frame_id, t_src_ns = struct.unpack(
                    HEADER_FORMAT, header_data
                )

                # Read JPEG data
                jpeg_data = self._read_exact(jpeg_len)
                if jpeg_data is None:
                    return None

            # Decode JPEG
            jpeg_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
            if image is None:
                logger.warning(f"Failed to decode frame {frame_id}")
                return None

            return Frame.from_array(
                data=image,
                frame_id=frame_id,
                t_src_ns=t_src_ns,
            )

        except Exception as e:
            logger.error(f"FIFO read error: {e}")
            return None

    def _read_exact(self, size: int) -> Optional[bytes]:
        """Read exactly `size` bytes from the FIFO."""
        data = b""
        remaining = size
        while remaining > 0:
            chunk = os.read(self._fd, remaining)
            if not chunk:
                return None  # EOF
            data += chunk
            remaining -= len(chunk)
        return data

    def close(self) -> None:
        """Close the FIFO."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def __enter__(self) -> "FIFOVideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        """Check if the FIFO is open."""
        return self._fd is not None

    def __iter__(self):
        """Iterate over frames."""
        return self

    def __next__(self) -> Frame:
        """Get next frame."""
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
