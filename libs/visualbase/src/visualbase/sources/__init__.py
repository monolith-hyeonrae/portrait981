from visualbase.sources.base import BaseSource
from visualbase.sources.file import FileSource
from visualbase.sources.image import ImageSource

__all__ = ["BaseSource", "FileSource", "ImageSource", "open_video"]


def open_video(path, *, fps=None):
    """Open a video file and return a (frames_iterator, cleanup_fn) tuple.

    Handles FPS-based frame skipping. Convenience for pipelines that need
    a frame iterator without managing source lifecycle directly.

    Args:
        path: Path to video file.
        fps: Target FPS for frame skipping. None = all frames.

    Returns:
        Tuple of (Iterator[Frame], Optional[cleanup_callable]).
    """
    source = FileSource(str(path))
    source.open()

    src_fps = source.fps or 30.0
    skip = max(1, int(src_fps / fps)) if fps and fps < src_fps else 1

    def _frames():
        read_count = 0
        try:
            while True:
                frame = source.read()
                if frame is None:
                    break
                read_count += 1
                if read_count % skip != 0:
                    continue
                yield frame
        finally:
            source.close()

    return _frames(), source.close
