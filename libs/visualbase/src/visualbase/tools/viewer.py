"""Frame viewer for debugging and visualization."""

from typing import Optional, Callable

import cv2
import numpy as np

from visualbase.core.frame import Frame
from visualbase.core.timestamp import ns_to_seconds


class FrameViewer:
    """Simple frame viewer using OpenCV for debugging.

    Args:
        window_name: Name of the display window.
        show_info: Whether to overlay frame info (id, timestamp, size).

    Example:
        >>> viewer = FrameViewer()
        >>> with VisualBase() as vb:
        ...     vb.connect(FileSource("video.mp4"))
        ...     for frame in vb.get_stream(fps=10):
        ...         if not viewer.show(frame):
        ...             break  # User pressed 'q'
        >>> viewer.close()
    """

    def __init__(
        self,
        window_name: str = "VisualBase Viewer",
        show_info: bool = True,
    ):
        self._window_name = window_name
        self._show_info = show_info
        self._window_created = False

    def show(
        self,
        frame: Frame,
        wait_ms: int = 1,
        overlay_fn: Optional[Callable[[np.ndarray, Frame], np.ndarray]] = None,
    ) -> bool:
        """Display a frame.

        Args:
            frame: Frame to display.
            wait_ms: Milliseconds to wait for key press. 0 = wait indefinitely.
            overlay_fn: Optional function to draw custom overlays.
                        Signature: (image, frame) -> image

        Returns:
            True to continue, False if user wants to quit (pressed 'q' or ESC).
        """
        display = frame.data.copy()

        # Apply custom overlay if provided
        if overlay_fn is not None:
            display = overlay_fn(display, frame)

        # Draw frame info overlay
        if self._show_info:
            display = self._draw_info(display, frame)

        # Create window on first use
        if not self._window_created:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

        cv2.imshow(self._window_name, display)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            return False

        return True

    def _draw_info(self, image: np.ndarray, frame: Frame) -> np.ndarray:
        """Draw frame info overlay."""
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)  # Green
        bg_color = (0, 0, 0)  # Black background

        # Prepare info lines
        t_sec = ns_to_seconds(frame.t_src_ns)
        lines = [
            f"Frame: {frame.frame_id}",
            f"Time: {t_sec:.3f}s",
            f"Size: {frame.width}x{frame.height}",
        ]

        # Draw each line with background
        y_offset = 20
        for line in lines:
            (text_w, text_h), baseline = cv2.getTextSize(
                line, font, font_scale, thickness
            )
            # Draw background rectangle
            cv2.rectangle(
                image,
                (5, y_offset - text_h - 2),
                (10 + text_w, y_offset + baseline),
                bg_color,
                -1,
            )
            # Draw text
            cv2.putText(
                image,
                line,
                (7, y_offset),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y_offset += text_h + 8

        return image

    def close(self) -> None:
        """Close the viewer window."""
        if self._window_created:
            cv2.destroyWindow(self._window_name)
            self._window_created = False

    def __enter__(self) -> "FrameViewer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def play(
    source_path: str,
    fps: int = 0,
    resolution: Optional[tuple[int, int]] = None,
    show_info: bool = True,
) -> None:
    """Quick play function for testing.

    Args:
        source_path: Path to video file.
        fps: Target fps (0 for original).
        resolution: Target resolution (None for original).
        show_info: Show frame info overlay.

    Example:
        >>> from visualbase.tools.viewer import play
        >>> play("video.mp4", fps=10, resolution=(640, 480))
    """
    from visualbase import VisualBase, FileSource

    with FrameViewer(show_info=show_info) as viewer:
        with VisualBase() as vb:
            vb.connect(FileSource(source_path))
            print(f"Playing: {source_path}")
            print(f"Original: {vb.source.fps:.1f} fps, {vb.source.width}x{vb.source.height}")
            print(f"Target: fps={fps or 'original'}, resolution={resolution or 'original'}")
            print("Press 'q' or ESC to quit")

            for frame in vb.get_stream(fps=fps, resolution=resolution):
                if not viewer.show(frame):
                    break

    print(f"Finished at frame {frame.frame_id}")
