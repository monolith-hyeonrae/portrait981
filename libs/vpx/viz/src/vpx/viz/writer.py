"""VideoSaver - Write annotated frames to video file."""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from vpx.viz.overlay import Overlay, TextOverlay


class VideoSaver:
    """Save annotated frames to a video file.

    Args:
        path: Output file path (e.g., "output.mp4").
        fps: Output video FPS.
        width: Frame width.
        height: Frame height.
        overlays: Dict mapping module_name to Overlay instance.
                  If None, TextOverlay is used for all modules.
        codec: FourCC codec string (default "mp4v").
    """

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        overlays: Optional[Dict[str, Overlay]] = None,
        codec: str = "mp4v",
    ):
        self._path = path
        self._overlays = overlays
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Failed to open video writer: {path}")

    def update(self, frame: Any, observations: Dict[str, Any]) -> None:
        """Draw overlays and write frame to video.

        Args:
            frame: Frame object with ``.data`` attribute, or raw ndarray.
            observations: Dict mapping module_name -> Observation.
        """
        img = frame if isinstance(frame, np.ndarray) else frame.data
        display = img.copy()

        y_offset = 30
        for mod_name, obs in observations.items():
            if self._overlays and mod_name in self._overlays:
                overlay = self._overlays[mod_name]
            else:
                overlay = TextOverlay(y_offset=y_offset)

            display = overlay.draw(display, obs)

            if isinstance(overlay, TextOverlay):
                signals = getattr(obs, "signals", {})
                y_offset += overlay.line_height * (1 + len(signals))

        self._writer.write(display)

    def close(self) -> None:
        """Release the video writer."""
        self._writer.release()
