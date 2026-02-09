"""FrameDisplay - Live cv2 window for vpx-viz."""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from vpx.viz.overlay import Overlay, TextOverlay


class FrameDisplay:
    """Live display window using cv2.imshow. ESC to quit.

    Args:
        title: Window title.
        overlays: Dict mapping module_name to Overlay instance.
                  If None, TextOverlay is used for all modules.
        wait_ms: cv2.waitKey delay in milliseconds.
    """

    def __init__(
        self,
        title: str = "vpx",
        overlays: Optional[Dict[str, Overlay]] = None,
        wait_ms: int = 1,
    ):
        self._title = title
        self._overlays = overlays
        self._wait_ms = wait_ms
        self._default_overlay_offset = 30

    def update(self, frame: Any, observations: Dict[str, Any]) -> bool:
        """Draw overlays and display the frame.

        Args:
            frame: Frame object with ``.data`` attribute, or raw ndarray.
            observations: Dict mapping module_name -> Observation.

        Returns:
            True to continue, False if user pressed ESC.
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

            # Stack y_offset for next TextOverlay
            if isinstance(overlay, TextOverlay):
                signals = getattr(obs, "signals", {})
                y_offset += overlay.line_height * (1 + len(signals))

        cv2.imshow(self._title, display)
        key = cv2.waitKey(self._wait_ms) & 0xFF
        return key != 27  # ESC

    def close(self) -> None:
        """Close the display window."""
        cv2.destroyAllWindows()
