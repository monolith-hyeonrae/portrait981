"""Overlay protocol and built-in overlays for vpx-viz."""

from typing import Any, Dict, Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class Overlay(Protocol):
    """Protocol for drawing overlays on frames."""

    def draw(self, frame: np.ndarray, obs: Any) -> np.ndarray:
        """Draw overlay onto frame and return the modified frame.

        Args:
            frame: BGR image array (H, W, 3).
            obs: Observation from an analyzer.

        Returns:
            Modified frame (may be the same array).
        """
        ...


class TextOverlay:
    """Generic text overlay: renders module name + signals as text.

    Args:
        y_offset: Starting Y position for text.
        font_scale: OpenCV font scale.
        color: BGR color tuple.
        thickness: Text thickness.
    """

    def __init__(
        self,
        y_offset: int = 30,
        font_scale: float = 0.45,
        color: tuple = (255, 255, 255),
        thickness: int = 1,
    ):
        self._y_offset = y_offset
        self._font_scale = font_scale
        self._color = color
        self._thickness = thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    @property
    def line_height(self) -> int:
        """Approximate height per text line in pixels."""
        return int(20 * self._font_scale / 0.45)

    def draw(self, frame: np.ndarray, obs: Any) -> np.ndarray:
        """Draw observation signals as text on the frame.

        Args:
            frame: BGR image array.
            obs: Observation with ``source`` and ``signals`` attributes.

        Returns:
            Modified frame.
        """
        source = getattr(obs, "source", "?")
        signals = getattr(obs, "signals", {})

        y = self._y_offset

        # Module name header
        cv2.putText(
            frame,
            f"[{source}]",
            (10, y),
            self._font,
            self._font_scale,
            self._color,
            self._thickness,
        )
        y += self.line_height

        # Signal key-value pairs
        for key, value in signals.items():
            if isinstance(value, float):
                text = f"  {key}={value:.3f}"
            else:
                text = f"  {key}={value}"
            cv2.putText(
                frame,
                text,
                (10, y),
                self._font,
                self._font_scale,
                self._color,
                self._thickness,
            )
            y += self.line_height

        return frame


class MarkOverlay:
    """Overlay that renders annotate() marks + text signals.

    Combines TextOverlay (signal text) with mark rendering from
    a module's annotate() method.

    Args:
        module: Analyzer module with an ``annotate(obs)`` method.
        y_offset: Starting Y position for text signals.
    """

    def __init__(self, module: Any, y_offset: int = 30):
        self._module = module
        self._text = TextOverlay(y_offset=y_offset)

    @property
    def line_height(self) -> int:
        return self._text.line_height

    def draw(self, frame: np.ndarray, obs: Any) -> np.ndarray:
        """Draw marks from annotate() then text signals."""
        from vpx.viz.renderer import render_marks

        # Render declarative marks (bbox, keypoints, bars, labels)
        marks = []
        if hasattr(self._module, "annotate"):
            try:
                marks = self._module.annotate(obs)
            except Exception:
                pass

        if marks:
            frame = render_marks(frame, marks)

        # Also draw text signals
        frame = self._text.draw(frame, obs)
        return frame
