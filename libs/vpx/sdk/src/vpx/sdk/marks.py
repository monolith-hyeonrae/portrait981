"""Declarative visualization mark types.

Marks describe *what* to draw, not *how*. A renderer (e.g. vpx-viz)
interprets marks and produces actual pixels.

All Mark types are frozen dataclasses — immutable, serializable, and
comparable with ``==`` for easy testing.

Example:
    >>> from vpx.sdk.marks import BBoxMark, LabelMark
    >>> marks = [
    ...     BBoxMark(x=0.3, y=0.2, w=0.2, h=0.3, label="ID:1", color=(0, 255, 0)),
    ...     LabelMark(text="main", x=0.3, y=0.18, color=(0, 255, 0)),
    ... ]
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DrawStyle:
    """App-level style overrides for module visualization.

    Passed to the renderer to customize colors, thickness, etc.
    without modifying the module's annotate() output.
    """

    color: tuple[int, int, int] | None = None
    thickness: int | None = None
    font_scale: float | None = None
    show_labels: bool = True
    show_confidence: bool = True
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class BBoxMark:
    """Bounding box. Coordinates are normalized [0, 1]."""

    x: float
    y: float
    w: float
    h: float
    label: str = ""
    color: tuple[int, int, int] | None = None
    thickness: int = 2
    confidence: float = 1.0


@dataclass(frozen=True)
class KeypointsMark:
    """Keypoints with optional connection lines.

    Each point is ``(x, y, confidence)``.
    """

    points: tuple[tuple[float, float, float], ...]
    connections: tuple[tuple[int, int], ...] = ()
    normalized: bool = False
    line_color: tuple[int, int, int] = (255, 200, 100)
    point_colors: dict[int, tuple[int, int, int]] | None = None
    point_radius: int = 3
    min_confidence: float = 0.3


@dataclass(frozen=True)
class BarMark:
    """Progress bar. Position is normalized [0, 1]."""

    x: float
    y: float
    w: float
    value: float  # [0, 1] fill ratio
    color: tuple[int, int, int] = (0, 255, 255)
    height_px: int = 6
    label: str = ""


@dataclass(frozen=True)
class LabelMark:
    """Text label."""

    text: str
    x: float
    y: float
    normalized: bool = True
    color: tuple[int, int, int] = (255, 255, 255)
    background: tuple[int, int, int] | None = None
    font_scale: float = 0.45


@dataclass(frozen=True)
class AxisMark:
    """3D pose axis visualization (6DRepNet-style).

    Draws X(red)/Y(green)/Z(blue) axes from a center point
    to visualize yaw/pitch/roll head orientation.
    All coordinates normalized [0, 1]. Angles in degrees.
    """

    cx: float
    cy: float
    yaw: float
    pitch: float
    roll: float
    size: float = 0.06
    thickness: int = 2


Mark = BBoxMark | KeypointsMark | BarMark | LabelMark | AxisMark

__all__ = [
    "DrawStyle",
    "Mark",
    "BBoxMark",
    "KeypointsMark",
    "BarMark",
    "LabelMark",
    "AxisMark",
]
