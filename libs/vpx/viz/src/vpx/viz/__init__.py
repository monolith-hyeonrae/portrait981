"""vpx-viz - Visualization overlays and display for vpx modules.

Example:
    >>> from vpx.viz import TextOverlay, FrameDisplay
    >>> display = FrameDisplay(title="vpx")
    >>> overlay = TextOverlay()
"""

from vpx.viz.overlay import Overlay, TextOverlay, MarkOverlay
from vpx.viz.display import FrameDisplay
from vpx.viz.writer import VideoSaver
from vpx.viz.renderer import render_marks

__all__ = ["Overlay", "TextOverlay", "MarkOverlay", "FrameDisplay", "VideoSaver", "render_marks"]
