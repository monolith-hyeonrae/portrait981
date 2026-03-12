"""Output types for portrait score analyzer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PortraitScoreOutput:
    """Output from PortraitScoreAnalyzer.

    Attributes:
        portrait_crop_box: Portrait crop bounding box (x, y, w, h) in pixels or None.
        image_size: (width, height) of the source frame in pixels.
        head_aesthetic: CLIP portrait quality score, normalized [0, 1].
            Defaults to 0.0 when open-clip is unavailable.
    """

    portrait_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None  # (w, h)
    head_aesthetic: float = 0.0


__all__ = ["PortraitScoreOutput"]
