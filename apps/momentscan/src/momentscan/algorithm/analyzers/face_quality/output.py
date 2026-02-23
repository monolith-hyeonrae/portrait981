"""Output type for face quality analyzer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FaceQualityOutput:
    """Output from FaceQualityAnalyzer.

    Attributes:
        head_crop_box: Head crop bounding box (x, y, w, h) in pixels or None.
        image_size: (width, height) of the source frame in pixels.
        head_blur: Laplacian variance of head_crop (sharpness).
        head_exposure: Mean brightness of head_crop (0-255).
    """

    head_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None  # (w, h)
    head_blur: float = 0.0
    head_exposure: float = 0.0


__all__ = ["FaceQualityOutput"]
