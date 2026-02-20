"""Output types for shot quality analyzer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ShotQualityOutput:
    """Output from ShotQualityAnalyzer.

    Attributes:
        head_crop_box: Head crop bounding box (x, y, w, h) in pixels or None.
        image_size: (width, height) of the source frame in pixels.

        Layer 1 — CV-based (always computed when face detected):
            head_blur: Laplacian variance of head_crop (sharpness).
            head_exposure: Mean brightness of head_crop (0-255).

        Layer 2 — LAION aesthetic (optional, requires open-clip-torch + weights):
            head_aesthetic: LAION aesthetic score for head_crop, normalized [0, 1].
            Defaults to 0.0 when weights are unavailable.
    """

    head_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None  # (w, h)

    # Layer 1: CV-based quality
    head_blur: float = 0.0
    head_exposure: float = 0.0

    # Layer 2: LAION aesthetic (optional)
    head_aesthetic: float = 0.0


__all__ = ["ShotQualityOutput"]
