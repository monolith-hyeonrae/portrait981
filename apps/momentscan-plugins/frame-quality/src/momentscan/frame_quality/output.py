"""Quality analyzer output dataclass."""

from dataclasses import dataclass


@dataclass
class QualityOutput:
    """Output from QualityAnalyzer.

    Attributes:
        blur_score: Laplacian variance (higher = sharper).
        brightness: Mean brightness [0-255].
        contrast: Standard deviation of brightness.
    """
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
