"""Output type for face AU analyzer."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FaceAUOutput:
    """Output from FaceAUAnalyzer.

    Attributes:
        au_intensities: Per-face AU intensity dictionaries.
            Each dict maps AU name (e.g., "AU12") to intensity (0-5 scale).
        au_presence: Per-face AU binary presence (intensity >= 1.0).
        face_bboxes: Per-face normalized bounding boxes (x, y, w, h) for overlay positioning.
    """

    au_intensities: List[Dict[str, float]] = field(default_factory=list)
    au_presence: List[Dict[str, bool]] = field(default_factory=list)
    face_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


__all__ = ["FaceAUOutput"]
