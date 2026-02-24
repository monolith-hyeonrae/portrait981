"""Output types for face.parse analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class FaceParseResult:
    """Per-face parsing result.

    Attributes:
        face_id: Matched face ID from face.detect.
        face_mask: Binary mask of face skin region (H, W) uint8 [0 or 255].
            Coordinates are relative to crop_box.
        crop_box: Face crop region in image pixel coords (x, y, w, h).
        class_map: Full 19-class segmentation map (H, W) uint8 [0..18].
            Coordinates are relative to crop_box.
    """

    face_id: int = 0
    face_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    crop_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    class_map: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )


@dataclass
class FaceParseOutput:
    """Output from FaceParseAnalyzer.

    Attributes:
        results: Per-face parsing results.
        face_bboxes: Normalized (x, y, w, h) bounding boxes for each face crop.
    """

    results: List[FaceParseResult] = field(default_factory=list)
    face_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


__all__ = ["FaceParseResult", "FaceParseOutput"]
