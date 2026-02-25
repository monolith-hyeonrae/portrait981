"""Output type for face quality analyzer."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FaceQualityResult:
    """Per-face quality measurement."""

    face_id: int = 0
    head_blur: float = 0.0
    head_exposure: float = 0.0
    head_crop_box: tuple = (0, 0, 0, 0)
    # Mask-based metrics
    mask_method: str = ""           # "parsing" | "landmark" | "center_patch"
    head_contrast: float = 0.0     # CV = std / mean (skin-tone invariant)
    clipped_ratio: float = 0.0     # overexposed (>250) pixel ratio
    crushed_ratio: float = 0.0     # underexposed (<5) pixel ratio
    parsing_coverage: float = 0.0  # face.parse mask pixels / crop pixels (0 = 미측정/미시도)
    # Semantic segmentation ratios (from class_map, info-only)
    seg_face: float = 0.0        # face region (skin+brow+eye+ear+nose+mouth+lip)
    seg_eye: float = 0.0         # classes 4+5
    seg_mouth: float = 0.0       # classes 11+12+13
    seg_hair: float = 0.0        # class 17
    eye_pixel_ratio: float = 0.0  # = seg_eye (eye_open 교차검증용)


@dataclass
class FaceQualityOutput:
    """Output from FaceQualityAnalyzer.

    Attributes:
        face_results: Per-face quality results for all detected faces.
        head_crop_box: Head crop bounding box (x, y, w, h) in pixels or None (main face).
        image_size: (width, height) of the source frame in pixels.
        head_blur: Laplacian variance of head_crop (sharpness) — main face.
        head_exposure: Mean brightness of head_crop (0-255) — main face.
        mask_method: Mask method used for main face.
        head_contrast: Local contrast (CV) for main face.
        clipped_ratio: Overexposed pixel ratio for main face.
        crushed_ratio: Underexposed pixel ratio for main face.
        parsing_coverage: face.parse mask pixels / crop pixels for main face (0 = 미측정).
    """

    face_results: List[FaceQualityResult] = field(default_factory=list)
    head_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None  # (w, h)
    head_blur: float = 0.0
    head_exposure: float = 0.0
    mask_method: str = ""
    head_contrast: float = 0.0
    clipped_ratio: float = 0.0
    crushed_ratio: float = 0.0
    parsing_coverage: float = 0.0
    seg_face: float = 0.0
    seg_eye: float = 0.0
    seg_mouth: float = 0.0
    seg_hair: float = 0.0
    eye_pixel_ratio: float = 0.0


__all__ = ["FaceQualityResult", "FaceQualityOutput"]
