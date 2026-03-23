"""Signal field definitions, normalization, and extraction.

Extended 45D signal vector (all fields already computed, zero additional inference cost):
- AU layer (12D): all LibreFace DISFA outputs
- Emotion layer (8D): all HSEmotion outputs
- Pose layer (3D): yaw_dev, pitch, roll
- Detection (3D): confidence, face_area_ratio, face_center_distance
- Face quality (5D): blur, exposure, contrast, clipped_ratio, crushed_ratio
- Frame quality (3D): blur_score, brightness, contrast
- Segmentation (4D): seg_face, seg_eye, seg_mouth, seg_hair
- Composites (3D): duchenne_smile, wild_intensity, chill_score
- CLIP mood (4D, dynamic): warm_smile, cool_gaze, playful_face, wild_energy
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Field group definitions
# ---------------------------------------------------------------------------

# AU fields (LibreFace, DISFA 0-5 scale) -- 12 (full output)
_AU_FIELDS: tuple[str, ...] = (
    "au1_inner_brow", "au2_outer_brow", "au4_brow_lowerer", "au5_upper_lid",
    "au6_cheek_raiser", "au9_nose_wrinkler", "au12_lip_corner", "au15_lip_depressor",
    "au17_chin_raiser", "au20_lip_stretcher", "au25_lips_part", "au26_jaw_drop",
)

# Emotion fields (HSEmotion, probability 0-1) -- 8 (full output)
_EMOTION_FIELDS: tuple[str, ...] = (
    "em_happy", "em_neutral", "em_surprise", "em_angry",
    "em_contempt", "em_disgust", "em_fear", "em_sad",
)

# Head pose fields -- 3
_POSE_FIELDS: tuple[str, ...] = (
    "head_yaw_dev", "head_pitch", "head_roll",
)

# Detection fields -- 3
_DETECTION_FIELDS: tuple[str, ...] = (
    "face_confidence", "face_area_ratio", "face_center_distance",
)

# Face quality fields -- 5
_FACE_QUALITY_FIELDS: tuple[str, ...] = (
    "face_blur", "face_exposure", "face_contrast", "clipped_ratio", "crushed_ratio",
)

# Frame quality fields -- 3
_FRAME_QUALITY_FIELDS: tuple[str, ...] = (
    "blur_score", "brightness", "contrast",
)

# Segmentation ratio fields -- 4
_SEGMENTATION_FIELDS: tuple[str, ...] = (
    "seg_face", "seg_eye", "seg_mouth", "seg_hair",
)

# Derived segmentation fields -- 4
_DERIVED_SEG_FIELDS: tuple[str, ...] = (
    "eye_visible_ratio", "mouth_open_ratio", "glasses_ratio", "backlight_score",
)

# Composite fields -- 3
_COMPOSITE_FIELDS: tuple[str, ...] = (
    "duchenne_smile", "wild_intensity", "chill_score",
)

# Default CLIP axis names (match catalog categories)
_DEFAULT_CLIP_AXIS_NAMES: tuple[str, ...] = (
    "warm_smile", "cool_gaze", "playful_face", "wild_energy",
)

# ---------------------------------------------------------------------------
# Signal field sets
# ---------------------------------------------------------------------------

# Extended: all available fields (45D base + 4D derived + 4D CLIP = 53D)
SIGNAL_FIELDS_EXTENDED: tuple[str, ...] = (
    _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
    + _DETECTION_FIELDS + _FACE_QUALITY_FIELDS
    + _FRAME_QUALITY_FIELDS + _SEGMENTATION_FIELDS
    + _DERIVED_SEG_FIELDS
    + _COMPOSITE_FIELDS + _DEFAULT_CLIP_AXIS_NAMES
)

# Legacy 21D (backward compatibility with existing catalogs)
_AU_FIELDS_LEGACY: tuple[str, ...] = (
    "au1_inner_brow", "au2_outer_brow", "au4_brow_lowerer", "au5_upper_lid",
    "au6_cheek_raiser", "au9_nose_wrinkler", "au12_lip_corner", "au15_lip_depressor",
    "au25_lips_part", "au26_jaw_drop",
)
_EMOTION_FIELDS_LEGACY: tuple[str, ...] = (
    "em_happy", "em_neutral", "em_surprise", "em_angry",
)
SIGNAL_FIELDS_LEGACY: tuple[str, ...] = (
    _AU_FIELDS_LEGACY + _EMOTION_FIELDS_LEGACY + _POSE_FIELDS + _DEFAULT_CLIP_AXIS_NAMES
)

# Default: extended
SIGNAL_FIELDS: tuple[str, ...] = SIGNAL_FIELDS_EXTENDED

# ---------------------------------------------------------------------------
# Normalization ranges: value -> [0, 1]
# ---------------------------------------------------------------------------

SIGNAL_RANGES: dict[str, tuple[float, float]] = {
    # AU (0-5 DISFA scale)
    "au1_inner_brow": (0.0, 5.0),
    "au2_outer_brow": (0.0, 5.0),
    "au4_brow_lowerer": (0.0, 5.0),
    "au5_upper_lid": (0.0, 5.0),
    "au6_cheek_raiser": (0.0, 5.0),
    "au9_nose_wrinkler": (0.0, 5.0),
    "au12_lip_corner": (0.0, 5.0),
    "au15_lip_depressor": (0.0, 5.0),
    "au17_chin_raiser": (0.0, 5.0),
    "au20_lip_stretcher": (0.0, 5.0),
    "au25_lips_part": (0.0, 5.0),
    "au26_jaw_drop": (0.0, 5.0),
    # Emotion (probability 0-1)
    "em_happy": (0.0, 1.0),
    "em_neutral": (0.0, 1.0),
    "em_surprise": (0.0, 1.0),
    "em_angry": (0.0, 1.0),
    "em_contempt": (0.0, 1.0),
    "em_disgust": (0.0, 1.0),
    "em_fear": (0.0, 1.0),
    "em_sad": (0.0, 1.0),
    # Pose
    "head_yaw_dev": (0.0, 60.0),
    "head_pitch": (-30.0, 30.0),
    "head_roll": (-30.0, 30.0),
    # Detection
    "face_confidence": (0.0, 1.0),
    "face_area_ratio": (0.0, 1.0),
    "face_center_distance": (0.0, 1.0),
    # Face quality
    "face_blur": (0.0, 500.0),        # Laplacian variance
    "face_exposure": (0.0, 255.0),     # mean brightness
    "face_contrast": (0.0, 1.0),       # CV = std/mean
    "clipped_ratio": (0.0, 1.0),       # overexposed ratio
    "crushed_ratio": (0.0, 1.0),       # underexposed ratio
    # Frame quality
    "blur_score": (0.0, 500.0),        # Laplacian variance
    "brightness": (0.0, 255.0),
    "contrast": (0.0, 128.0),          # std of grayscale
    # Segmentation ratios (0-1)
    "seg_face": (0.0, 1.0),
    "seg_eye": (0.0, 1.0),
    "seg_mouth": (0.0, 1.0),
    "seg_hair": (0.0, 1.0),
    # Derived segmentation (0-1)
    "eye_visible_ratio": (0.0, 0.15),     # eye pixels / face area
    "mouth_open_ratio": (0.0, 0.15),      # mouth_in pixels / face area
    "glasses_ratio": (0.0, 0.5),          # glasses pixels / face area
    "backlight_score": (0.0, 100.0),      # brightness - face_exposure
    # Composites (0-1)
    "duchenne_smile": (0.0, 1.0),
    "wild_intensity": (0.0, 1.0),
    "chill_score": (0.0, 1.0),
}

# CLIP axes: default range is (0.0, 1.0), added dynamically
_DEFAULT_CLIP_RANGE: tuple[float, float] = (0.0, 1.0)

_NDIM = len(SIGNAL_FIELDS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_signal_fields(
    clip_axis_names: Optional[list[str]] = None,
    extended: bool = True,
) -> tuple[str, ...]:
    """Return signal field tuple.

    Args:
        clip_axis_names: CLIP axis name list. ``None`` uses the default 4 axes.
        extended: If True, return extended 45D+. If False, return legacy 21D.

    Returns:
        Signal field tuple.
    """
    axes = tuple(clip_axis_names) if clip_axis_names else _DEFAULT_CLIP_AXIS_NAMES
    if extended:
        return (
            _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
            + _DETECTION_FIELDS + _FACE_QUALITY_FIELDS
            + _FRAME_QUALITY_FIELDS + _SEGMENTATION_FIELDS
            + _DERIVED_SEG_FIELDS
            + _COMPOSITE_FIELDS + axes
        )
    return _AU_FIELDS_LEGACY + _EMOTION_FIELDS_LEGACY + _POSE_FIELDS + axes


def normalize_signal(value: float, field: str) -> float:
    """Normalize a single signal value to [0, 1]."""
    lo, hi = SIGNAL_RANGES.get(field, _DEFAULT_CLIP_RANGE)
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def extract_signal_vector_from_dict(
    signals: dict[str, float],
    signal_fields: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """Extract a normalized signal vector from a dict.

    Args:
        signals: signal name -> raw value dict.
        signal_fields: signal field order. ``None`` uses default :data:`SIGNAL_FIELDS`.

    Returns:
        ``(D,)`` normalized signal vector.
    """
    fields = signal_fields or SIGNAL_FIELDS
    ndim = len(fields)
    vec = np.zeros(ndim, dtype=np.float64)
    for i, f in enumerate(fields):
        raw = float(signals.get(f, 0.0))
        vec[i] = normalize_signal(raw, f)
    return vec
