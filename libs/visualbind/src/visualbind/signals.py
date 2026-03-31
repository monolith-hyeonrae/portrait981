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

# Detection fields -- 4
_DETECTION_FIELDS: tuple[str, ...] = (
    "face_confidence", "face_area_ratio", "face_center_distance", "face_aspect_ratio",
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

# Derived segmentation fields -- 6
_DERIVED_SEG_FIELDS: tuple[str, ...] = (
    "eye_visible_ratio", "mouth_open_ratio", "glasses_ratio", "backlight_score",
    "nose_position_x", "nose_position_y",
)

# Lighting fields -- 15
# Lighting fields -- 19
_LIGHTING_FIELDS: tuple[str, ...] = (
    # 9분면 기반 (skin only)
    "lighting_ratio", "face_brightness_std", "highlight_ratio", "shadow_ratio",
    "light_direction_x", "light_direction_y", "rembrandt_score", "light_hardness",
    # DPR SH summary
    "sh_dir_x", "sh_dir_y", "sh_dir_strength",
    # DPR SH 9계수 raw (XGBoost가 패턴 자동 학습)
    "sh_0", "sh_1", "sh_2", "sh_3", "sh_4", "sh_5", "sh_6", "sh_7", "sh_8",
)

# ---------------------------------------------------------------------------
# Signal field sets
# ---------------------------------------------------------------------------

# Legacy compat (v1 catalog_scoring references these)
_DEFAULT_CLIP_AXIS_NAMES: tuple[str, ...] = ()
_DEFAULT_CLIP_RANGE: tuple[float, float] = (0.0, 1.0)
_COMPOSITE_FIELDS: tuple[str, ...] = ()

# 47D: all frozen model outputs + derived signals + lighting
SIGNAL_FIELDS_EXTENDED: tuple[str, ...] = (
    _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
    + _DETECTION_FIELDS + _FACE_QUALITY_FIELDS
    + _FRAME_QUALITY_FIELDS + _SEGMENTATION_FIELDS
    + _DERIVED_SEG_FIELDS + _LIGHTING_FIELDS
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
    _AU_FIELDS_LEGACY + _EMOTION_FIELDS_LEGACY + _POSE_FIELDS
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
    "head_yaw_dev": (0.0, 90.0),
    "head_pitch": (-30.0, 30.0),
    "head_roll": (-30.0, 30.0),
    # Detection
    "face_confidence": (0.0, 1.0),
    "face_area_ratio": (0.0, 1.0),
    "face_center_distance": (0.0, 1.0),
    "face_aspect_ratio": (0.0, 2.0),    # bbox w/h — frontal ~0.75, profile ~0.5
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
    "nose_position_x": (0.0, 1.0),        # 코 x 위치 (0=좌측, 0.5=정면, 1=우측)
    "nose_position_y": (0.0, 1.0),        # 코 y 위치 (0=상단, 0.5=중간, 1=하단)
    # ⚠️ Lighting signals 좌표계: 이미지 좌표 기준으로 통일.
    #   양수 = 이미지 오른쪽이 밝음 (x), 이미지 위쪽이 밝음 (y)
    #   DPR 출력은 이미지 좌표계 (변환 불필요).
    "lighting_ratio": (1.0, 3.0),         # bright_side / dark_side (1=uniform, >1.5=dramatic)
    "face_brightness_std": (0.0, 80.0),   # intra-face brightness variation (skin only)
    "highlight_ratio": (0.0, 0.5),        # fraction of skin pixels above mean+2σ
    "shadow_ratio": (0.0, 0.5),           # fraction of skin pixels below mean-1.5σ
    "light_direction_x": (-1.0, 1.0),     # 4분면 Michelson (양수=이미지 우측 밝음, skin only)
    "light_direction_y": (-1.0, 1.0),     # 4분면 Michelson (양수=이미지 상단 밝음, skin only)
    "rembrandt_score": (0.0, 1.0),        # Rembrandt triangle detected (skin only)
    "light_hardness": (0.0, 1.0),         # 0=soft(overcast), 1=hard(direct sun)
    # DPR SH (이미지 좌표계, 반전 불필요)
    #   sh_dir_x > 0 = 이미지 우측 밝음, < 0 = 좌측 밝음
    #   sh_dir_y > 0 = 이미지 상단 밝음
    "sh_dir_x": (-1.0, 1.0),
    "sh_dir_y": (-1.0, 1.0),
    "sh_dir_strength": (0.0, 1.0),
    # DPR SH 9계수 raw [amb, Y, Z, X, YX, YZ, 3Z²-1, XZ, X²-Y²]
    "sh_0": (0.0, 2.0),                   # ambient
    "sh_1": (-1.0, 1.0),                  # Y (depth)
    "sh_2": (-1.0, 1.0),                  # Z (상하)
    "sh_3": (-1.0, 1.0),                  # X (좌우)
    "sh_4": (-1.0, 1.0),                  # YX (2nd order)
    "sh_5": (-1.0, 1.0),                  # YZ
    "sh_6": (-1.0, 1.0),                  # 3Z²-1
    "sh_7": (-1.0, 1.0),                  # XZ
    "sh_8": (-1.0, 1.0),                  # X²-Y²
}
_DEFAULT_CLIP_RANGE: tuple[float, float] = (0.0, 1.0)

_NDIM = len(SIGNAL_FIELDS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_signal_fields(
    extended: bool = True,
) -> tuple[str, ...]:
    """Return signal field tuple.

    Args:
        extended: If True, return 43D. If False, return legacy 17D.

    Returns:
        Signal field tuple.
    """
    if extended:
        return SIGNAL_FIELDS_EXTENDED
    return SIGNAL_FIELDS_LEGACY


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
