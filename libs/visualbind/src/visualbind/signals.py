"""Signal field definitions, normalization, and extraction.

23D base signal vector:
- AU layer (10D): AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU25, AU26
- Emotion layer (4D): em_happy, em_neutral, em_surprise, em_angry
- Pose layer (3D): head_yaw_dev, head_pitch, head_roll
- Confidence (1D): detect_confidence
- Face size (1D): face_size_ratio
- Mood layer (4D, dynamic): CLIP text axes (default: warm_smile, cool_gaze, playful_face, wild_energy)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Field group definitions
# ---------------------------------------------------------------------------

# AU fields (LibreFace, DISFA 0-5 scale) -- 10
_AU_FIELDS: tuple[str, ...] = (
    "au1_inner_brow", "au2_outer_brow", "au4_brow_lowerer", "au5_upper_lid",
    "au6_cheek_raiser", "au9_nose_wrinkler", "au12_lip_corner", "au15_lip_depressor",
    "au25_lips_part", "au26_jaw_drop",
)

# Emotion fields (HSEmotion, probability 0-1) -- 4
_EMOTION_FIELDS: tuple[str, ...] = (
    "em_happy", "em_neutral", "em_surprise", "em_angry",
)

# Head pose fields (normalized, Fisher ratio for auto-importance) -- 3
# head_yaw_dev: |yaw| frontal deviation (L/R symmetric -> abs to prevent cancellation)
_POSE_FIELDS: tuple[str, ...] = (
    "head_yaw_dev", "head_pitch", "head_roll",
)

# Confidence field -- 1
_CONFIDENCE_FIELD: tuple[str, ...] = ("detect_confidence",)

# Face size field -- 1
_FACE_SIZE_FIELD: tuple[str, ...] = ("face_size_ratio",)

# Default CLIP axis names (match catalog categories)
_DEFAULT_CLIP_AXIS_NAMES: tuple[str, ...] = (
    "warm_smile", "cool_gaze", "playful_face", "wild_energy",
)

# ---------------------------------------------------------------------------
# Composite signal vector: 23D base (AU 10 + Emotion 4 + Pose 3 + Confidence 1 + FaceSize 1 + CLIP 4)
# ---------------------------------------------------------------------------

SIGNAL_FIELDS: tuple[str, ...] = (
    _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
    + _CONFIDENCE_FIELD + _FACE_SIZE_FIELD
    + _DEFAULT_CLIP_AXIS_NAMES
)

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
    "au25_lips_part": (0.0, 5.0),
    "au26_jaw_drop": (0.0, 5.0),
    # Emotion (probability)
    "em_happy": (0.0, 1.0),
    "em_neutral": (0.0, 1.0),
    "em_surprise": (0.0, 1.0),
    "em_angry": (0.0, 1.0),
    # Pose
    "head_yaw_dev": (0.0, 60.0),
    "head_pitch": (-30.0, 30.0),
    "head_roll": (-30.0, 30.0),
    # Confidence + Face size (both 0-1)
    "detect_confidence": (0.0, 1.0),
    "face_size_ratio": (0.0, 1.0),
}

# CLIP axes: default range is (0.0, 1.0), added dynamically
_DEFAULT_CLIP_RANGE: tuple[float, float] = (0.0, 1.0)

_NDIM = len(SIGNAL_FIELDS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_signal_fields(clip_axis_names: Optional[list[str]] = None) -> tuple[str, ...]:
    """Return AU + Emotion + Pose + Confidence + FaceSize fixed fields + dynamic CLIP axis names.

    Args:
        clip_axis_names: CLIP axis name list.  ``None`` uses the default 4 axes.

    Returns:
        Signal field tuple.
    """
    axes = tuple(clip_axis_names) if clip_axis_names else _DEFAULT_CLIP_AXIS_NAMES
    return _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS + _CONFIDENCE_FIELD + _FACE_SIZE_FIELD + axes


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
        signal_fields: signal field order.  ``None`` uses default :data:`SIGNAL_FIELDS`.

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
