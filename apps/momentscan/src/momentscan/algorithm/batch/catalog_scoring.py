"""Thin wrapper — real implementation in visualbind.

momentscan-specific functions (FrameRecord extraction/scoring) remain here.
Generic signal/profile/matching logic is in visualbind.

21D signal vector (momentscan catalog format):
- AU layer (10D): AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU25, AU26
- Emotion layer (4D): em_happy, em_neutral, em_surprise, em_angry
- Pose layer (3D): head_yaw_dev, head_pitch, head_roll
- Mood layer (4D): dynamic CLIP text axes (default: warm_smile, cool_gaze, playful_face, wild_energy)
"""

from __future__ import annotations

import logging
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from momentscan.algorithm.batch.types import FrameRecord

# -- Re-exports from visualbind (generic signal/profile/matching) --
from visualbind.signals import (
    SIGNAL_RANGES,
    normalize_signal,
    _AU_FIELDS,
    _EMOTION_FIELDS,
    _POSE_FIELDS,
    _DEFAULT_CLIP_AXIS_NAMES,
    _DEFAULT_CLIP_RANGE,
)
from visualbind.profile import (
    CategoryProfile,
    load_profiles,
)
from visualbind.profile import save_profiles as _vb_save_profiles
from visualbind.signals import extract_signal_vector_from_dict as _vb_extract_signal_vector_from_dict
from visualbind.strategies.catalog import (
    compute_importance_weights,
    match_category,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# momentscan 21D signal fields (no detect_confidence / face_size_ratio)
# Existing catalog _profile.json files are 21D with this field order.
# ---------------------------------------------------------------------------
SIGNAL_FIELDS: tuple[str, ...] = _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS + _DEFAULT_CLIP_AXIS_NAMES

_NDIM = len(SIGNAL_FIELDS)


def get_signal_fields(clip_axis_names: Optional[List[str]] = None) -> tuple[str, ...]:
    """AU + Emotion + Pose fixed fields + dynamic CLIP axis names (21D base).

    Args:
        clip_axis_names: CLIP axis name list.  None uses the default 4 axes.

    Returns:
        Signal field tuple.
    """
    axes = tuple(clip_axis_names) if clip_axis_names else _DEFAULT_CLIP_AXIS_NAMES
    return _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS + axes


# ---------------------------------------------------------------------------
# Wrappers that default to momentscan 21D fields
# ---------------------------------------------------------------------------

def extract_signal_vector_from_dict(
    signals: dict[str, float],
    signal_fields: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """Extract normalized signal vector from dict (defaults to 21D momentscan fields).

    Args:
        signals: signal name -> raw value dict.
        signal_fields: signal field order.  None uses momentscan 21D SIGNAL_FIELDS.

    Returns:
        (D,) normalized signal vector.
    """
    fields = signal_fields or SIGNAL_FIELDS
    return _vb_extract_signal_vector_from_dict(signals, signal_fields=fields)


def save_profiles(
    catalog_path: Path,
    profiles: List[CategoryProfile],
) -> None:
    """Save _profile.json for each category (with momentscan 21D signal_fields metadata).

    Args:
        catalog_path: catalog root directory.
        profiles: profiles to save.
    """
    _vb_save_profiles(catalog_path, profiles, signal_fields=SIGNAL_FIELDS)


# ---------------------------------------------------------------------------
# momentscan-specific: depends on FrameRecord
# ---------------------------------------------------------------------------

def extract_signal_vector(
    record: FrameRecord,
    signal_fields: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """FrameRecord에서 정규화된 시그널 벡터 추출.

    Args:
        record: FrameRecord.
        signal_fields: 시그널 필드 순서. None이면 기본 SIGNAL_FIELDS (21D).

    Returns:
        (D,) 정규화된 시그널 벡터.
    """
    fields = signal_fields or SIGNAL_FIELDS
    ndim = len(fields)
    vec = np.zeros(ndim, dtype=np.float64)
    for i, f in enumerate(fields):
        if f == "head_yaw_dev":
            raw = abs(float(getattr(record, "head_yaw", 0.0)))
        elif f in _AU_FIELDS or f in _EMOTION_FIELDS or f in _POSE_FIELDS:
            raw = float(getattr(record, f, 0.0))
        else:
            # CLIP axis -> clip_axes dict
            raw = float(record.clip_axes.get(f, 0.0))
        vec[i] = normalize_signal(raw, f)
    return vec


def compute_catalog_scores(
    record: FrameRecord,
    profiles: List[CategoryProfile],
) -> None:
    """FrameRecord에 catalog_best, catalog_primary, catalog_scores 설정.

    signal-profile 기반 매칭으로 모든 카테고리별 유사도를 계산하고
    best match를 catalog_best/catalog_primary에 저장한다.

    Args:
        record: 시그널이 채워진 FrameRecord (in-place 수정).
        profiles: 카테고리 프로파일 리스트.
    """
    if not profiles:
        return

    frame_vec = extract_signal_vector(record)
    scores: dict[str, float] = {}
    best_sim, best_name = -1.0, ""

    for profile in profiles:
        diff = frame_vec - profile.mean_signals
        weighted_sq = profile.importance_weights * (diff ** 2)
        d = sqrt(float(weighted_sq.sum()))
        sim = 1.0 / (1.0 + d)
        scores[profile.name] = sim
        if sim > best_sim:
            best_sim, best_name = sim, profile.name

    record.catalog_best = best_sim
    record.catalog_primary = best_name
    record.catalog_scores = scores


# ---------------------------------------------------------------------------
# load_clip_axes: momentscan-specific (depends on AxisDefinition)
# ---------------------------------------------------------------------------

def load_clip_axes(catalog_path: Path) -> list:
    """카탈로그에서 CLIP axis 정의 로딩.

    각 카테고리의 category.yaml에서 clip_axis 섹션을 읽어
    AxisDefinition 리스트로 변환.

    Args:
        catalog_path: 카탈로그 루트 디렉토리.

    Returns:
        AxisDefinition 리스트. clip_axis 미정의 카테고리는 건너뜀.
        하나도 없으면 빈 리스트.
    """
    from momentscan.portrait_score.backends.clip_portrait import AxisDefinition

    categories_dir = catalog_path / "categories"
    if not categories_dir.is_dir():
        return []

    axes = []
    for cat_dir in sorted(categories_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue

        yaml_path = cat_dir / "category.yaml"
        if not yaml_path.exists():
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        clip_axis = data.get("clip_axis")
        if not clip_axis:
            continue

        prompts = clip_axis.get("prompts", [])
        neg_prompts = clip_axis.get("neg_prompts", [])
        action = clip_axis.get("action", "select")
        threshold = float(data.get("threshold", 0.5))

        if not prompts:
            continue

        axes.append(AxisDefinition(
            name=data.get("name", cat_dir.name),
            prompts=tuple(prompts),
            neg_prompts=tuple(neg_prompts),
            action=action,
            threshold=threshold,
        ))

    if axes:
        logger.info("Loaded %d CLIP axes from %s", len(axes), catalog_path)

    return axes
