"""Signal-Profile Catalog: 참조 이미지 기반 멀티시그널 매칭 엔진.

참조 이미지를 동일 파이프라인으로 분석하여 시그널 프로파일(centroid) 생성.
프레임과 프로파일 간 시그널 공간 가중 유클리디안 거리로 매칭.

21차원 시그널 벡터:
- AU 레이어 (10D): AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU25, AU26 (LibreFace, DISFA 0-5)
- 감정 레이어 (4D): em_happy, em_neutral, em_surprise, em_angry (HSEmotion, 확률 0-1)
- 포즈 레이어 (3D): head_yaw, head_pitch, head_roll (정규화, Fisher ratio로 자동 중요도 결정)
- 분위기 레이어 (4D): 동적 CLIP text axes (기본: warm_smile, cool_gaze, playful_face, wild_energy)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from momentscan.algorithm.batch.types import FrameRecord

logger = logging.getLogger(__name__)

# AU 필드 (LibreFace, DISFA 0-5 스케일) — 10개
_AU_FIELDS: tuple[str, ...] = (
    "au1_inner_brow", "au2_outer_brow", "au4_brow_lowerer", "au5_upper_lid",
    "au6_cheek_raiser", "au9_nose_wrinkler", "au12_lip_corner", "au15_lip_depressor",
    "au25_lips_part", "au26_jaw_drop",
)

# Emotion 필드 (HSEmotion, 확률 0-1) — 4개
_EMOTION_FIELDS: tuple[str, ...] = (
    "em_happy", "em_neutral", "em_surprise", "em_angry",
)

# Head pose 필드 (정규화, Fisher ratio로 자동 중요도) — 3개
# head_yaw_dev: |yaw| 정면 이탈도 (좌/우 대칭 → abs로 상쇄 방지)
_POSE_FIELDS: tuple[str, ...] = (
    "head_yaw_dev", "head_pitch", "head_roll",
)

# 기본 CLIP axis 이름 (카탈로그 카테고리와 일치)
_DEFAULT_CLIP_AXIS_NAMES: tuple[str, ...] = (
    "warm_smile", "cool_gaze", "playful_face", "wild_energy",
)

# 시그널 벡터 필드 (AU + Emotion + Pose + CLIP) — 21D
SIGNAL_FIELDS: tuple[str, ...] = _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS + _DEFAULT_CLIP_AXIS_NAMES


def get_signal_fields(clip_axis_names: Optional[List[str]] = None) -> tuple[str, ...]:
    """AU + Emotion + Pose 고정 필드 + 동적 CLIP axis 이름.

    Args:
        clip_axis_names: CLIP axis 이름 리스트. None이면 기본 4축.

    Returns:
        시그널 필드 튜플.
    """
    axes = tuple(clip_axis_names) if clip_axis_names else _DEFAULT_CLIP_AXIS_NAMES
    return _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS + axes


# 각 시그널의 정규화 범위 (distance 계산 전 [0,1] 정규화)
SIGNAL_RANGES: dict[str, tuple[float, float]] = {
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
    "em_happy": (0.0, 1.0),
    "em_neutral": (0.0, 1.0),
    "em_surprise": (0.0, 1.0),
    "em_angry": (0.0, 1.0),
    "head_yaw_dev": (0.0, 60.0),
    "head_pitch": (-30.0, 30.0),
    "head_roll": (-30.0, 30.0),
}
# CLIP axes: default range is (0.0, 1.0), added dynamically
_DEFAULT_CLIP_RANGE: tuple[float, float] = (0.0, 1.0)

_NDIM = len(SIGNAL_FIELDS)


@dataclass(frozen=True)
class CategoryProfile:
    """카테고리별 시그널 프로파일.

    참조 이미지 분석 결과의 평균 시그널 + Fisher ratio importance weights.
    """

    name: str
    mean_signals: np.ndarray       # (D,) 정규화된 평균 시그널
    importance_weights: np.ndarray  # (D,) Fisher ratio weights (sum=1)
    n_refs: int                    # 참조 이미지 수


def normalize_signal(value: float, field: str) -> float:
    """단일 시그널을 [0, 1] 범위로 정규화."""
    lo, hi = SIGNAL_RANGES.get(field, _DEFAULT_CLIP_RANGE)
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def extract_signal_vector(
    record: FrameRecord,
    signal_fields: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """FrameRecord에서 정규화된 시그널 벡터 추출.

    Args:
        record: FrameRecord.
        signal_fields: 시그널 필드 순서. None이면 기본 SIGNAL_FIELDS.

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
            # CLIP axis → clip_axes dict
            raw = float(record.clip_axes.get(f, 0.0))
        vec[i] = normalize_signal(raw, f)
    return vec


def extract_signal_vector_from_dict(
    signals: dict[str, float],
    signal_fields: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """dict에서 정규화된 시그널 벡터 추출 (catalog build용).

    Args:
        signals: 시그널 이름 → raw 값 dict.
        signal_fields: 시그널 필드 순서. None이면 기본 SIGNAL_FIELDS.

    Returns:
        (D,) 정규화된 시그널 벡터.
    """
    fields = signal_fields or SIGNAL_FIELDS
    ndim = len(fields)
    vec = np.zeros(ndim, dtype=np.float64)
    for i, f in enumerate(fields):
        raw = float(signals.get(f, 0.0))
        vec[i] = normalize_signal(raw, f)
    return vec


def compute_importance_weights(
    category_vectors: dict[str, np.ndarray],
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """카테고리별 pairwise Fisher ratio importance weights 계산.

    각 카테고리별로 다른 모든 카테고리와의 pairwise Fisher ratio를 구한 뒤 평균.
    이전 global inter-variance 방식의 문제 해결:
    - global: cool_gaze의 em_neutral 분산이 wild_energy weights까지 지배
    - pairwise: 각 카테고리는 실제 이웃 카테고리와의 차이에 집중

    fisher_ij(d) = (mean_i(d) - mean_j(d))^2 / (var_i(d) + var_j(d) + eps)
    fisher_i(d) = mean_j(fisher_ij(d))  (leave-one-out average)

    정규화 전 sqrt 변환으로 dynamic range 압축:
    - AU/emotion 차원의 극단적 Fisher 값이 pose/CLIP 차원을 압도하는 것을 방지
    - 100:1 비율 → sqrt → 10:1 비율

    Args:
        category_vectors: 카테고리 이름 → (N, D) 시그널 벡터 행렬.
        epsilon: 분모 안정화 상수.

    Returns:
        카테고리 이름 → (D,) importance weights (sum=1).
    """
    if not category_vectors:
        return {}

    cat_names = sorted(category_vectors.keys())
    n_cats = len(cat_names)

    # 카테고리별 평균과 분산
    cat_means = np.zeros((n_cats, _NDIM), dtype=np.float64)
    cat_vars = np.zeros((n_cats, _NDIM), dtype=np.float64)

    for ci, name in enumerate(cat_names):
        vecs = category_vectors[name]  # (N, D)
        if len(vecs) == 0:
            continue
        cat_means[ci] = vecs.mean(axis=0)
        cat_vars[ci] = vecs.var(axis=0) if len(vecs) > 1 else np.zeros(_NDIM)

    result = {}
    for ci, name in enumerate(cat_names):
        # Pairwise Fisher ratio against each other category, then average
        fisher_sum = np.zeros(_NDIM, dtype=np.float64)
        n_pairs = 0
        for cj in range(n_cats):
            if cj == ci:
                continue
            inter_sq = (cat_means[ci] - cat_means[cj]) ** 2  # (D,)
            pooled_var = cat_vars[ci] + cat_vars[cj]  # (D,)
            fisher_sum += inter_sq / (pooled_var + epsilon)
            n_pairs += 1

        if n_pairs > 0:
            fisher = fisher_sum / n_pairs
        else:
            fisher = np.ones(_NDIM, dtype=np.float64)

        # sqrt 변환: dynamic range 압축 (dominant dimension 억제)
        fisher = np.sqrt(fisher)

        # Normalize to sum=1
        total = fisher.sum()
        if total > 0:
            weights = fisher / total
        else:
            weights = np.ones(_NDIM, dtype=np.float64) / _NDIM
        result[name] = weights

    return result


def match_category(
    frame_vec: np.ndarray,
    profiles: List[CategoryProfile],
) -> Tuple[float, str]:
    """프레임 시그널 벡터를 가장 가까운 카테고리에 매칭.

    가중 유클리디안 거리 → similarity 변환:
        d = sqrt(sum(w_i * (x_i - mu_i)^2))
        sim = 1.0 / (1.0 + d)

    Args:
        frame_vec: (D,) 정규화된 시그널 벡터.
        profiles: 카테고리 프로파일 리스트.

    Returns:
        (similarity_score, category_name). 프로파일이 비어있으면 (0.0, "").
    """
    if not profiles:
        return 0.0, ""

    best_sim = -1.0
    best_name = ""

    for profile in profiles:
        diff = frame_vec - profile.mean_signals
        weighted_sq = profile.importance_weights * (diff ** 2)
        d = sqrt(float(weighted_sq.sum()))
        sim = 1.0 / (1.0 + d)

        if sim > best_sim:
            best_sim = sim
            best_name = profile.name

    return best_sim, best_name


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


def load_profiles(catalog_path: Path) -> List[CategoryProfile]:
    """카탈로그 디렉토리에서 카테고리별 _profile.json 로딩.

    카탈로그 구조:
        catalog_path/
          categories/
            warm_smile/_profile.json
            cool_gaze/_profile.json
            ...

    _profile.json 형식:
        {
            "name": "warm_smile",
            "mean_signals": [0.5, 0.1, ...],  # (D,)
            "importance_weights": [0.08, 0.12, ...],  # (D,) sum=1
            "n_refs": 7
        }

    Args:
        catalog_path: 카탈로그 루트 디렉토리.

    Returns:
        CategoryProfile 리스트.

    Raises:
        FileNotFoundError: catalog_path 또는 categories/ 미존재.
    """
    categories_dir = catalog_path / "categories"
    if not categories_dir.is_dir():
        raise FileNotFoundError(
            f"Categories directory not found: {categories_dir}"
        )

    profiles: List[CategoryProfile] = []
    for cat_dir in sorted(categories_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue

        profile_path = cat_dir / "_profile.json"
        if not profile_path.exists():
            # Skip empty placeholder categories (no ref images → no profile)
            has_refs = any(f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif") for f in cat_dir.iterdir())
            if has_refs:
                raise FileNotFoundError(
                    f"Missing _profile.json in category '{cat_dir.name}' "
                    f"(has reference images). Run: momentscan catalog-build {catalog_path}"
                )
            continue

        with open(profile_path) as f:
            data = json.load(f)

        mean_signals = np.array(data["mean_signals"], dtype=np.float64)
        importance_weights = np.array(data["importance_weights"], dtype=np.float64)

        if len(mean_signals) != _NDIM:
            raise ValueError(
                f"Profile '{cat_dir.name}' has {len(mean_signals)} signals "
                f"(expected {_NDIM}). Regenerate with: momentscan catalog-build {catalog_path}"
            )

        profiles.append(CategoryProfile(
            name=data.get("name", cat_dir.name),
            mean_signals=mean_signals,
            importance_weights=importance_weights,
            n_refs=data.get("n_refs", 0),
        ))

    if not profiles:
        raise ValueError(
            f"No valid category profiles found in {categories_dir}. "
            f"Run: momentscan catalog-build {catalog_path}"
        )

    logger.info("Loaded %d catalog profiles from %s", len(profiles), catalog_path)
    return profiles


def save_profiles(
    catalog_path: Path,
    profiles: List[CategoryProfile],
) -> None:
    """카테고리별 _profile.json 저장.

    Args:
        catalog_path: 카탈로그 루트 디렉토리.
        profiles: 저장할 프로파일 리스트.
    """
    for profile in profiles:
        cat_dir = catalog_path / "categories" / profile.name
        cat_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "name": profile.name,
            "mean_signals": profile.mean_signals.tolist(),
            "importance_weights": profile.importance_weights.tolist(),
            "n_refs": profile.n_refs,
            "signal_fields": list(SIGNAL_FIELDS),
        }

        profile_path = cat_dir / "_profile.json"
        with open(profile_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d profiles to %s", len(profiles), catalog_path)
