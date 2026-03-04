"""Signal-Profile Catalog 빌더.

참조 이미지를 파이프라인 analyzer로 분석하여 카테고리별 시그널 프로파일을 생성.

Usage:
    1. CLI: ``momentscan catalog-build catalogs/portrait-v1``
    2. Python:
        >>> from momentscan.algorithm.batch.catalog_build import build_catalog_profiles
        >>> profiles = build_catalog_profiles(Path("catalogs/portrait-v1"))
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from momentscan.algorithm.batch.catalog_scoring import (
    SIGNAL_FIELDS,
    CategoryProfile,
    compute_importance_weights,
    extract_signal_vector_from_dict,
    save_profiles,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}

# 참조 이미지 분석에 필요한 analyzer 이름
_REQUIRED_ANALYZERS = [
    "face.detect",
    "face.expression",
    "face.au",
    "head.pose",
    "portrait.score",
]


def build_profiles_from_signals(
    signals_by_category: Dict[str, List[Dict[str, float]]],
) -> List[CategoryProfile]:
    """시그널 dict에서 카테고리 프로파일 생성 (순수 수학).

    Args:
        signals_by_category: 카테고리 이름 → 참조 이미지별 시그널 dict 리스트.
            각 dict의 key는 SIGNAL_FIELDS의 필드명.

    Returns:
        CategoryProfile 리스트.
    """
    if not signals_by_category:
        return []

    # 카테고리별 정규화된 시그널 벡터 행렬
    category_vectors: Dict[str, np.ndarray] = {}
    for cat_name, signal_list in signals_by_category.items():
        if not signal_list:
            continue
        vecs = np.stack([
            extract_signal_vector_from_dict(s) for s in signal_list
        ])
        category_vectors[cat_name] = vecs

    if not category_vectors:
        return []

    # Fisher ratio importance weights
    importance = compute_importance_weights(category_vectors)

    profiles = []
    for cat_name in sorted(category_vectors.keys()):
        vecs = category_vectors[cat_name]
        mean_vec = vecs.mean(axis=0)
        weights = importance[cat_name]

        profiles.append(CategoryProfile(
            name=cat_name,
            mean_signals=mean_vec,
            importance_weights=weights,
            n_refs=len(vecs),
        ))

    return profiles


def _discover_categories(catalog_path: Path) -> Dict[str, List[Path]]:
    """카탈로그 디렉토리에서 카테고리별 참조 이미지 경로를 찾는다.

    Returns:
        카테고리 이름 → 이미지 파일 경로 리스트.
    """
    categories_dir = catalog_path / "categories"
    if not categories_dir.is_dir():
        raise FileNotFoundError(
            f"Categories directory not found: {categories_dir}"
        )

    result: Dict[str, List[Path]] = {}
    for cat_dir in sorted(categories_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue

        # 카테고리 이름: category.yaml에서 override 가능
        cat_name = cat_dir.name
        cat_yaml_path = cat_dir / "category.yaml"
        if cat_yaml_path.exists():
            try:
                import yaml
                with open(cat_yaml_path) as f:
                    cat_yaml = yaml.safe_load(f) or {}
                cat_name = cat_yaml.get("name", cat_name)
            except ImportError:
                pass

        images = sorted(
            f for f in cat_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and not f.name.startswith(".")
        )
        if images:
            result[cat_name] = images
        else:
            logger.warning("Category %s: no images found — skipping", cat_name)

    return result


def _extract_signals_from_observations(obs_by_source: Dict[str, Any]) -> Dict[str, float]:
    """Analyzer observation dict에서 시그널을 추출.

    extract.py의 개별 추출 함수와 동일한 로직을 재사용하되,
    FrameRecord 대신 plain dict로 반환.

    Args:
        obs_by_source: source name → Observation 매핑.

    Returns:
        시그널 이름 → 값 dict.
    """
    signals: Dict[str, float] = {}

    # face.detect → geometric head pose (fallback)
    fd_obs = obs_by_source.get("face.detect")
    if fd_obs is not None:
        data = getattr(fd_obs, "data", None)
        faces = getattr(data, "faces", None) if data else None
        if faces:
            face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
            signals["head_yaw_dev"] = abs(float(getattr(face, "yaw", 0.0)))
            signals["head_pitch"] = float(getattr(face, "pitch", 0.0))
            signals["head_roll"] = float(getattr(face, "roll", 0.0))

    # head.pose → 6DRepNet precise override
    pose_obs = obs_by_source.get("head.pose")
    if pose_obs is not None:
        data = getattr(pose_obs, "data", None)
        estimates = getattr(data, "estimates", None) if data else None
        if estimates:
            est = estimates[0]
            yaw = getattr(est, "yaw", None)
            if yaw is not None:
                signals["head_yaw_dev"] = abs(float(yaw))
            pitch = getattr(est, "pitch", None)
            if pitch is not None:
                signals["head_pitch"] = float(pitch)
            roll = getattr(est, "roll", None)
            if roll is not None:
                signals["head_roll"] = float(roll)

    # face.expression → emotion probabilities
    expr_obs = obs_by_source.get("face.expression")
    if expr_obs is not None:
        data = getattr(expr_obs, "data", None)
        faces = getattr(data, "faces", None) if data else None
        if faces:
            face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
            face_signals = getattr(face, "signals", {}) or {}
            signals["em_happy"] = float(face_signals.get("em_happy", 0.0))
            signals["em_neutral"] = float(face_signals.get("em_neutral", 0.0))
            signals["em_surprise"] = float(face_signals.get("em_surprise", 0.0))
            signals["em_angry"] = float(face_signals.get("em_angry", 0.0))

    # face.au → 10 AU intensities
    au_obs = obs_by_source.get("face.au")
    if au_obs is not None:
        data = getattr(au_obs, "data", None)
        au_list = getattr(data, "au_intensities", None) if data else None
        if au_list:
            au = au_list[0]
            for au_name, field_name in [
                ("AU1", "au1_inner_brow"), ("AU2", "au2_outer_brow"),
                ("AU4", "au4_brow_lowerer"), ("AU5", "au5_upper_lid"),
                ("AU6", "au6_cheek_raiser"), ("AU9", "au9_nose_wrinkler"),
                ("AU12", "au12_lip_corner"), ("AU15", "au15_lip_depressor"),
                ("AU25", "au25_lips_part"), ("AU26", "au26_jaw_drop"),
            ]:
                signals[field_name] = float(au.get(au_name, 0.0))

    # portrait.score → CLIP axes (axis name = signal field name)
    ps_obs = obs_by_source.get("portrait.score")
    if ps_obs is not None:
        metadata = getattr(ps_obs, "metadata", None) or {}
        clip_axes = metadata.get("_clip_axes")
        if clip_axes:
            for ax in clip_axes:
                # Axis name matches catalog category name (e.g., "warm_smile")
                signals[ax.name] = float(ax.score)

    return signals


def _analyze_image(
    image: np.ndarray,
    analyzers: Dict[str, Any],
) -> Dict[str, Any]:
    """단일 이미지를 analyzer DAG로 분석.

    Stateful analyzer (BBoxSmoother, clip_stride 캐시, tracker)를
    이미지마다 reset하여 독립적 분석을 보장한다.

    Args:
        image: BGR 이미지.
        analyzers: analyzer name → Module 인스턴스.

    Returns:
        source name → Observation 매핑.
    """
    from visualbase import Frame

    # Reset stateful analyzers to prevent state bleeding between images
    for module in analyzers.values():
        if hasattr(module, "reset"):
            module.reset()

    frame = Frame.from_array(image, frame_id=0, t_src_ns=0)

    obs_by_source: Dict[str, Any] = {}

    # face.detect (no deps)
    if "face.detect" in analyzers:
        obs = analyzers["face.detect"].process(frame)
        if obs is not None:
            # Verify at least one face was detected
            data = getattr(obs, "data", None)
            faces = getattr(data, "faces", None) if data else None
            if faces:
                obs_by_source["face.detect"] = obs

    if "face.detect" not in obs_by_source:
        return obs_by_source

    face_deps = {"face.detect": obs_by_source["face.detect"]}

    # head.pose (deps: face.detect)
    if "head.pose" in analyzers:
        obs = analyzers["head.pose"].process(frame, deps=face_deps)
        if obs is not None:
            obs_by_source["head.pose"] = obs

    # face.expression (deps: face.detect)
    if "face.expression" in analyzers:
        obs = analyzers["face.expression"].process(frame, deps=face_deps)
        if obs is not None:
            obs_by_source["face.expression"] = obs

    # face.au (deps: face.detect)
    if "face.au" in analyzers:
        obs = analyzers["face.au"].process(frame, deps=face_deps)
        if obs is not None:
            obs_by_source["face.au"] = obs

    # portrait.score (deps: face.detect)
    if "portrait.score" in analyzers:
        obs = analyzers["portrait.score"].process(frame, deps=face_deps)
        if obs is not None:
            obs_by_source["portrait.score"] = obs

    return obs_by_source


def _load_analyzers(analyzer_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Entry point에서 analyzer 모듈을 로드하고 초기화.

    Args:
        analyzer_names: 로드할 analyzer 이름. None이면 _REQUIRED_ANALYZERS 전체.

    Returns:
        analyzer name → 초기화된 Module 인스턴스.
    """
    from importlib.metadata import entry_points

    names = analyzer_names or _REQUIRED_ANALYZERS
    analyzers = {}

    eps = entry_points(group="visualpath.modules")
    ep_map = {ep.name: ep for ep in eps}

    for name in names:
        ep = ep_map.get(name)
        if ep is None:
            logger.warning("Analyzer %s not found in entry points — skipping", name)
            continue
        try:
            cls = ep.load()
            module = cls()
            module.initialize()
            analyzers[name] = module
            logger.info("Loaded analyzer: %s", name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)

    return analyzers


def _build_manifest(catalog_path: Path, categories: Dict[str, List[Path]]) -> Dict[str, str]:
    """파일 해시 manifest (캐시 무효화용)."""
    manifest = {}
    for cat_name, images in sorted(categories.items()):
        for img_path in images:
            h = hashlib.sha256(img_path.read_bytes()).hexdigest()
            manifest[str(img_path.relative_to(catalog_path))] = h
    return manifest


def _cache_path(catalog_path: Path) -> Path:
    """Signal profile 캐시 디렉토리."""
    return catalog_path / "_cache" / "signal_profiles"


def _load_cache(
    catalog_path: Path, current_manifest: Dict[str, str],
) -> Optional[List[CategoryProfile]]:
    """캐시된 프로파일 로드 (manifest 일치 시)."""
    cache_dir = _cache_path(catalog_path)
    manifest_path = cache_dir / "manifest.json"
    profiles_path = cache_dir / "profiles.json"

    if not manifest_path.exists() or not profiles_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            cached_manifest = json.load(f)
        if cached_manifest != current_manifest:
            logger.info("Signal profile cache invalidated (manifest changed)")
            return None

        with open(profiles_path) as f:
            data = json.load(f)

        profiles = []
        for p in data:
            profiles.append(CategoryProfile(
                name=p["name"],
                mean_signals=np.array(p["mean_signals"], dtype=np.float64),
                importance_weights=np.array(p["importance_weights"], dtype=np.float64),
                n_refs=p["n_refs"],
            ))
        logger.info("Loaded %d profiles from cache", len(profiles))
        return profiles
    except Exception as e:
        logger.warning("Cache load failed: %s", e)
        return None


def _save_cache(
    catalog_path: Path,
    profiles: List[CategoryProfile],
    manifest: Dict[str, str],
) -> None:
    """프로파일 캐시 저장."""
    cache_dir = _cache_path(catalog_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "name": p.name,
            "mean_signals": p.mean_signals.tolist(),
            "importance_weights": p.importance_weights.tolist(),
            "n_refs": p.n_refs,
        }
        for p in profiles
    ]

    with open(cache_dir / "profiles.json", "w") as f:
        json.dump(data, f, indent=2)

    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved profile cache to %s", cache_dir)


def build_catalog_profiles(
    catalog_path: Path,
    analyzer_names: Optional[List[str]] = None,
    cache_enabled: bool = True,
) -> List[CategoryProfile]:
    """참조 이미지 분석 → 카테고리별 시그널 프로파일 생성.

    1. 카테고리별 참조 이미지 발견
    2. (캐시 체크)
    3. 각 이미지에 face.detect + face.au + face.expression + head.pose + portrait.score 실행
    4. 시그널 벡터 추출
    5. 카테고리별 mean + importance weights 계산
    6. _profile.json + 캐시 저장

    Args:
        catalog_path: 카탈로그 루트 디렉토리.
        analyzer_names: 사용할 analyzer 이름. None이면 기본 세트.
        cache_enabled: 캐시 사용 여부.

    Returns:
        CategoryProfile 리스트.
    """
    categories = _discover_categories(catalog_path)
    if not categories:
        raise ValueError(f"No categories with images found in {catalog_path}")

    total_images = sum(len(imgs) for imgs in categories.values())
    logger.info(
        "Discovered %d categories, %d total reference images",
        len(categories), total_images,
    )

    # Cache check
    if cache_enabled:
        manifest = _build_manifest(catalog_path, categories)
        cached = _load_cache(catalog_path, manifest)
        if cached is not None:
            return cached

    # Load analyzers
    analyzers = _load_analyzers(analyzer_names)
    if "face.detect" not in analyzers:
        raise RuntimeError("face.detect analyzer is required but failed to load")

    # Analyze reference images
    signals_by_category: Dict[str, List[Dict[str, float]]] = {}

    for cat_name, image_paths in sorted(categories.items()):
        cat_signals = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Failed to load: %s", img_path)
                continue

            obs_by_source = _analyze_image(img, analyzers)
            if "face.detect" not in obs_by_source:
                logger.warning("No face detected in %s — skipping", img_path.name)
                continue

            signals = _extract_signals_from_observations(obs_by_source)
            cat_signals.append(signals)
            logger.debug(
                "%s/%s: %s",
                cat_name, img_path.name,
                {k: f"{v:.2f}" for k, v in signals.items()},
            )

        if cat_signals:
            signals_by_category[cat_name] = cat_signals
            logger.info(
                "Category %s: %d/%d images analyzed",
                cat_name, len(cat_signals), len(image_paths),
            )
        else:
            logger.warning("Category %s: no valid images — skipping", cat_name)

    # Build profiles
    profiles = build_profiles_from_signals(signals_by_category)

    # Cleanup analyzers
    for module in analyzers.values():
        if hasattr(module, "cleanup"):
            try:
                module.cleanup()
            except Exception:
                pass

    if not profiles:
        raise ValueError("No valid profiles could be generated")

    # Save _profile.json per category
    save_profiles(catalog_path, profiles)

    # Save cache
    if cache_enabled:
        manifest = _build_manifest(catalog_path, categories)
        _save_cache(catalog_path, profiles, manifest)

    return profiles


from dataclasses import dataclass, field


@dataclass
class SeparationMetrics:
    """카탈로그 카테고리 분리도 지표."""

    n_categories: int = 0
    pairwise_distances: Dict[str, float] = field(default_factory=dict)
    min_distance: float = float("inf")
    min_pair: tuple = ("", "")
    mean_distance: float = 0.0
    silhouette_approx: float = 0.0
    warnings: List[str] = field(default_factory=list)


def compute_separation_metrics(
    profiles: List[CategoryProfile],
) -> SeparationMetrics:
    """카테고리 프로파일 간 분리도 계산.

    경고 기준:
    - min_distance < 0.15: "카테고리 X와 Y가 너무 가깝습니다"
    - silhouette < 0.3: "전체 카테고리 분리도가 낮습니다"
    - 단일 카테고리 참조 이미지 1장: "참조 이미지 추가 권장"
    """
    m = SeparationMetrics(n_categories=len(profiles))

    if len(profiles) < 2:
        if profiles:
            p = profiles[0]
            if p.n_refs < 2:
                m.warnings.append(f"{p.name}: only {p.n_refs} ref(s) — consider adding more")
        return m

    # Pairwise weighted Euclidean distances
    distances: List[float] = []
    for i, p1 in enumerate(profiles):
        for j, p2 in enumerate(profiles):
            if j <= i:
                continue
            diff = p1.mean_signals - p2.mean_signals
            # Average importance weights from both categories
            w = (p1.importance_weights + p2.importance_weights) / 2.0
            d = float(np.sqrt(np.sum(w * diff ** 2)))
            key = f"{p1.name}|{p2.name}"
            m.pairwise_distances[key] = d
            distances.append(d)

            if d < m.min_distance:
                m.min_distance = d
                m.min_pair = (p1.name, p2.name)

    m.mean_distance = float(np.mean(distances)) if distances else 0.0

    # Simplified silhouette: (mean_inter - mean_intra) / max(mean_inter, mean_intra)
    # mean_intra approximated by importance weight spread within each category
    intra_dists = []
    for p in profiles:
        # Intra-cluster spread ~ std of importance weights (proxy)
        intra_dists.append(float(np.std(p.mean_signals)))

    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter = m.mean_distance
    denom = max(mean_inter, mean_intra, 1e-9)
    m.silhouette_approx = (mean_inter - mean_intra) / denom

    # Warnings
    for key, d in m.pairwise_distances.items():
        if d < 0.15:
            a, b = key.split("|")
            m.warnings.append(f"{a} <-> {b}: distance {d:.3f} — categories too close")

    if m.silhouette_approx < 0.3:
        m.warnings.append(
            f"Overall silhouette {m.silhouette_approx:.2f} — low category separation"
        )

    for p in profiles:
        if p.n_refs < 2:
            m.warnings.append(f"{p.name}: only {p.n_refs} ref(s) — consider adding more")

    return m


def print_separation_report(
    profiles: List[CategoryProfile],
    metrics: Optional[SeparationMetrics] = None,
) -> str:
    """카테고리 간 거리 매트릭스 + 주요 시그널 리포트 생성.

    Args:
        profiles: Category profiles to report on.
        metrics: Pre-computed metrics; computed if None.

    Returns:
        포맷된 리포트 문자열.
    """
    if not profiles:
        return "No profiles to report."

    if metrics is None:
        metrics = compute_separation_metrics(profiles)

    lines = []
    lines.append("=== Category Separation Report ===")

    # Pairwise distances
    for key, d in metrics.pairwise_distances.items():
        a, b = key.split("|")
        quality = "GOOD" if d > 0.40 else "CLOSE" if d > 0.15 else "OVERLAP"
        symbol = "\u2713" if quality == "GOOD" else "\u26a0" if quality == "CLOSE" else "\u2717"
        lines.append(f"  {a} <-> {b}: {d:.3f} {symbol}")

    if metrics.pairwise_distances:
        lines.append("")
        lines.append(
            f"  Min distance: {metrics.min_distance:.3f} "
            f"({metrics.min_pair[0]} <-> {metrics.min_pair[1]})"
        )
        lines.append(f"  Silhouette:   {metrics.silhouette_approx:.2f}")

    lines.append("")
    lines.append("=== Top-3 Discriminative Signals ===")

    for p in profiles:
        indices = np.argsort(p.importance_weights)[::-1][:3]
        top3 = [
            f"{SIGNAL_FIELDS[idx]}({p.importance_weights[idx]:.2f})"
            for idx in indices
        ]
        lines.append(f"  {p.name}: {', '.join(top3)}")

    # Warnings
    if metrics.warnings:
        lines.append("")
        lines.append("=== Warnings ===")
        for w in metrics.warnings:
            lines.append(f"  \u26a0 {w}")

    return "\n".join(lines)
