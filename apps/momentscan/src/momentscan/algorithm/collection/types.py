"""Unified collection data types.

CollectionRecord = FrameRecord signals + IdentityRecord embeddings.
CollectionEngine uses these to select diverse portrait frames via Pose × Category grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CollectionRecord:
    """Per-frame record combining signals + ArcFace embedding.

    Replaces the dual FrameRecord (70+ scalar) + IdentityRecord (embedding)
    accumulation with a single record optimized for selection.
    """

    frame_idx: int
    timestamp_ms: float

    # ArcFace 512D (from face.detect, L2-normalized)
    e_id: Optional[np.ndarray] = None

    # Face detection
    face_detected: bool = False
    face_confidence: float = 0.0
    face_area_ratio: float = 0.0

    # Head pose
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    # Expression
    smile_intensity: float = 0.0
    em_neutral: float = 0.0
    mouth_open_ratio: float = 0.0
    eye_open_ratio: float = 0.0

    # Quality
    blur_score: float = 0.0
    face_blur: float = 0.0
    face_identity: float = 0.0  # ArcFace anchor similarity

    # AU (pivot classification) — 10 Action Units
    au_intensities: Optional[Dict[str, float]] = None
    au1_inner_brow: float = 0.0
    au2_outer_brow: float = 0.0
    au4_brow_lowerer: float = 0.0
    au5_upper_lid: float = 0.0
    au6_cheek_raiser: float = 0.0
    au9_nose_wrinkler: float = 0.0
    au12_lip_corner: float = 0.0
    au15_lip_depressor: float = 0.0
    au25_lips_part: float = 0.0
    au26_jaw_drop: float = 0.0

    # Emotion probabilities (from face.expression / HSEmotion)
    em_happy: float = 0.0
    em_surprise: float = 0.0
    em_angry: float = 0.0

    # CLIP axes (dynamic, impact scoring)
    clip_axes: Dict[str, float] = field(default_factory=dict)

    # Gate
    gate_passed: bool = True
    passenger_suitability: float = 0.0

    # Crop/image info (export)
    face_crop_box: Optional[tuple] = None
    face_bbox: Optional[tuple] = None
    image_size: Optional[tuple] = None

    # Catalog
    catalog_best: float = 0.0
    catalog_primary: str = ""
    catalog_category: str = ""
    catalog_scores: Dict[str, float] = field(default_factory=dict)

    # Bind (visualbind XGBoost)
    bind_best: float = 0.0
    bind_primary: str = ""
    bind_scores: Dict[str, float] = field(default_factory=dict)
    bind_pose: str = ""  # XGBoost pose prediction (front/angle/side)

    person_id: int = 0


@dataclass
class CollectionConfig:
    """Configuration for CollectionEngine."""

    # Identity parameters
    tau_id: float = 0.35
    medoid_max_candidates: int = 200
    anchor_max_yaw: float = 15.0  # medoid 후보 선택용

    # Grid selection (replaces anchor/coverage/challenge)
    grid_top_k: int = 2                  # 셀당 최대 프레임
    grid_min_interval_ms: float = 1500.0
    grid_quality_alpha: float = 0.3      # quality 보정 계수

    # Gate thresholds (단일 게이트)
    gate_face_confidence: float = 0.7
    gate_blur_min: float = 50.0

    # Clip extraction
    clip_pre_sec: float = 1.0
    clip_post_sec: float = 1.5

    # Pivots path
    pivots_dir: Optional[str] = None


@dataclass
class SelectedFrame:
    """A selected frame from the grid-based collection process."""

    frame_idx: int
    timestamp_ms: float
    set_type: str = "grid"           # backward compat
    pose_name: str = ""
    pivot_name: str = ""             # = category_name (catalog) 또는 pivot_name (fallback)
    cell_key: str = ""               # "frontal|warm_smile"
    quality_score: float = 0.0
    cell_score: float = 0.0          # pose_fit * catalog_sim * (1 + α * quality)
    catalog_sim: float = 0.0
    pose_fit: float = 0.0
    stable_score: float = 0.0
    face_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None

    # Backward compat aliases
    @property
    def impact_score(self) -> float:
        return self.cell_score

    @property
    def combined_score(self) -> float:
        return self.cell_score


@dataclass
class PersonCollection:
    """Collection result for one person (grid-based)."""

    person_id: int
    prototype_frame_idx: int = -1
    grid: Dict[str, List[SelectedFrame]] = field(default_factory=dict)
    # key = "pose_name|category_name"
    pose_coverage: Dict[str, int] = field(default_factory=dict)
    category_coverage: Dict[str, int] = field(default_factory=dict)
    catalog_mode: bool = False

    # Backward compat properties
    @property
    def anchor_frames(self) -> List[SelectedFrame]:
        return self.all_frames()

    @property
    def coverage_frames(self) -> List[SelectedFrame]:
        return []

    @property
    def challenge_frames(self) -> List[SelectedFrame]:
        return []

    @property
    def pivot_coverage(self) -> Dict[str, int]:
        return self.category_coverage

    def all_frames(self) -> List[SelectedFrame]:
        return [f for frames in self.grid.values() for f in frames]

    def query(
        self,
        *,
        pose_name: Optional[str] = None,
        pivot_name: Optional[str] = None,
        top_k: int = 3,
    ) -> List[SelectedFrame]:
        """Context-based top-k selection."""
        frames = self.all_frames()
        filtered = []
        for f in frames:
            if pose_name is not None and f.pose_name != pose_name:
                continue
            if pivot_name is not None and f.pivot_name != pivot_name:
                continue
            filtered.append(f)
        filtered.sort(key=lambda f: f.cell_score, reverse=True)
        return filtered[:top_k]

    def find_nearest(
        self,
        target_pose: str,
        target_category: str,
    ) -> Optional[SelectedFrame]:
        """Find nearest occupied cell frame.

        Fallback chain: exact cell → same pose → same category → best overall.
        """
        # Exact match
        key = f"{target_pose}|{target_category}"
        if key in self.grid and self.grid[key]:
            return max(self.grid[key], key=lambda f: f.cell_score)

        # Same pose
        same_pose = [
            f for k, frames in self.grid.items()
            for f in frames if f.pose_name == target_pose
        ]
        if same_pose:
            return max(same_pose, key=lambda f: f.cell_score)

        # Same category
        same_cat = [
            f for k, frames in self.grid.items()
            for f in frames if f.pivot_name == target_category
        ]
        if same_cat:
            return max(same_cat, key=lambda f: f.cell_score)

        # Best overall
        all_f = self.all_frames()
        if all_f:
            return max(all_f, key=lambda f: f.cell_score)

        return None


@dataclass
class CollectionResult:
    """Full collection result."""

    persons: Dict[int, PersonCollection] = field(default_factory=dict)
    frame_count: int = 0
    config: Optional[CollectionConfig] = None
    _timeseries: Optional[Dict[str, Any]] = field(default=None, repr=False)
