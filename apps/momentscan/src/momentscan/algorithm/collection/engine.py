"""Unified collection engine — Pose × Category grid selection.

Selects diverse portrait frames using a grid of (pose, category) cells.
Each cell gets top-k frames ranked by cell_score = pose_fit × catalog_sim × (1 + α × quality).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from momentscan.algorithm.collection.pivots import (
    DEFAULT_PIVOTS,
    DEFAULT_POSES,
    ExpressionPivot,
    PoseTarget,
    classify_expression,
    classify_pose,
    load_collection_pivots,
)
from momentscan.algorithm.collection.types import (
    CollectionConfig,
    CollectionRecord,
    CollectionResult,
    PersonCollection,
    SelectedFrame,
)

logger = logging.getLogger(__name__)


class CollectionEngine:
    """Unified collection engine — Pose × Category grid selection.

    1. Quality gate → ArcFace embedding filter
    2. Medoid prototype (ArcFace cosine similarity center)
    3. Stability filter (cos(e_id, prototype) > tau_id)
    4. Pose classification + category assignment (catalog or AU-rule fallback)
    5. Quality scoring (blur, size, frontalness, confidence)
    6. Cell scoring: pose_fit × catalog_sim × (1 + α × quality)
    7. Grid selection: per-cell top-k with temporal dedup
    """

    def __init__(
        self,
        config: Optional[CollectionConfig] = None,
        poses: Optional[List[PoseTarget]] = None,
        pivots: Optional[List[ExpressionPivot]] = None,
        catalog_profiles: Optional[List] = None,
    ):
        self.config = config or CollectionConfig()
        self._poses = poses or list(DEFAULT_POSES)
        self._pivots = pivots or list(DEFAULT_PIVOTS)
        self._catalog_profiles = catalog_profiles or []

    @classmethod
    def from_collection_path(
        cls,
        collection_path: Optional[str] = None,
        config: Optional[CollectionConfig] = None,
    ) -> "CollectionEngine":
        """Create engine from a single collection/catalog directory.

        Args:
            collection_path: Path to collection/catalog directory. When set,
                loads poses, pivots, and signal profiles from the directory.
                None = use built-in defaults.
            config: Optional CollectionConfig override.
        """
        if collection_path is None:
            return cls(config=config)

        dir_path = Path(collection_path)
        poses, pivots = load_collection_pivots(dir_path)

        from momentscan.algorithm.batch.catalog_scoring import load_profiles
        catalog_profiles = load_profiles(dir_path)

        return cls(config=config, poses=poses, pivots=pivots, catalog_profiles=catalog_profiles)

    def collect(self, records: List[CollectionRecord]) -> CollectionResult:
        """Run the unified collection pipeline.

        Args:
            records: CollectionRecords accumulated during on_frame().

        Returns:
            CollectionResult with selected frames per person.
        """
        if not records:
            return CollectionResult(frame_count=0, config=self.config)

        # Group by person_id
        groups: Dict[int, List[CollectionRecord]] = defaultdict(list)
        for r in records:
            groups[r.person_id].append(r)

        persons: Dict[int, PersonCollection] = {}
        for pid, pid_records in groups.items():
            person = self._collect_person(pid, pid_records)
            if person is not None:
                persons[pid] = person

        return CollectionResult(
            persons=persons,
            frame_count=len(records),
            config=self.config,
        )

    def _collect_person(
        self, person_id: int, records: List[CollectionRecord]
    ) -> Optional[PersonCollection]:
        """Collect frames for one person using grid selection."""
        cfg = self.config
        use_catalog = bool(self._catalog_profiles)

        # (a) Strict gate → medoid candidates
        strict = [r for r in records if self._pass_strict_gate(r)]
        if len(strict) < 3:
            logger.info(
                "Person %d: too few strict-gate frames (%d) — skipping",
                person_id, len(strict),
            )
            return None

        # (b) Medoid prototype from frontal strict-gate frames
        medoid_candidates = [r for r in strict if abs(r.head_yaw) <= cfg.anchor_max_yaw]
        if not medoid_candidates:
            medoid_candidates = strict
        prototype, proto_idx = self._compute_medoid(medoid_candidates)

        # (c) Score all records: stability + classify + quality + cell_score
        scored: List[Tuple[CollectionRecord, SelectedFrame]] = []
        for r in records:
            if r.e_id is None:
                continue
            stable = float(np.dot(r.e_id, prototype))
            if stable < cfg.tau_id:
                continue

            pose_name = classify_pose(r.head_yaw, r.head_pitch, self._poses) or "other"
            pose_fit = self._compute_pose_fit(r, pose_name)

            if use_catalog:
                category = r.catalog_primary or "other"
                catalog_sim = r.catalog_best
            else:
                category = classify_expression(r, self._pivots)
                catalog_sim = 1.0  # fallback: degenerate to quality × pose_fit

            quality = self._compute_quality(r)
            cell_score = self._compute_cell_score(
                pose_fit, catalog_sim, quality, cfg.grid_quality_alpha,
            )
            cell_key = f"{pose_name}|{category}"

            sf = SelectedFrame(
                frame_idx=r.frame_idx,
                timestamp_ms=r.timestamp_ms,
                set_type="grid",
                pose_name=pose_name,
                pivot_name=category,
                cell_key=cell_key,
                quality_score=quality,
                cell_score=cell_score,
                catalog_sim=catalog_sim,
                pose_fit=pose_fit,
                stable_score=stable,
                face_crop_box=r.face_crop_box,
                image_size=r.image_size,
            )
            scored.append((r, sf))

        if not scored:
            return None

        # (d) Grid selection: per-cell top-k with temporal dedup
        grid = self._select_grid(scored)

        # Coverage statistics
        pose_cov: Dict[str, int] = defaultdict(int)
        cat_cov: Dict[str, int] = defaultdict(int)
        for key, frames in grid.items():
            for f in frames:
                pose_cov[f.pose_name] += 1
                cat_cov[f.pivot_name] += 1

        n_total = sum(len(fs) for fs in grid.values())
        mode_str = "catalog" if use_catalog else "fallback"
        logger.info(
            "Person %d: %d frames in %d cells (%s) (from %d scored / %d total)",
            person_id, n_total, len(grid), mode_str,
            len(scored), len(records),
        )

        return PersonCollection(
            person_id=person_id,
            prototype_frame_idx=proto_idx,
            grid=grid,
            pose_coverage=dict(pose_cov),
            category_coverage=dict(cat_cov),
            catalog_mode=use_catalog,
        )

    # ── Medoid ──

    def _compute_medoid(
        self, records: List[CollectionRecord]
    ) -> Tuple[np.ndarray, int]:
        """ArcFace embedding medoid (max average similarity)."""
        cfg = self.config

        if len(records) > cfg.medoid_max_candidates:
            indices = np.random.default_rng(42).choice(
                len(records), cfg.medoid_max_candidates, replace=False
            )
            candidates = [records[i] for i in indices]
        else:
            candidates = records

        embeddings = np.array([r.e_id for r in candidates])
        sim_matrix = embeddings @ embeddings.T
        avg_sim = sim_matrix.mean(axis=1)
        medoid_idx = int(np.argmax(avg_sim))

        return embeddings[medoid_idx].copy(), candidates[medoid_idx].frame_idx

    # ── Quality scoring ──

    def _compute_quality(self, r: CollectionRecord) -> float:
        """Per-frame quality score.

        blur_norm * 0.3 + face_size * 0.3 + frontalness * 0.2 + confidence * 0.2
        """
        blur_source = r.face_blur if r.face_blur > 0 else r.blur_score
        blur_norm = min(blur_source / 500.0, 1.0) if blur_source > 0 else 0.5
        face_size_norm = min(r.face_area_ratio / 0.3, 1.0)
        frontalness = max(0.0, 1.0 - abs(r.head_yaw) / 45.0)
        confidence = r.face_confidence

        return (
            0.3 * blur_norm
            + 0.3 * face_size_norm
            + 0.2 * frontalness
            + 0.2 * confidence
        )

    # ── Pose fit ──

    def _compute_pose_fit(self, r: CollectionRecord, pose_name: str) -> float:
        """Pose fit: 1.0 - (distance / r_accept), clamped [0, 1]."""
        target = next((p for p in self._poses if p.name == pose_name), None)
        if target is None:
            return 0.5  # "other" pose gets neutral fit

        d_yaw = abs(r.head_yaw) - target.yaw
        d_pitch = r.head_pitch - target.pitch
        dist = math.sqrt(d_yaw ** 2 + d_pitch ** 2)
        return max(0.0, 1.0 - dist / target.r_accept)

    # ── Cell scoring ──

    @staticmethod
    def _compute_cell_score(
        pose_fit: float,
        catalog_sim: float,
        quality: float,
        alpha: float,
    ) -> float:
        """cell_score = pose_fit × catalog_sim × (1 + α × quality)."""
        return pose_fit * catalog_sim * (1.0 + alpha * quality)

    # ── Grid selection ──

    def _select_grid(
        self, scored: List[Tuple[CollectionRecord, SelectedFrame]]
    ) -> Dict[str, List[SelectedFrame]]:
        """Per-cell top-k selection with temporal dedup."""
        cfg = self.config

        # Group by cell_key
        cells: Dict[str, List[Tuple[CollectionRecord, SelectedFrame]]] = defaultdict(list)
        for r, sf in scored:
            cells[sf.cell_key].append((r, sf))

        grid: Dict[str, List[SelectedFrame]] = {}
        for key, items in cells.items():
            # Sort by cell_score descending
            items.sort(key=lambda x: x[1].cell_score, reverse=True)

            selected: List[SelectedFrame] = []
            selected_ts: List[float] = []

            for r, sf in items:
                if len(selected) >= cfg.grid_top_k:
                    break
                if self._too_close_temporally(
                    sf.timestamp_ms, selected_ts, cfg.grid_min_interval_ms
                ):
                    continue
                selected.append(sf)
                selected_ts.append(sf.timestamp_ms)

            if selected:
                grid[key] = selected

        return grid

    # ── Gate ──

    def _pass_strict_gate(self, r: CollectionRecord) -> bool:
        cfg = self.config
        if r.e_id is None:
            return False
        if r.face_confidence < cfg.gate_face_confidence:
            return False
        if r.blur_score > 0 and r.blur_score < cfg.gate_blur_min:
            return False
        return True

    # ── Helpers ──

    @staticmethod
    def _too_close_temporally(
        timestamp_ms: float,
        existing: List[float],
        min_interval_ms: float,
    ) -> bool:
        for ts in existing:
            if abs(timestamp_ms - ts) < min_interval_ms:
                return True
        return False
