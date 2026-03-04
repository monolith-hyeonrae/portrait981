"""Tests for collection/engine.py — grid-based selection."""

import numpy as np
import pytest
from typing import Optional

from momentscan.algorithm.collection.engine import CollectionEngine
from momentscan.algorithm.collection.types import (
    CollectionConfig,
    CollectionRecord,
    CollectionResult,
)


def _make_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    """L2-normalized random embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_record(
    frame_idx: int = 0,
    timestamp_ms: float = 0.0,
    e_id_seed: Optional[int] = None,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    smile_intensity: float = 0.0,
    face_confidence: float = 0.9,
    face_area_ratio: float = 0.05,
    blur_score: float = 100.0,
    au12_lip_corner: float = 0.0,
    au25_lips_part: float = 0.0,
    person_id: int = 0,
) -> CollectionRecord:
    """Create test CollectionRecord."""
    return CollectionRecord(
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        e_id=_make_embedding(512, e_id_seed) if e_id_seed is not None else None,
        face_detected=True,
        face_confidence=face_confidence,
        face_area_ratio=face_area_ratio,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        smile_intensity=smile_intensity,
        blur_score=blur_score,
        au12_lip_corner=au12_lip_corner,
        au25_lips_part=au25_lips_part,
        person_id=person_id,
    )


def _make_similar_records(n: int, base_seed: int = 42, **kwargs) -> list:
    """Create n records with similar embeddings (same person)."""
    base_emb = _make_embedding(512, base_seed)
    records = []
    for i in range(n):
        r = _make_record(
            frame_idx=i,
            timestamp_ms=i * 2000.0,  # 2s apart
            e_id_seed=base_seed,
            **kwargs,
        )
        # Perturb embedding slightly
        noise = np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.05
        r.e_id = base_emb + noise
        r.e_id = r.e_id / np.linalg.norm(r.e_id)
        records.append(r)
    return records


class TestCollectionEngineEmpty:
    def test_empty_records(self):
        engine = CollectionEngine()
        result = engine.collect([])
        assert result.frame_count == 0
        assert len(result.persons) == 0

    def test_too_few_strict_gate_frames(self):
        """Low confidence → no person built."""
        engine = CollectionEngine()
        records = [
            _make_record(frame_idx=i, e_id_seed=i, face_confidence=0.3)
            for i in range(10)
        ]
        result = engine.collect(records)
        assert len(result.persons) == 0

    def test_no_embeddings(self):
        """Records without e_id → no person built."""
        engine = CollectionEngine()
        records = [
            _make_record(frame_idx=i, face_confidence=0.9)
            for i in range(10)
        ]
        result = engine.collect(records)
        assert len(result.persons) == 0


class TestCollectionEngineBasic:
    def test_basic_collection(self):
        """100 similar records → grid selection produces frames."""
        records = []
        base_emb = _make_embedding(512, 42)
        for i in range(100):
            r = _make_record(
                frame_idx=i,
                timestamp_ms=i * 200.0,
                e_id_seed=42,
                face_confidence=0.9,
                blur_score=100.0,
                head_yaw=(i % 20 - 10) * 3.0,
                head_pitch=(i % 10 - 5) * 3.0,
                smile_intensity=0.5 if i % 5 == 0 else 0.1,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.05
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        engine = CollectionEngine()
        result = engine.collect(records)

        assert result.frame_count == 100
        assert 0 in result.persons

        person = result.persons[0]
        all_frames = person.all_frames()
        assert len(all_frames) > 0
        assert len(person.grid) > 0
        assert person.prototype_frame_idx >= 0

        for f in all_frames:
            assert f.set_type == "grid"
            assert f.quality_score > 0
            assert f.pose_name != ""
            assert f.cell_key != ""

    def test_grid_diverse_cells(self):
        """Grid should span multiple pose × category cells."""
        records = _make_similar_records(
            50, base_seed=42,
            face_confidence=0.9, blur_score=100.0,
        )
        # Spread yaw across different poses
        for i, r in enumerate(records):
            r.head_yaw = [0, 30, -30, 55, -55][i % 5] + np.random.default_rng(i).normal(0, 3)
            # Vary expression for different pivots
            r.au12_lip_corner = [0.0, 2.0, 0.5, 3.0][i % 4]
            r.au25_lips_part = [0.0, 0.5, 2.0, 2.5][i % 4]

        engine = CollectionEngine()
        result = engine.collect(records)

        if 0 in result.persons:
            person = result.persons[0]
            pose_names = {f.pose_name for f in person.all_frames()}
            assert len(pose_names) >= 2

    def test_grid_top_k(self):
        """Each cell should have at most grid_top_k frames."""
        cfg = CollectionConfig(grid_top_k=1)
        records = _make_similar_records(
            20, base_seed=42,
            face_confidence=0.9, blur_score=100.0,
        )
        # All same pose
        for r in records:
            r.head_yaw = 0.0

        engine = CollectionEngine(config=cfg)
        result = engine.collect(records)

        if 0 in result.persons:
            for key, frames in result.persons[0].grid.items():
                assert len(frames) <= 1


class TestCollectionEngineCellScore:
    def test_cell_score_formula(self):
        """Verify cell_score = pose_fit × catalog_sim × (1 + α × quality)."""
        pose_fit = 0.8
        catalog_sim = 0.9
        quality = 0.7
        alpha = 0.3

        expected = pose_fit * catalog_sim * (1.0 + alpha * quality)
        actual = CollectionEngine._compute_cell_score(pose_fit, catalog_sim, quality, alpha)
        assert actual == pytest.approx(expected)

    def test_cell_score_zero_pose_fit(self):
        """Zero pose_fit → zero cell_score."""
        score = CollectionEngine._compute_cell_score(0.0, 0.9, 0.7, 0.3)
        assert score == 0.0

    def test_cell_score_zero_catalog_sim(self):
        """Zero catalog_sim → zero cell_score."""
        score = CollectionEngine._compute_cell_score(0.8, 0.0, 0.7, 0.3)
        assert score == 0.0

    def test_quality_boosts_score(self):
        """Higher quality → higher cell_score."""
        low_q = CollectionEngine._compute_cell_score(0.8, 0.9, 0.1, 0.3)
        high_q = CollectionEngine._compute_cell_score(0.8, 0.9, 0.9, 0.3)
        assert high_q > low_q


class TestCollectionEnginePoseFit:
    def test_pose_fit_exact_match(self):
        """Frame at exact pose center → fit close to 1.0."""
        engine = CollectionEngine()
        r = _make_record(head_yaw=0.0, head_pitch=0.0)
        fit = engine._compute_pose_fit(r, "frontal")
        assert fit == pytest.approx(1.0)

    def test_pose_fit_decreases_with_distance(self):
        engine = CollectionEngine()
        r_close = _make_record(head_yaw=5.0, head_pitch=0.0)
        r_far = _make_record(head_yaw=14.0, head_pitch=0.0)
        fit_close = engine._compute_pose_fit(r_close, "frontal")
        fit_far = engine._compute_pose_fit(r_far, "frontal")
        assert fit_close > fit_far

    def test_pose_fit_other_pose(self):
        """'other' pose gets neutral 0.5 fit."""
        engine = CollectionEngine()
        r = _make_record(head_yaw=80.0, head_pitch=0.0)
        fit = engine._compute_pose_fit(r, "other")
        assert fit == 0.5


class TestCollectionEngineCatalogMode:
    def test_fallback_mode_no_catalog(self):
        """Without catalog_profiles, engine uses AU-rule fallback."""
        records = _make_similar_records(20, base_seed=42)
        for r in records:
            r.au12_lip_corner = 2.0  # smile trigger

        engine = CollectionEngine()
        result = engine.collect(records)

        if 0 in result.persons:
            person = result.persons[0]
            assert person.catalog_mode is False
            # All frames should have catalog_sim=1.0 in fallback
            for f in person.all_frames():
                assert f.catalog_sim == 1.0

    def test_catalog_mode_with_profiles(self):
        """With catalog_profiles, engine uses catalog-driven category."""
        from momentscan.algorithm.batch.catalog_scoring import CategoryProfile, SIGNAL_FIELDS

        _NDIM = len(SIGNAL_FIELDS)
        profiles = [
            CategoryProfile(
                name="warm_smile",
                mean_signals=np.full(_NDIM, 0.8),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
        ]

        records = _make_similar_records(20, base_seed=42)
        for r in records:
            r.catalog_best = 0.85
            r.catalog_primary = "warm_smile"

        engine = CollectionEngine(catalog_profiles=profiles)
        result = engine.collect(records)

        if 0 in result.persons:
            person = result.persons[0]
            assert person.catalog_mode is True


class TestCollectionEngineDedup:
    def test_temporal_dedup(self):
        """Consecutive frames should be deduplicated by time interval."""
        cfg = CollectionConfig(
            grid_top_k=5,
            grid_min_interval_ms=2000.0,
        )
        records = _make_similar_records(20, base_seed=42)
        # All 100ms apart → temporal dedup limits selection
        for i, r in enumerate(records):
            r.timestamp_ms = i * 100.0

        engine = CollectionEngine(config=cfg)
        result = engine.collect(records)

        if 0 in result.persons:
            # With 100ms spacing and 2000ms min interval, only 1 frame per cell
            for key, frames in result.persons[0].grid.items():
                assert len(frames) <= 1

    def test_well_spaced_selection(self):
        """Well-spaced frames should all be selected (up to top-k)."""
        cfg = CollectionConfig(
            grid_top_k=5,
            grid_min_interval_ms=1000.0,
        )
        records = _make_similar_records(20, base_seed=42)
        for i, r in enumerate(records):
            r.timestamp_ms = i * 2000.0  # 2s apart
            r.head_yaw = 0.0  # all same pose

        engine = CollectionEngine(config=cfg)
        result = engine.collect(records)

        if 0 in result.persons:
            total = len(result.persons[0].all_frames())
            assert total >= 2  # At least some frames selected


class TestCollectionEngineFromPath:
    def test_from_collection_path_none(self):
        engine = CollectionEngine.from_collection_path(None)
        assert len(engine._poses) == 4  # DEFAULT_POSES (no three-quarter-up)
        assert len(engine._pivots) == 6  # DEFAULT_PIVOTS (4 AU + 2 fallback)

    def test_from_collection_path_with_yaml(self, tmp_path):
        """collection_path with poses/ + categories/ → loads both."""
        import json

        coll_dir = tmp_path / "portrait-v1"
        poses_dir = coll_dir / "poses"
        poses_dir.mkdir(parents=True)
        (poses_dir / "frontal.yaml").write_text(
            "name: frontal\nyaw: 0\npitch: 0\nr_accept: 15\n"
        )

        # Need valid catalog profiles (fail-hard)
        from momentscan.algorithm.batch.catalog_scoring import SIGNAL_FIELDS
        _NDIM = len(SIGNAL_FIELDS)
        categories_dir = coll_dir / "categories"
        cat_dir = categories_dir / "warm_smile"
        cat_dir.mkdir(parents=True)
        (cat_dir / "_profile.json").write_text(json.dumps({
            "name": "warm_smile",
            "mean_signals": [0.5] * _NDIM,
            "importance_weights": [1.0 / _NDIM] * _NDIM,
            "n_refs": 5,
        }))

        engine = CollectionEngine.from_collection_path(str(coll_dir))
        assert len(engine._poses) == 1
        assert engine._poses[0].name == "frontal"
        assert len(engine._catalog_profiles) == 1


class TestCollectionEngineMedoid:
    def test_medoid_with_outlier(self):
        """Medoid should be in cluster, not outlier."""
        engine = CollectionEngine()

        base = _make_embedding(512, seed=100)
        records = []
        for i in range(4):
            r = _make_record(frame_idx=i, e_id_seed=100)
            r.e_id = base + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.01
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        outlier = _make_record(frame_idx=99, e_id_seed=999)
        records.append(outlier)

        prototype, proto_idx = engine._compute_medoid(records)
        assert proto_idx != 99

    def test_medoid_subsampling(self):
        cfg = CollectionConfig(medoid_max_candidates=10)
        engine = CollectionEngine(config=cfg)

        records = [_make_record(frame_idx=i, e_id_seed=i) for i in range(50)]
        prototype, proto_idx = engine._compute_medoid(records)
        assert prototype.shape == (512,)
