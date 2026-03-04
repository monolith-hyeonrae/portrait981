"""Tests for collection/types.py."""

import numpy as np
import pytest

from momentscan.algorithm.collection.types import (
    CollectionConfig,
    CollectionRecord,
    CollectionResult,
    PersonCollection,
    SelectedFrame,
)


class TestCollectionRecord:
    def test_defaults(self):
        r = CollectionRecord(frame_idx=0, timestamp_ms=0.0)
        assert r.e_id is None
        assert r.face_detected is False
        assert r.gate_passed is True
        assert r.person_id == 0

    def test_with_embedding(self):
        emb = np.random.randn(512).astype(np.float32)
        r = CollectionRecord(frame_idx=1, timestamp_ms=100.0, e_id=emb)
        assert r.e_id is not None
        assert r.e_id.shape == (512,)

    def test_signal_fields(self):
        r = CollectionRecord(
            frame_idx=0, timestamp_ms=0.0,
            smile_intensity=0.8,
            au12_lip_corner=2.5,
            clip_axes={"warm_smile": 0.7},
        )
        assert r.smile_intensity == 0.8
        assert r.au12_lip_corner == 2.5
        assert r.clip_axes["warm_smile"] == 0.7


class TestCollectionConfig:
    def test_defaults(self):
        cfg = CollectionConfig()
        assert cfg.tau_id == 0.35
        assert cfg.grid_top_k == 2
        assert cfg.anchor_max_yaw == 15.0
        assert cfg.clip_pre_sec == 1.0
        assert cfg.clip_post_sec == 1.5
        assert cfg.grid_quality_alpha == 0.3

    def test_custom(self):
        cfg = CollectionConfig(tau_id=0.5, grid_top_k=3)
        assert cfg.tau_id == 0.5
        assert cfg.grid_top_k == 3


class TestSelectedFrame:
    def test_basic(self):
        f = SelectedFrame(
            frame_idx=42, timestamp_ms=4200.0,
            pose_name="frontal",
            pivot_name="smile",
            cell_key="frontal|smile",
            quality_score=0.9,
            cell_score=0.8,
            catalog_sim=0.95,
            pose_fit=0.85,
        )
        assert f.set_type == "grid"
        assert f.pose_name == "frontal"
        assert f.pivot_name == "smile"
        assert f.cell_key == "frontal|smile"

    def test_backward_compat_properties(self):
        f = SelectedFrame(
            frame_idx=0, timestamp_ms=0.0,
            cell_score=0.75,
        )
        assert f.impact_score == 0.75
        assert f.combined_score == 0.75


class TestPersonCollection:
    def _make_person(self) -> PersonCollection:
        grid = {
            "frontal|smile": [
                SelectedFrame(
                    frame_idx=0, timestamp_ms=0.0,
                    pose_name="frontal", pivot_name="smile",
                    cell_key="frontal|smile",
                    quality_score=0.9, cell_score=0.9,
                    catalog_sim=0.95, pose_fit=0.85,
                ),
            ],
            "three-quarter|neutral": [
                SelectedFrame(
                    frame_idx=10, timestamp_ms=1000.0,
                    pose_name="three-quarter", pivot_name="neutral",
                    cell_key="three-quarter|neutral",
                    quality_score=0.7, cell_score=0.7,
                    catalog_sim=0.8, pose_fit=0.9,
                ),
            ],
            "frontal|excited": [
                SelectedFrame(
                    frame_idx=20, timestamp_ms=2000.0,
                    pose_name="frontal", pivot_name="excited",
                    cell_key="frontal|excited",
                    quality_score=0.8, cell_score=0.8,
                    catalog_sim=0.88, pose_fit=0.92,
                ),
            ],
            "side-profile|surprised": [
                SelectedFrame(
                    frame_idx=30, timestamp_ms=3000.0,
                    pose_name="side-profile", pivot_name="surprised",
                    cell_key="side-profile|surprised",
                    quality_score=0.5, cell_score=0.5,
                    catalog_sim=0.7, pose_fit=0.6,
                ),
            ],
        }
        return PersonCollection(
            person_id=0,
            prototype_frame_idx=0,
            grid=grid,
            pose_coverage={"frontal": 2, "three-quarter": 1, "side-profile": 1},
            category_coverage={"smile": 1, "neutral": 1, "excited": 1, "surprised": 1},
        )

    def test_all_frames(self):
        person = self._make_person()
        assert len(person.all_frames()) == 4

    def test_backward_compat_anchor_frames(self):
        """anchor_frames returns all_frames for backward compat."""
        person = self._make_person()
        assert len(person.anchor_frames) == 4

    def test_backward_compat_coverage_challenge_empty(self):
        person = self._make_person()
        assert len(person.coverage_frames) == 0
        assert len(person.challenge_frames) == 0

    def test_backward_compat_pivot_coverage(self):
        person = self._make_person()
        assert person.pivot_coverage == person.category_coverage

    def test_query_by_pose(self):
        person = self._make_person()
        results = person.query(pose_name="frontal")
        assert len(results) == 2
        assert all(f.pose_name == "frontal" for f in results)

    def test_query_by_pivot(self):
        person = self._make_person()
        results = person.query(pivot_name="smile")
        assert len(results) == 1
        assert results[0].frame_idx == 0

    def test_query_top_k(self):
        person = self._make_person()
        results = person.query(top_k=2)
        assert len(results) == 2
        # Sorted by cell_score descending
        assert results[0].cell_score >= results[1].cell_score

    def test_query_no_match(self):
        person = self._make_person()
        results = person.query(pose_name="nonexistent")
        assert len(results) == 0


class TestPersonCollectionFindNearest:
    def _make_person(self) -> PersonCollection:
        grid = {
            "frontal|smile": [
                SelectedFrame(
                    frame_idx=0, timestamp_ms=0.0,
                    pose_name="frontal", pivot_name="smile",
                    cell_key="frontal|smile",
                    cell_score=0.9,
                ),
            ],
            "three-quarter|neutral": [
                SelectedFrame(
                    frame_idx=10, timestamp_ms=1000.0,
                    pose_name="three-quarter", pivot_name="neutral",
                    cell_key="three-quarter|neutral",
                    cell_score=0.7,
                ),
            ],
        }
        return PersonCollection(person_id=0, grid=grid)

    def test_exact_match(self):
        person = self._make_person()
        f = person.find_nearest("frontal", "smile")
        assert f is not None
        assert f.frame_idx == 0

    def test_same_pose_fallback(self):
        """No exact cell but same pose exists."""
        person = self._make_person()
        f = person.find_nearest("frontal", "excited")
        assert f is not None
        assert f.pose_name == "frontal"

    def test_same_category_fallback(self):
        """No exact cell, no same pose, but same category exists."""
        person = self._make_person()
        f = person.find_nearest("side-profile", "smile")
        assert f is not None
        assert f.pivot_name == "smile"

    def test_best_overall_fallback(self):
        """No matching pose or category → returns best overall."""
        person = self._make_person()
        f = person.find_nearest("looking-up", "wild")
        assert f is not None
        assert f.cell_score == 0.9  # best overall

    def test_empty_grid(self):
        person = PersonCollection(person_id=0, grid={})
        f = person.find_nearest("frontal", "smile")
        assert f is None


class TestCollectionResult:
    def test_empty(self):
        r = CollectionResult()
        assert r.frame_count == 0
        assert len(r.persons) == 0

    def test_with_config(self):
        cfg = CollectionConfig(tau_id=0.5)
        r = CollectionResult(config=cfg)
        assert r.config.tau_id == 0.5
