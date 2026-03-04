"""Tests for signal-profile catalog scoring engine."""

import json
import pytest
import numpy as np
from pathlib import Path

from momentscan.algorithm.batch.types import FrameRecord
from momentscan.algorithm.batch.catalog_scoring import (
    SIGNAL_FIELDS,
    SIGNAL_RANGES,
    CategoryProfile,
    normalize_signal,
    extract_signal_vector,
    extract_signal_vector_from_dict,
    compute_importance_weights,
    match_category,
    compute_catalog_scores,
    load_profiles,
    save_profiles,
)

_NDIM = len(SIGNAL_FIELDS)


# ── normalize_signal ──

class TestNormalizeSignal:
    def test_clamp_min(self):
        assert normalize_signal(-1.0, "au6_cheek_raiser") == 0.0

    def test_clamp_max(self):
        assert normalize_signal(10.0, "au6_cheek_raiser") == 1.0

    def test_midpoint(self):
        assert normalize_signal(2.5, "au6_cheek_raiser") == pytest.approx(0.5)

    def test_zero_one_range(self):
        assert normalize_signal(0.5, "em_happy") == pytest.approx(0.5)

    def test_au_range(self):
        assert normalize_signal(2.5, "au12_lip_corner") == pytest.approx(0.5)


# ── extract_signal_vector ──

class TestExtractSignalVector:
    def test_default_record(self):
        r = FrameRecord(frame_idx=0, timestamp_ms=0.0)
        vec = extract_signal_vector(r)
        assert vec.shape == (_NDIM,)
        # au1_inner_brow=0 → normalized to 0.0
        assert vec[0] == pytest.approx(0.0)
        # em_happy=0 → normalized to 0
        em_happy_idx = list(SIGNAL_FIELDS).index("em_happy")
        assert vec[em_happy_idx] == pytest.approx(0.0)

    def test_with_signals(self):
        r = FrameRecord(
            frame_idx=0, timestamp_ms=0.0,
            au6_cheek_raiser=2.5,  # → 0.5
            em_happy=0.8,  # → 0.8
            clip_axes={"warm_smile": 0.6},  # → 0.6
        )
        vec = extract_signal_vector(r)
        au6_idx = list(SIGNAL_FIELDS).index("au6_cheek_raiser")
        em_happy_idx = list(SIGNAL_FIELDS).index("em_happy")
        clip_idx = list(SIGNAL_FIELDS).index("warm_smile")
        assert vec[au6_idx] == pytest.approx(0.5)
        assert vec[em_happy_idx] == pytest.approx(0.8)
        assert vec[clip_idx] == pytest.approx(0.6)


class TestExtractSignalVectorFromDict:
    def test_empty_dict(self):
        vec = extract_signal_vector_from_dict({})
        assert vec.shape == (_NDIM,)
        # All AU fields default to 0.0 (bottom of range)
        assert vec[0] == pytest.approx(0.0)

    def test_partial_dict(self):
        vec = extract_signal_vector_from_dict({
            "em_happy": 1.0,
            "au12_lip_corner": 5.0,
        })
        au12_idx = list(SIGNAL_FIELDS).index("au12_lip_corner")
        em_happy_idx = list(SIGNAL_FIELDS).index("em_happy")
        assert vec[au12_idx] == pytest.approx(1.0)  # 5/5 = 1.0
        assert vec[em_happy_idx] == pytest.approx(1.0)  # 1.0/1.0 = 1.0


# ── compute_importance_weights ──

class TestComputeImportanceWeights:
    def test_empty(self):
        assert compute_importance_weights({}) == {}

    def test_single_category(self):
        vecs = np.random.rand(5, _NDIM)
        result = compute_importance_weights({"cat": vecs})
        assert "cat" in result
        assert result["cat"].shape == (_NDIM,)
        assert result["cat"].sum() == pytest.approx(1.0)

    def test_two_distinct_categories(self):
        """Two categories differing mainly in em_happy → em_happy signals get higher weight."""
        n = 10
        em_happy_idx = list(SIGNAL_FIELDS).index("em_happy")
        cat_a = np.full((n, _NDIM), 0.5)
        cat_b = np.full((n, _NDIM), 0.5)
        # Category A: high em_happy
        cat_a[:, em_happy_idx] = 0.9
        # Category B: low em_happy
        cat_b[:, em_happy_idx] = 0.1
        # Add small noise
        rng = np.random.default_rng(42)
        cat_a += rng.normal(0, 0.01, cat_a.shape)
        cat_b += rng.normal(0, 0.01, cat_b.shape)

        weights = compute_importance_weights({"smile": cat_a, "neutral": cat_b})
        # em_happy should have highest weight for both
        for name, w in weights.items():
            assert w.sum() == pytest.approx(1.0)
            assert w[em_happy_idx] > 0.1  # em_happy has significant weight

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(123)
        cats = {
            "a": rng.random((5, _NDIM)),
            "b": rng.random((5, _NDIM)),
            "c": rng.random((5, _NDIM)),
        }
        weights = compute_importance_weights(cats)
        for name, w in weights.items():
            assert w.sum() == pytest.approx(1.0)


# ── match_category ──

class TestMatchCategory:
    def _make_profile(self, name, mean_vec, weights=None):
        if weights is None:
            weights = np.ones(_NDIM) / _NDIM
        return CategoryProfile(
            name=name,
            mean_signals=mean_vec,
            importance_weights=weights,
            n_refs=5,
        )

    def test_empty_profiles(self):
        vec = np.zeros(_NDIM)
        sim, name = match_category(vec, [])
        assert sim == 0.0
        assert name == ""

    def test_identical_vector(self):
        mean = np.full(_NDIM, 0.5)
        profile = self._make_profile("test", mean)
        sim, name = match_category(mean, [profile])
        assert sim == pytest.approx(1.0)  # distance=0 → sim=1/(1+0)=1.0
        assert name == "test"

    def test_selects_closest(self):
        frame = np.full(_NDIM, 0.8)
        p1 = self._make_profile("far", np.full(_NDIM, 0.2))
        p2 = self._make_profile("close", np.full(_NDIM, 0.7))
        sim, name = match_category(frame, [p1, p2])
        assert name == "close"

    def test_similarity_range(self):
        """Similarity should be in (0, 1]."""
        frame = np.random.rand(_NDIM)
        profiles = [
            self._make_profile("a", np.random.rand(_NDIM)),
            self._make_profile("b", np.random.rand(_NDIM)),
        ]
        sim, name = match_category(frame, profiles)
        assert 0.0 < sim <= 1.0

    def test_weighted_distance(self):
        """Higher weight on differing dimension → lower similarity."""
        frame = np.full(_NDIM, 0.5)
        mean = np.full(_NDIM, 0.5)
        mean[0] = 0.0  # dimension 0 differs

        # Uniform weights
        w_uniform = np.ones(_NDIM) / _NDIM
        p_uniform = self._make_profile("uniform", mean, w_uniform)

        # Heavy weight on dimension 0
        w_heavy = np.zeros(_NDIM)
        w_heavy[0] = 1.0
        p_heavy = self._make_profile("heavy", mean, w_heavy)

        sim_uniform, _ = match_category(frame, [p_uniform])
        sim_heavy, _ = match_category(frame, [p_heavy])

        # Heavy weighting on the differing dimension → lower similarity
        assert sim_heavy < sim_uniform


# ── compute_catalog_scores ──

class TestComputeCatalogScores:
    def test_no_profiles(self):
        r = FrameRecord(frame_idx=0, timestamp_ms=0.0)
        compute_catalog_scores(r, [])
        # No change
        assert r.catalog_best == 0.0
        assert r.catalog_primary == ""
        assert r.catalog_scores == {}

    def test_sets_catalog_fields(self):
        profile = CategoryProfile(
            name="warm_smile",
            mean_signals=np.full(_NDIM, 0.5),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        )
        r = FrameRecord(frame_idx=0, timestamp_ms=0.0)
        compute_catalog_scores(r, [profile])
        assert r.catalog_best > 0.0
        assert r.catalog_primary == "warm_smile"
        assert "warm_smile" in r.catalog_scores
        assert r.catalog_scores["warm_smile"] == r.catalog_best

    def test_overwrites_existing(self):
        profile = CategoryProfile(
            name="cool_gaze",
            mean_signals=np.full(_NDIM, 0.5),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        )
        r = FrameRecord(frame_idx=0, timestamp_ms=0.0)
        r.catalog_best = 0.99
        r.catalog_primary = "old_cat"
        compute_catalog_scores(r, [profile])
        assert r.catalog_primary == "cool_gaze"

    def test_catalog_scores_all_categories(self):
        """All category similarities are stored in catalog_scores dict."""
        profiles = [
            CategoryProfile(
                name="warm_smile",
                mean_signals=np.full(_NDIM, 0.3),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
            CategoryProfile(
                name="cool_gaze",
                mean_signals=np.full(_NDIM, 0.7),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
        ]
        r = FrameRecord(frame_idx=0, timestamp_ms=0.0)
        compute_catalog_scores(r, profiles)
        assert len(r.catalog_scores) == 2
        assert "warm_smile" in r.catalog_scores
        assert "cool_gaze" in r.catalog_scores
        assert r.catalog_scores[r.catalog_primary] == r.catalog_best
        for sim in r.catalog_scores.values():
            assert 0.0 < sim <= 1.0


# ── load_profiles / save_profiles ──

class TestLoadSaveProfiles:
    def test_roundtrip(self, tmp_path):
        profiles = [
            CategoryProfile(
                name="warm_smile",
                mean_signals=np.array([0.5] * _NDIM),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=7,
            ),
            CategoryProfile(
                name="cool_gaze",
                mean_signals=np.array([0.3] * _NDIM),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
        ]

        save_profiles(tmp_path, profiles)
        loaded = load_profiles(tmp_path)

        assert len(loaded) == 2
        assert loaded[0].name == "cool_gaze"  # sorted by dir name
        assert loaded[1].name == "warm_smile"
        np.testing.assert_array_almost_equal(
            loaded[1].mean_signals, profiles[0].mean_signals
        )
        np.testing.assert_array_almost_equal(
            loaded[1].importance_weights, profiles[0].importance_weights
        )
        assert loaded[1].n_refs == 7

    def test_missing_categories_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_profiles(tmp_path)

    def test_hidden_dirs_only_raises(self, tmp_path):
        """Only hidden dirs in categories/ → ValueError (no valid profiles)."""
        categories = tmp_path / "categories"
        categories.mkdir()
        # Hidden dir
        hidden = categories / "_cache"
        hidden.mkdir()
        (hidden / "_profile.json").write_text("{}")

        with pytest.raises(ValueError, match="No valid category profiles"):
            load_profiles(tmp_path)

    def test_invalid_dimension_raises(self, tmp_path):
        """Wrong signal dimension → ValueError."""
        categories = tmp_path / "categories"
        cat_dir = categories / "bad"
        cat_dir.mkdir(parents=True)
        (cat_dir / "_profile.json").write_text(json.dumps({
            "name": "bad",
            "mean_signals": [0.5, 0.5],  # wrong dim
            "importance_weights": [0.5, 0.5],
            "n_refs": 1,
        }))

        with pytest.raises(ValueError, match="has 2 signals"):
            load_profiles(tmp_path)

    def test_missing_profile_with_refs_raises(self, tmp_path):
        """Category dir with ref images but no _profile.json → FileNotFoundError."""
        categories = tmp_path / "categories"
        cat_dir = categories / "warm_smile"
        cat_dir.mkdir(parents=True)
        (cat_dir / "ref_001.jpg").write_bytes(b"\xff\xd8")  # has refs

        with pytest.raises(FileNotFoundError, match="Missing _profile.json"):
            load_profiles(tmp_path)

    def test_empty_category_skipped(self, tmp_path):
        """Category dir without refs or profile → skipped (not error)."""
        categories = tmp_path / "categories"
        # Empty placeholder category
        (categories / "cool_gaze").mkdir(parents=True)
        (categories / "cool_gaze" / "category.yaml").write_text("name: cool_gaze\n")
        # Valid category with profile
        cat_dir = categories / "warm_smile"
        cat_dir.mkdir(parents=True)
        (cat_dir / "_profile.json").write_text(json.dumps({
            "name": "warm_smile",
            "mean_signals": [0.5] * _NDIM,
            "importance_weights": [1.0 / _NDIM] * _NDIM,
            "n_refs": 5,
        }))

        profiles = load_profiles(tmp_path)
        assert len(profiles) == 1
        assert profiles[0].name == "warm_smile"

    def test_profile_json_has_signal_fields(self, tmp_path):
        """Saved profile includes signal_fields for documentation."""
        profiles = [
            CategoryProfile(
                name="test",
                mean_signals=np.zeros(_NDIM),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=3,
            ),
        ]
        save_profiles(tmp_path, profiles)

        profile_path = tmp_path / "categories" / "test" / "_profile.json"
        data = json.loads(profile_path.read_text())
        assert "signal_fields" in data
        assert data["signal_fields"] == list(SIGNAL_FIELDS)
