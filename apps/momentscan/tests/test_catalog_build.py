"""Tests for signal-profile catalog builder."""

import json
import pytest
import numpy as np
from pathlib import Path

from momentscan.algorithm.batch.catalog_scoring import (
    SIGNAL_FIELDS,
    CategoryProfile,
)
from momentscan.algorithm.batch.catalog_build import (
    build_profiles_from_signals,
    compute_separation_metrics,
    _discover_categories,
    print_separation_report,
)

_NDIM = len(SIGNAL_FIELDS)


def _create_test_catalog(
    tmp_path: Path,
    categories: dict[str, int],
) -> Path:
    """테스트 카탈로그 디렉토리 생성.

    Args:
        categories: 카테고리 이름 → 참조 이미지 수.

    Returns:
        카탈로그 루트 경로.
    """
    catalog = tmp_path / "test_catalog"
    cats_dir = catalog / "categories"

    for cat_name, n_images in categories.items():
        cat_dir = cats_dir / cat_name
        cat_dir.mkdir(parents=True)
        for i in range(n_images):
            # Create minimal JPEG (1x1 pixel)
            img_path = cat_dir / f"ref_{i:03d}.jpg"
            img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    return catalog


# ── build_profiles_from_signals ──

class TestBuildProfilesFromSignals:
    def test_empty_input(self):
        assert build_profiles_from_signals({}) == []

    def test_single_category(self):
        signals = {
            "warm_smile": [
                {"em_happy": 0.8, "au12_lip_corner": 3.0, "warm_smile": 0.7},
                {"em_happy": 0.9, "au12_lip_corner": 3.5, "warm_smile": 0.8},
            ]
        }
        profiles = build_profiles_from_signals(signals)
        assert len(profiles) == 1
        assert profiles[0].name == "warm_smile"
        assert profiles[0].n_refs == 2
        assert profiles[0].mean_signals.shape == (_NDIM,)
        assert profiles[0].importance_weights.shape == (_NDIM,)

    def test_two_categories(self):
        signals = {
            "warm_smile": [
                {"em_happy": 0.9, "au12_lip_corner": 4.0, "warm_smile": 0.8},
                {"em_happy": 0.8, "au12_lip_corner": 3.5, "warm_smile": 0.7},
            ],
            "cool_gaze": [
                {"em_neutral": 0.9, "cool_gaze": 0.8, "em_happy": 0.1},
                {"em_neutral": 0.85, "cool_gaze": 0.75, "em_happy": 0.05},
            ],
        }
        profiles = build_profiles_from_signals(signals)
        assert len(profiles) == 2
        names = {p.name for p in profiles}
        assert names == {"warm_smile", "cool_gaze"}

    def test_importance_weights_sum_to_one(self):
        signals = {
            "a": [{"em_happy": 0.9}] * 5,
            "b": [{"em_happy": 0.1}] * 5,
        }
        profiles = build_profiles_from_signals(signals)
        for p in profiles:
            assert p.importance_weights.sum() == pytest.approx(1.0)

    def test_mean_signals_computed_correctly(self):
        signals = {
            "test": [
                {"em_happy": 0.4},
                {"em_happy": 0.6},
            ]
        }
        profiles = build_profiles_from_signals(signals)
        # em_happy range [0,1] → normalized values 0.4 and 0.6
        # mean = 0.5
        em_happy_idx = list(SIGNAL_FIELDS).index("em_happy")
        assert profiles[0].mean_signals[em_happy_idx] == pytest.approx(0.5)

    def test_skip_empty_category(self):
        signals = {
            "valid": [{"em_happy": 0.5}],
            "empty": [],
        }
        profiles = build_profiles_from_signals(signals)
        assert len(profiles) == 1
        assert profiles[0].name == "valid"

    def test_discriminative_signals_get_higher_weight(self):
        """Categories that differ in one dimension → that dimension gets higher weight."""
        n = 20
        cat_a_sigs = [{"au12_lip_corner": 4.0} for _ in range(n)]
        cat_b_sigs = [{"au12_lip_corner": 0.5} for _ in range(n)]

        profiles = build_profiles_from_signals({"a": cat_a_sigs, "b": cat_b_sigs})
        au12_idx = list(SIGNAL_FIELDS).index("au12_lip_corner")

        for p in profiles:
            # au12 should be the most important signal
            assert au12_idx == np.argmax(p.importance_weights)


# ── _discover_categories ──

class TestDiscoverCategories:
    def test_discover(self, tmp_path):
        catalog = _create_test_catalog(tmp_path, {"a": 3, "b": 2})
        cats = _discover_categories(catalog)
        assert set(cats.keys()) == {"a", "b"}
        assert len(cats["a"]) == 3
        assert len(cats["b"]) == 2

    def test_no_categories_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _discover_categories(tmp_path / "nonexistent")

    def test_skip_hidden_dirs(self, tmp_path):
        catalog = _create_test_catalog(tmp_path, {"real": 2})
        hidden = catalog / "categories" / "_cache"
        hidden.mkdir()
        (hidden / "ref_001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        cats = _discover_categories(catalog)
        assert "_cache" not in cats
        assert "real" in cats

    def test_skip_empty_category(self, tmp_path):
        catalog = _create_test_catalog(tmp_path, {"has_images": 3})
        empty_dir = catalog / "categories" / "no_images"
        empty_dir.mkdir()

        cats = _discover_categories(catalog)
        assert "no_images" not in cats

    def test_category_yaml_name_override(self, tmp_path):
        catalog = _create_test_catalog(tmp_path, {"dir_name": 2})
        cat_yaml = catalog / "categories" / "dir_name" / "category.yaml"
        try:
            import yaml
            cat_yaml.write_text(yaml.dump({"name": "display_name"}))

            cats = _discover_categories(catalog)
            assert "display_name" in cats
            assert "dir_name" not in cats
        except ImportError:
            pytest.skip("pyyaml not installed")


# ── compute_separation_metrics ──

class TestSeparationMetrics:
    def test_empty(self):
        m = compute_separation_metrics([])
        assert m.n_categories == 0
        assert len(m.pairwise_distances) == 0
        assert len(m.warnings) == 0

    def test_single_profile(self):
        p = CategoryProfile(
            name="only",
            mean_signals=np.full(_NDIM, 0.5),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=1,
        )
        m = compute_separation_metrics([p])
        assert m.n_categories == 1
        assert len(m.pairwise_distances) == 0
        assert any("only" in w and "1 ref" in w for w in m.warnings)

    def test_two_distant_profiles(self):
        p1 = CategoryProfile(
            name="high",
            mean_signals=np.full(_NDIM, 0.9),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        )
        p2 = CategoryProfile(
            name="low",
            mean_signals=np.full(_NDIM, 0.1),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        )
        m = compute_separation_metrics([p1, p2])
        assert m.n_categories == 2
        assert len(m.pairwise_distances) == 1
        assert m.min_distance > 0.15
        assert m.mean_distance > 0
        assert m.min_pair == ("high", "low")

    def test_overlapping_profiles_warning(self):
        p1 = CategoryProfile(
            name="a",
            mean_signals=np.full(_NDIM, 0.5),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        )
        p2 = CategoryProfile(
            name="b",
            mean_signals=np.full(_NDIM, 0.5 + 0.01),  # nearly identical
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        )
        m = compute_separation_metrics([p1, p2])
        assert m.min_distance < 0.15
        assert any("too close" in w for w in m.warnings)

    def test_low_refs_warning(self):
        p1 = CategoryProfile(
            name="few",
            mean_signals=np.full(_NDIM, 0.8),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=1,
        )
        p2 = CategoryProfile(
            name="many",
            mean_signals=np.full(_NDIM, 0.2),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=10,
        )
        m = compute_separation_metrics([p1, p2])
        assert any("few" in w and "1 ref" in w for w in m.warnings)

    def test_three_categories(self):
        profiles = [
            CategoryProfile(
                name=f"cat{i}",
                mean_signals=np.full(_NDIM, i * 0.3),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            )
            for i in range(3)
        ]
        m = compute_separation_metrics(profiles)
        assert m.n_categories == 3
        # C(3,2) = 3 pairs
        assert len(m.pairwise_distances) == 3


# ── print_separation_report ──

class TestSeparationReport:
    def test_empty(self):
        report = print_separation_report([])
        assert "No profiles" in report

    def test_basic_report(self):
        profiles = [
            CategoryProfile(
                name="warm",
                mean_signals=np.full(_NDIM, 0.8),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
            CategoryProfile(
                name="cool",
                mean_signals=np.full(_NDIM, 0.2),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
        ]
        report = print_separation_report(profiles)
        assert "warm" in report
        assert "cool" in report
        assert "distance" in report
        assert "Discriminative Signals" in report

    def test_single_profile(self):
        profiles = [
            CategoryProfile(
                name="only",
                mean_signals=np.full(_NDIM, 0.5),
                importance_weights=np.ones(_NDIM) / _NDIM,
                n_refs=5,
            ),
        ]
        report = print_separation_report(profiles)
        assert "only" in report
        # No pairwise distance for single profile
        assert "Min distance" not in report


# ── Integration: signal-profile scoring roundtrip ──

class TestScoringRoundtrip:
    """build_profiles_from_signals → match_category → compute_catalog_scores."""

    def test_warm_smile_matches_warm_frame(self):
        from momentscan.algorithm.batch.catalog_scoring import (
            extract_signal_vector,
            match_category,
        )
        from momentscan.algorithm.batch.types import FrameRecord

        signals = {
            "warm_smile": [
                {"em_happy": 0.9, "au12_lip_corner": 4.0, "au6_cheek_raiser": 3.0,
                 "warm_smile": 0.8, "em_neutral": 0.1},
            ] * 5,
            "cool_gaze": [
                {"em_happy": 0.05, "em_neutral": 0.9,
                 "cool_gaze": 0.8, "warm_smile": 0.1},
            ] * 5,
        }
        profiles = build_profiles_from_signals(signals)

        # Frame with warm smile signals
        frame = FrameRecord(
            frame_idx=0, timestamp_ms=0.0,
            smile_intensity=0.85,
            em_happy=0.85,
            au12_lip_corner=3.8,
            au6_cheek_raiser=2.8,
            clip_axes={"warm_smile": 0.75},
            em_neutral=0.15,
        )
        vec = extract_signal_vector(frame)
        sim, cat = match_category(vec, profiles)
        assert cat == "warm_smile"

    def test_cool_gaze_matches_cool_frame(self):
        from momentscan.algorithm.batch.catalog_scoring import (
            extract_signal_vector,
            match_category,
        )
        from momentscan.algorithm.batch.types import FrameRecord

        signals = {
            "warm_smile": [
                {"em_happy": 0.9, "au12_lip_corner": 4.0,
                 "warm_smile": 0.8, "em_neutral": 0.1},
            ] * 5,
            "cool_gaze": [
                {"em_happy": 0.05, "em_neutral": 0.9,
                 "cool_gaze": 0.8, "warm_smile": 0.1},
            ] * 5,
        }
        profiles = build_profiles_from_signals(signals)

        # Frame with cool gaze signals
        frame = FrameRecord(
            frame_idx=0, timestamp_ms=0.0,
            smile_intensity=0.1,
            em_happy=0.1,
            em_neutral=0.85,
            clip_axes={"cool_gaze": 0.75, "warm_smile": 0.15},
        )
        vec = extract_signal_vector(frame)
        sim, cat = match_category(vec, profiles)
        assert cat == "cool_gaze"
