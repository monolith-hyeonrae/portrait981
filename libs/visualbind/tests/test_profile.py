"""Tests for visualbind.profile."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from visualbind.profile import CategoryProfile, load_profiles, save_profiles
from visualbind.signals import SIGNAL_FIELDS, _NDIM


@pytest.fixture
def sample_profiles():
    return [
        CategoryProfile(
            name="warm_smile",
            mean_signals=np.random.rand(_NDIM),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=5,
        ),
        CategoryProfile(
            name="cool_gaze",
            mean_signals=np.random.rand(_NDIM),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=3,
        ),
    ]


class TestCategoryProfile:
    def test_frozen(self):
        p = CategoryProfile(
            name="test",
            mean_signals=np.zeros(_NDIM),
            importance_weights=np.ones(_NDIM) / _NDIM,
            n_refs=1,
        )
        with pytest.raises(AttributeError):
            p.name = "changed"  # type: ignore[misc]


class TestSaveLoadProfiles:
    def test_roundtrip(self, tmp_path: Path, sample_profiles):
        save_profiles(tmp_path, sample_profiles)

        loaded = load_profiles(tmp_path)
        assert len(loaded) == 2
        assert loaded[0].name == "cool_gaze"  # sorted alphabetically
        assert loaded[1].name == "warm_smile"

        np.testing.assert_allclose(
            loaded[1].mean_signals,
            sample_profiles[0].mean_signals,
            atol=1e-10,
        )

    def test_load_missing_categories_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_profiles(tmp_path)

    def test_load_empty_categories(self, tmp_path: Path):
        (tmp_path / "categories").mkdir()
        with pytest.raises(ValueError, match="No valid"):
            load_profiles(tmp_path)

    def test_load_dimension_mismatch(self, tmp_path: Path):
        cat_dir = tmp_path / "categories" / "bad"
        cat_dir.mkdir(parents=True)
        data = {
            "name": "bad",
            "mean_signals": [0.0, 0.1],  # wrong dimension
            "importance_weights": [0.5, 0.5],
            "n_refs": 1,
        }
        (cat_dir / "_profile.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="signals"):
            load_profiles(tmp_path)
