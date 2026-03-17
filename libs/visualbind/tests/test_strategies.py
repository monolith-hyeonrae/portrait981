"""Tests for visualbind.strategies (catalog + tree)."""

from __future__ import annotations

import numpy as np
import pytest

from visualbind.profile import CategoryProfile
from visualbind.signals import _NDIM
from visualbind.strategies import BindingStrategy
from visualbind.strategies.catalog import (
    CatalogStrategy,
    compute_importance_weights,
    match_category,
)


class TestComputeImportanceWeights:
    def test_empty(self):
        assert compute_importance_weights({}) == {}

    def test_single_category(self):
        vecs = np.random.rand(5, _NDIM)
        result = compute_importance_weights({"cat_a": vecs})
        assert "cat_a" in result
        assert result["cat_a"].shape == (_NDIM,)
        assert result["cat_a"].sum() == pytest.approx(1.0)

    def test_two_categories_weights_sum(self):
        rng = np.random.RandomState(42)
        vecs_a = rng.rand(10, _NDIM)
        vecs_b = rng.rand(10, _NDIM) + 1.0  # shifted
        result = compute_importance_weights({"a": vecs_a, "b": vecs_b})
        for w in result.values():
            assert w.sum() == pytest.approx(1.0)

    def test_discriminative_dimension_gets_higher_weight(self):
        """When one dimension clearly separates categories, it should get higher weight."""
        rng = np.random.RandomState(42)
        n = 50
        # All dims similar except dim 0
        vecs_a = rng.rand(n, _NDIM) * 0.1
        vecs_b = rng.rand(n, _NDIM) * 0.1
        vecs_a[:, 0] = 0.1  # dim 0: clearly different
        vecs_b[:, 0] = 0.9
        result = compute_importance_weights({"a": vecs_a, "b": vecs_b})
        # dim 0 should have highest weight
        assert result["a"][0] == result["a"].max()


class TestMatchCategory:
    def test_empty_profiles(self):
        sim, name = match_category(np.zeros(_NDIM), [])
        assert sim == 0.0
        assert name == ""

    def test_exact_match(self):
        centroid = np.random.rand(_NDIM)
        weights = np.ones(_NDIM) / _NDIM
        profile = CategoryProfile("test", centroid, weights, 5)
        sim, name = match_category(centroid, [profile])
        assert name == "test"
        assert sim == pytest.approx(1.0)  # distance=0 -> sim=1

    def test_closer_profile_wins(self):
        frame = np.zeros(_NDIM)
        near = CategoryProfile("near", np.full(_NDIM, 0.1), np.ones(_NDIM) / _NDIM, 3)
        far = CategoryProfile("far", np.full(_NDIM, 0.9), np.ones(_NDIM) / _NDIM, 3)
        sim, name = match_category(frame, [near, far])
        assert name == "near"


class TestCatalogStrategy:
    def test_implements_protocol(self):
        assert isinstance(CatalogStrategy(), BindingStrategy)

    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        vecs = {
            "smile": rng.rand(10, _NDIM),
            "neutral": rng.rand(10, _NDIM) + 0.5,
        }
        strategy = CatalogStrategy()
        strategy.fit(vecs)
        assert len(strategy.profiles) == 2

        scores = strategy.predict(vecs["smile"][0])
        assert "smile" in scores
        assert "neutral" in scores
        # First sample should be closer to its own category
        assert scores["smile"] > scores["neutral"]

    def test_predict_with_prebuilt_profiles(self):
        centroid = np.zeros(_NDIM)
        weights = np.ones(_NDIM) / _NDIM
        profiles = [CategoryProfile("zero", centroid, weights, 5)]
        strategy = CatalogStrategy(profiles=profiles)
        scores = strategy.predict(np.zeros(_NDIM))
        assert scores["zero"] == pytest.approx(1.0)

    def test_predict_empty(self):
        strategy = CatalogStrategy()
        scores = strategy.predict(np.zeros(_NDIM))
        assert scores == {}


class TestTreeStrategy:
    """Test TreeStrategy with logistic regression fallback."""

    def test_implements_protocol(self):
        from visualbind.strategies.tree import TreeStrategy
        assert isinstance(TreeStrategy(), BindingStrategy)

    def test_fit_and_predict_logistic(self):
        from visualbind.strategies.tree import TreeStrategy

        rng = np.random.RandomState(42)
        # Create clearly separable data
        vecs_a = rng.rand(30, _NDIM) * 0.3
        vecs_b = rng.rand(30, _NDIM) * 0.3 + 0.7
        strategy = TreeStrategy(use_xgboost=False)
        strategy.fit({"cat_a": vecs_a, "cat_b": vecs_b})
        assert len(strategy.classes) == 2

        scores = strategy.predict(vecs_a[0])
        assert "cat_a" in scores
        assert "cat_b" in scores
        assert scores["cat_a"] > scores["cat_b"]

    def test_predict_empty(self):
        from visualbind.strategies.tree import TreeStrategy

        strategy = TreeStrategy()
        assert strategy.predict(np.zeros(_NDIM)) == {}
