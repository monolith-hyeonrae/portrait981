"""Tests for visualbind.evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from visualbind.evaluator import compare_strategies
from visualbind.signals import _NDIM
from visualbind.strategies.catalog import CatalogStrategy


class TestCompareStrategies:
    def test_basic_comparison(self):
        rng = np.random.RandomState(42)
        # Create separable train data
        train = {
            "alpha": rng.rand(20, _NDIM) * 0.3,
            "beta": rng.rand(20, _NDIM) * 0.3 + 0.7,
        }

        # Fit strategies
        catalog = CatalogStrategy()
        catalog.fit(train)

        # Test data from same distributions
        test_vecs = np.vstack([
            rng.rand(10, _NDIM) * 0.3,
            rng.rand(10, _NDIM) * 0.3 + 0.7,
        ])
        test_labels = np.array(["alpha"] * 10 + ["beta"] * 10)

        results = compare_strategies(
            {"catalog": catalog},
            test_vecs,
            test_labels,
        )

        assert "catalog" in results
        entry = results["catalog"]
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert "per_class_accuracy" in entry
        assert "confusion_matrix" in entry
        assert entry["classes"] == ["alpha", "beta"]
