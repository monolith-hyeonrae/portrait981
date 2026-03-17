"""Tests for visualbind.analyzer (Day 0 analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from visualbind.analyzer import (
    analyze_distributions,
    compute_correlation_matrix,
    compute_neff,
    generate_report,
)


class TestCorrelationMatrix:
    def test_identity_for_uncorrelated(self):
        rng = np.random.RandomState(42)
        # Large enough sample for stable correlations
        vectors = rng.randn(1000, 5)
        corr = compute_correlation_matrix(vectors)
        assert corr.shape == (5, 5)
        # Diagonal should be ~1
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=0.01)
        # Off-diagonal should be ~0
        off_diag = corr[np.triu_indices(5, k=1)]
        assert np.abs(off_diag).max() < 0.1

    def test_single_sample(self):
        vectors = np.array([[1.0, 2.0, 3.0]])
        corr = compute_correlation_matrix(vectors)
        np.testing.assert_array_equal(corr, np.eye(3))

    def test_perfectly_correlated(self):
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        vectors = np.hstack([x, x * 2 + 1])
        corr = compute_correlation_matrix(vectors)
        assert corr[0, 1] == pytest.approx(1.0)


class TestNeff:
    def test_identity_matrix(self):
        """N_eff of identity = D."""
        d = 10
        neff = compute_neff(np.eye(d))
        assert neff == pytest.approx(d, abs=0.01)

    def test_fully_correlated(self):
        """N_eff of fully correlated matrix = 1."""
        d = 5
        corr = np.ones((d, d))
        neff = compute_neff(corr)
        assert neff == pytest.approx(1.0, abs=0.01)

    def test_partial_correlation(self):
        """N_eff between 1 and D."""
        d = 4
        corr = np.eye(d)
        corr[0, 1] = corr[1, 0] = 0.8  # pair correlated
        neff = compute_neff(corr)
        assert 1.0 < neff < d

    def test_zero_matrix(self):
        neff = compute_neff(np.zeros((3, 3)))
        assert neff == 0.0


class TestAnalyzeDistributions:
    def test_basic(self):
        rng = np.random.RandomState(42)
        vectors = rng.rand(100, 3)
        fields = ("a", "b", "c")
        result = analyze_distributions(vectors, fields)

        assert result["n_samples"] == 100
        assert result["n_dims"] == 3
        assert set(result["fields"].keys()) == {"a", "b", "c"}
        for stats in result["fields"].values():
            assert 0.0 <= stats["mean"] <= 1.0
            assert stats["std"] > 0

    def test_empty(self):
        vectors = np.zeros((0, 3))
        result = analyze_distributions(vectors, ("a", "b", "c"))
        assert result["n_samples"] == 0
        assert result["fields"] == {}

    def test_zero_fraction(self):
        vectors = np.zeros((10, 2))
        vectors[0, 0] = 1.0  # only 1 non-zero in dim 0
        result = analyze_distributions(vectors, ("x", "y"))
        assert result["fields"]["y"]["zero_frac"] == pytest.approx(1.0)
        assert result["fields"]["x"]["zero_frac"] == pytest.approx(0.9)


class TestGenerateReport:
    def test_basic_report(self):
        rng = np.random.RandomState(42)
        vectors = rng.rand(50, 5)
        fields = ("a", "b", "c", "d", "e")
        report = generate_report(vectors, fields)
        assert "N_eff" in report
        assert "Samples: 50" in report

    def test_insufficient_samples(self):
        vectors = np.array([[1.0, 2.0]])
        report = generate_report(vectors, ("a", "b"))
        assert "Insufficient" in report
