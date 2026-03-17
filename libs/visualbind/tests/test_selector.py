"""Tests for visualbind.selector."""

import numpy as np
import pytest

from visualbind.selector import select_frames, SelectionResult, SelectedFrame


class FakeStrategy:
    """Fake strategy that returns deterministic scores."""

    def __init__(self, categories):
        self._categories = categories

    def fit(self, vectors, **kwargs):
        pass

    def predict(self, frame_vec):
        # Score = sum of first N dimensions, different offset per category
        base = float(frame_vec.sum())
        return {cat: base + i * 0.1 for i, cat in enumerate(self._categories)}


class TestSelectFrames:
    def test_basic(self):
        vectors = np.array([
            [0.1, 0.2],
            [0.5, 0.6],
            [0.3, 0.4],
            [0.9, 0.8],
        ])
        strategy = FakeStrategy(["cat_a", "cat_b"])
        result = select_frames(vectors, strategy, top_k=2)

        assert result.frame_count == 4
        assert len(result.per_category) == 2
        assert len(result.per_category["cat_a"]) == 2
        assert len(result.per_category["cat_b"]) == 2

        # Top scores should be from frame 3 (highest sum)
        assert result.per_category["cat_a"][0].index == 3
        assert result.per_category["cat_b"][0].index == 3

    def test_gate_mask(self):
        vectors = np.array([
            [0.9, 0.9],  # high score but gated out
            [0.5, 0.5],  # passes gate
            [0.3, 0.3],  # passes gate
        ])
        gate = np.array([False, True, True])
        strategy = FakeStrategy(["cat_a"])
        result = select_frames(vectors, strategy, top_k=2, gate_mask=gate)

        # Frame 0 should be excluded
        indices = [f.index for f in result.per_category["cat_a"]]
        assert 0 not in indices
        assert 1 in indices

    def test_min_score(self):
        vectors = np.array([
            [0.01, 0.01],  # low score
            [0.5, 0.5],    # good score
        ])
        strategy = FakeStrategy(["cat_a"])
        result = select_frames(vectors, strategy, top_k=5, min_score=0.1)

        # Only frame 1 should pass min_score
        assert len(result.per_category["cat_a"]) == 1

    def test_empty(self):
        vectors = np.empty((0, 5))
        strategy = FakeStrategy(["cat_a"])
        result = select_frames(vectors, strategy)

        assert result.frame_count == 0
        assert len(result.frames) == 0

    def test_specific_categories(self):
        vectors = np.array([[0.5, 0.5]])
        strategy = FakeStrategy(["cat_a", "cat_b", "cat_c"])
        result = select_frames(vectors, strategy, categories=["cat_b"])

        assert "cat_b" in result.per_category
        assert "cat_a" not in result.per_category
        assert "cat_c" not in result.per_category

    def test_top_k_ordering(self):
        vectors = np.array([
            [0.1, 0.1],
            [0.3, 0.3],
            [0.5, 0.5],
            [0.2, 0.2],
        ])
        strategy = FakeStrategy(["cat_a"])
        result = select_frames(vectors, strategy, top_k=3)

        scores = [f.score for f in result.per_category["cat_a"]]
        assert scores == sorted(scores, reverse=True)
        assert len(scores) == 3
