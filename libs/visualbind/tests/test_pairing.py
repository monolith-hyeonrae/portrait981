"""Tests for PairMiner."""

import pytest
import numpy as np

from visualbind import (
    AgreementEngine,
    AgreementResult,
    CrossCheck,
    HintCollector,
    HintFrame,
    PairMiner,
    SourceSpec,
)


def _make_frames_with_agreements(n_high=5, n_low=5, n_ambig=3):
    """Create HintFrames with known agreement scores."""
    collector = HintCollector({
        "a": SourceSpec(signals=("x", "y")),
    })

    frames = []
    agreements = []

    # High agreement frames
    for i in range(n_high):
        f = collector.collect_from_signals({"a": {"x": 0.8 + i * 0.01, "y": 0.9}}, frame_id=i)
        frames.append(f)
        agreements.append(AgreementResult(score=0.7 + i * 0.02, n_available=1, n_total=1))

    # Low agreement frames
    for i in range(n_low):
        f = collector.collect_from_signals({"a": {"x": 0.1 + i * 0.01, "y": 0.1}}, frame_id=n_high + i)
        frames.append(f)
        agreements.append(AgreementResult(score=0.05 + i * 0.01, n_available=1, n_total=1))

    # Ambiguous frames
    for i in range(n_ambig):
        f = collector.collect_from_signals({"a": {"x": 0.5, "y": 0.5}}, frame_id=n_high + n_low + i)
        frames.append(f)
        agreements.append(AgreementResult(score=0.35, n_available=1, n_total=1))

    return frames, agreements


class TestPairMiner:
    def test_basic_mining(self):
        frames, agreements = _make_frames_with_agreements()
        miner = PairMiner(
            positive_threshold=0.5,
            negative_threshold=0.2,
            max_pairs_per_anchor=2,
            seed=42,
        )
        result = miner.mine(frames, agreements)

        assert result.n_high == 5
        assert result.n_low == 5
        assert result.n_ambiguous == 3
        assert len(result.pairs) > 0

    def test_pair_structure(self):
        frames, agreements = _make_frames_with_agreements()
        miner = PairMiner(positive_threshold=0.5, negative_threshold=0.2, seed=42)
        result = miner.mine(frames, agreements)

        for pair in result.pairs:
            assert len(pair.anchor) == len(pair.positive) == len(pair.negative)
            assert pair.anchor_agreement >= 0.5
            assert pair.negative_agreement < 0.2

    def test_no_high_frames(self):
        frames, agreements = _make_frames_with_agreements(n_high=0)
        miner = PairMiner(positive_threshold=0.5, negative_threshold=0.2)
        result = miner.mine(frames, agreements)

        assert len(result.pairs) == 0
        assert result.n_high == 0

    def test_no_low_frames(self):
        frames, agreements = _make_frames_with_agreements(n_low=0)
        miner = PairMiner(positive_threshold=0.5, negative_threshold=0.2)
        result = miner.mine(frames, agreements)

        assert len(result.pairs) == 0
        assert result.n_low == 0

    def test_single_high_frame(self):
        frames, agreements = _make_frames_with_agreements(n_high=1, n_low=5)
        miner = PairMiner(positive_threshold=0.5, negative_threshold=0.2)
        result = miner.mine(frames, agreements)

        # Need at least 2 high frames for anchor-positive pair
        assert len(result.pairs) == 0

    def test_as_arrays(self):
        frames, agreements = _make_frames_with_agreements()
        miner = PairMiner(positive_threshold=0.5, negative_threshold=0.2, seed=42)
        result = miner.mine(frames, agreements)

        anchors, positives, negatives = result.as_arrays()
        n = len(result.pairs)
        assert anchors.shape[0] == n
        assert positives.shape[0] == n
        assert negatives.shape[0] == n
        assert anchors.shape[1] == positives.shape[1] == negatives.shape[1]

    def test_empty_as_arrays(self):
        miner = PairMiner()
        result = miner.mine([], [])
        anchors, positives, negatives = result.as_arrays()
        assert anchors.shape == (0, 0)

    def test_length_mismatch_raises(self):
        miner = PairMiner()
        with pytest.raises(ValueError, match="same length"):
            miner.mine([HintFrame()], [])

    def test_deterministic_with_seed(self):
        frames, agreements = _make_frames_with_agreements()
        miner1 = PairMiner(positive_threshold=0.5, negative_threshold=0.2, seed=123)
        miner2 = PairMiner(positive_threshold=0.5, negative_threshold=0.2, seed=123)

        r1 = miner1.mine(frames, agreements)
        r2 = miner2.mine(frames, agreements)

        assert len(r1.pairs) == len(r2.pairs)
        for p1, p2 in zip(r1.pairs, r2.pairs):
            assert p1.anchor == p2.anchor
            assert p1.negative == p2.negative
