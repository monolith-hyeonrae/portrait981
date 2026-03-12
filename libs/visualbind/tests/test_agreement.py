"""Tests for AgreementEngine."""

from dataclasses import dataclass, field
from typing import Dict

import pytest

from visualbind import (
    AgreementEngine,
    CrossCheck,
    HintCollector,
    HintFrame,
    HintVector,
    SourceSpec,
)


def _make_frame(signals_by_source: Dict[str, Dict[str, float]]) -> HintFrame:
    """Create a HintFrame from raw signal dicts (no normalization)."""
    collector = HintCollector({
        source: SourceSpec(signals=tuple(signals.keys()))
        for source, signals in signals_by_source.items()
    })
    return collector.collect_from_signals(signals_by_source)


class TestPositiveAgreement:
    def test_both_high_agrees(self):
        """Both signals high → high agreement."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="positive"),
        ])
        frame = _make_frame({"a": {"x": 0.9}, "b": {"y": 0.8}})
        result = engine.compute(frame)

        assert result.score > 0.5
        assert result.n_available == 1

    def test_both_low_disagrees(self):
        """Both signals low → low agreement (product is low)."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="positive"),
        ])
        frame = _make_frame({"a": {"x": 0.1}, "b": {"y": 0.1}})
        result = engine.compute(frame)

        assert result.score < 0.1

    def test_one_high_one_low_disagrees(self):
        """One high, one low → disagreement."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="positive"),
        ])
        frame = _make_frame({"a": {"x": 0.9}, "b": {"y": 0.1}})
        result = engine.compute(frame)

        assert result.score < 0.2

    def test_perfect_agreement(self):
        """Both at 1.0 → maximum agreement."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="positive"),
        ])
        frame = _make_frame({"a": {"x": 1.0}, "b": {"y": 1.0}})
        result = engine.compute(frame)

        assert abs(result.score - 1.0) < 1e-8


class TestNegativeAgreement:
    def test_a_high_b_low_agrees(self):
        """A high + B low → agreement for negative relation."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="negative"),
        ])
        frame = _make_frame({"a": {"x": 0.9}, "b": {"y": 0.1}})
        result = engine.compute(frame)

        assert result.score > 0.5

    def test_both_high_disagrees(self):
        """Both high → disagreement for negative relation."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="negative"),
        ])
        frame = _make_frame({"a": {"x": 0.9}, "b": {"y": 0.9}})
        result = engine.compute(frame)

        assert result.score < 0.2


class TestMissingSources:
    def test_missing_source_not_available(self):
        """Missing source → check marked as unavailable."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y"),
        ])
        frame = _make_frame({"a": {"x": 0.5}})  # b missing
        result = engine.compute(frame)

        assert result.n_available == 0
        assert result.n_total == 1
        assert result.score == 0.0
        assert not result.details[0].available

    def test_coverage(self):
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y"),
            CrossCheck("a", "x", "c", "z"),
        ])
        frame = _make_frame({"a": {"x": 0.5}, "b": {"y": 0.5}})
        result = engine.compute(frame)

        assert result.n_available == 1
        assert result.n_total == 2
        assert abs(result.coverage - 0.5) < 1e-8


class TestMultipleChecks:
    def test_weighted_average(self):
        """Multiple checks → weighted average of agreements."""
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", relation="positive", weight=2.0),
            CrossCheck("a", "x", "c", "z", relation="positive", weight=1.0),
        ])
        frame = _make_frame({
            "a": {"x": 1.0},
            "b": {"y": 1.0},  # perfect agreement
            "c": {"z": 0.0},  # zero agreement
        })
        result = engine.compute(frame)

        # Weighted: (2.0 * 1.0 + 1.0 * 0.0) / (2.0 + 1.0) = 2/3
        assert abs(result.score - 2.0 / 3.0) < 1e-8

    def test_all_agree(self):
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y"),
            CrossCheck("a", "x", "c", "z"),
        ])
        frame = _make_frame({
            "a": {"x": 0.8},
            "b": {"y": 0.8},
            "c": {"z": 0.8},
        })
        result = engine.compute(frame)

        assert result.score > 0.5
        assert result.n_available == 2


class TestConfidenceWeighting:
    def test_low_confidence_reduces_weight(self):
        """Low observer confidence reduces the effective weight of a check."""
        frame = HintFrame(frame_id=1)
        frame.hints["a"] = HintVector(
            source="a", names=("x",), values=(1.0,), raw_values=(1.0,),
            confidence=0.5,
        )
        frame.hints["b"] = HintVector(
            source="b", names=("y",), values=(1.0,), raw_values=(1.0,),
            confidence=1.0,
        )
        frame.hints["c"] = HintVector(
            source="c", names=("z",), values=(1.0,), raw_values=(1.0,),
            confidence=1.0,
        )

        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", weight=1.0),  # confidence=min(0.5,1.0)=0.5
            CrossCheck("b", "y", "c", "z", weight=1.0),  # confidence=min(1.0,1.0)=1.0
        ])
        result = engine.compute(frame)

        # Both checks have agreement=1.0
        # Effective weights: 1.0*0.5=0.5 and 1.0*1.0=1.0
        # Score: (0.5*1.0 + 1.0*1.0) / (0.5 + 1.0) = 1.5/1.5 = 1.0
        assert abs(result.score - 1.0) < 1e-8

    def test_without_confidence_weighting(self):
        """Without confidence weighting, all checks have equal effective weight."""
        frame = HintFrame(frame_id=1)
        frame.hints["a"] = HintVector(
            source="a", names=("x",), values=(1.0,), raw_values=(1.0,),
            confidence=0.1,  # very low
        )
        frame.hints["b"] = HintVector(
            source="b", names=("y",), values=(1.0,), raw_values=(1.0,),
            confidence=1.0,
        )

        engine = AgreementEngine(
            [CrossCheck("a", "x", "b", "y")],
            confidence_weighted=False,
        )
        result = engine.compute(frame)

        # Without confidence weighting, both agree at 1.0
        assert abs(result.score - 1.0) < 1e-8


class TestCheckDetails:
    def test_details_populated(self):
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y", description="test check"),
        ])
        frame = _make_frame({"a": {"x": 0.7}, "b": {"y": 0.6}})
        result = engine.compute(frame)

        assert len(result.details) == 1
        detail = result.details[0]
        assert detail.available is True
        assert abs(detail.value_a - 0.7) < 1e-8
        assert abs(detail.value_b - 0.6) < 1e-8
        assert detail.check.description == "test check"


class TestEdgeCases:
    def test_no_checks(self):
        engine = AgreementEngine([])
        frame = _make_frame({"a": {"x": 1.0}})
        result = engine.compute(frame)

        assert result.score == 0.0
        assert result.n_total == 0

    def test_empty_frame(self):
        engine = AgreementEngine([
            CrossCheck("a", "x", "b", "y"),
        ])
        frame = HintFrame()
        result = engine.compute(frame)

        assert result.score == 0.0
        assert result.n_available == 0


class TestPortrait981Scenario:
    """Integration test: portrait981 cross-validation scenario."""

    def test_genuine_smile_high_agreement(self):
        """Genuine Duchenne smile: expression happy + AU6+AU12 active."""
        collector = HintCollector({
            "face.expression": SourceSpec(
                signals=("em_happy", "em_neutral", "em_surprise"),
            ),
            "face.au": SourceSpec(
                signals=("AU6", "AU12", "AU25"),
                normalize="minmax",
                range=(0.0, 5.0),
            ),
        })

        engine = AgreementEngine([
            CrossCheck("face.expression", "em_happy", "face.au", "AU12",
                       relation="positive", description="smile → lip corner"),
            CrossCheck("face.expression", "em_happy", "face.au", "AU6",
                       relation="positive", description="smile → cheek raise"),
        ])

        # Genuine smile: high happy + high AU6/AU12
        frame = collector.collect_from_signals({
            "face.expression": {"em_happy": 0.85, "em_neutral": 0.1, "em_surprise": 0.05},
            "face.au": {"AU6": 3.5, "AU12": 4.0, "AU25": 0.5},
        })
        result = engine.compute(frame)

        assert result.score > 0.4  # strong agreement
        assert result.n_available == 2

    def test_fake_smile_low_agreement(self):
        """Fake smile: expression says happy but no AU6 (Duchenne marker)."""
        collector = HintCollector({
            "face.expression": SourceSpec(
                signals=("em_happy", "em_neutral"),
            ),
            "face.au": SourceSpec(
                signals=("AU6", "AU12"),
                normalize="minmax",
                range=(0.0, 5.0),
            ),
        })

        engine = AgreementEngine([
            CrossCheck("face.expression", "em_happy", "face.au", "AU6",
                       relation="positive", description="smile → cheek raise"),
        ])

        # Fake smile: happy expression but AU6 (cheek raiser) inactive
        frame = collector.collect_from_signals({
            "face.expression": {"em_happy": 0.8, "em_neutral": 0.15},
            "face.au": {"AU6": 0.2, "AU12": 2.5},
        })
        result = engine.compute(frame)

        # Low agreement: AU6 is very low despite high happy
        assert result.score < 0.15

    def test_neutral_face_consistent(self):
        """Neutral face: all signals low → consistent but not 'exciting'."""
        collector = HintCollector({
            "face.expression": SourceSpec(signals=("em_happy",)),
            "face.au": SourceSpec(
                signals=("AU6", "AU12"),
                normalize="minmax",
                range=(0.0, 5.0),
            ),
        })

        engine = AgreementEngine([
            CrossCheck("face.expression", "em_happy", "face.au", "AU12",
                       relation="positive"),
        ])

        frame = collector.collect_from_signals({
            "face.expression": {"em_happy": 0.05},
            "face.au": {"AU6": 0.1, "AU12": 0.2},
        })
        result = engine.compute(frame)

        # Both low → product is very low
        assert result.score < 0.01
