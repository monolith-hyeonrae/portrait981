"""Tests for HintCollector."""

import math
from dataclasses import dataclass, field
from typing import Dict

import pytest

from visualbind import HintCollector, SourceSpec


@dataclass
class FakeObservation:
    source: str
    signals: Dict[str, float] = field(default_factory=dict)


class TestHintCollectorBasic:
    def test_empty_observations(self):
        collector = HintCollector({
            "face.expression": SourceSpec(signals=("em_happy",)),
        })
        frame = collector.collect([], frame_id=1)
        assert frame.frame_id == 1
        assert len(frame.hints) == 0

    def test_collect_single_source(self):
        collector = HintCollector({
            "face.expression": SourceSpec(signals=("em_happy", "em_neutral")),
        })
        obs = FakeObservation(
            source="face.expression",
            signals={"em_happy": 0.8, "em_neutral": 0.15},
        )
        frame = collector.collect([obs], frame_id=5)

        assert "face.expression" in frame.hints
        hint = frame.hints["face.expression"]
        assert hint.source == "face.expression"
        assert hint.names == ("em_happy", "em_neutral")
        assert hint.raw_values == (0.8, 0.15)
        assert hint.values == (0.8, 0.15)  # no normalization

    def test_missing_signal_defaults_to_zero(self):
        collector = HintCollector({
            "face.expression": SourceSpec(signals=("em_happy", "em_sad")),
        })
        obs = FakeObservation(
            source="face.expression",
            signals={"em_happy": 0.7},
        )
        frame = collector.collect([obs])

        hint = frame.hints["face.expression"]
        assert hint.raw_values == (0.7, 0.0)

    def test_unregistered_source_ignored(self):
        collector = HintCollector({
            "face.expression": SourceSpec(signals=("em_happy",)),
        })
        obs = FakeObservation(source="body.pose", signals={"raised": 1.0})
        frame = collector.collect([obs])

        assert "body.pose" not in frame.hints
        assert len(frame.hints) == 0

    def test_multiple_sources(self):
        collector = HintCollector({
            "face.expression": SourceSpec(signals=("em_happy",)),
            "face.au": SourceSpec(signals=("AU12",)),
        })
        obs_expr = FakeObservation("face.expression", {"em_happy": 0.9})
        obs_au = FakeObservation("face.au", {"AU12": 3.5})
        frame = collector.collect([obs_expr, obs_au])

        assert len(frame.hints) == 2
        assert frame.get_signal("face.expression", "em_happy") == 0.9
        assert frame.get_signal("face.au", "AU12") == 3.5

    def test_sources_property(self):
        collector = HintCollector({
            "a": SourceSpec(signals=("x",)),
            "b": SourceSpec(signals=("y",)),
        })
        assert sorted(collector.sources) == ["a", "b"]

    def test_total_signals(self):
        collector = HintCollector({
            "a": SourceSpec(signals=("x", "y")),
            "b": SourceSpec(signals=("z",)),
        })
        assert collector.total_signals() == 3


class TestHintCollectorDict:
    """Test dict-based spec construction."""

    def test_dict_spec(self):
        collector = HintCollector({
            "face.expression": {
                "signals": ["em_happy", "em_neutral"],
                "normalize": "softmax",
            },
        })
        assert collector.specs["face.expression"].normalize == "softmax"
        assert collector.specs["face.expression"].signals == ("em_happy", "em_neutral")


class TestNormalizationNone:
    def test_passthrough(self):
        collector = HintCollector({
            "src": SourceSpec(signals=("a", "b"), normalize="none"),
        })
        obs = FakeObservation("src", {"a": 42.0, "b": -3.0})
        frame = collector.collect([obs])

        hint = frame.hints["src"]
        assert hint.values == (42.0, -3.0)
        assert hint.raw_values == (42.0, -3.0)


class TestNormalizationMinmax:
    def test_basic(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("yaw",),
                normalize="minmax",
                range=(-90.0, 90.0),
            ),
        })
        frame = collector.collect([FakeObservation("src", {"yaw": 0.0})])
        assert abs(frame.hints["src"].values[0] - 0.5) < 1e-8

    def test_clamped(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("x",),
                normalize="minmax",
                range=(0.0, 1.0),
            ),
        })
        # Above range
        frame = collector.collect([FakeObservation("src", {"x": 2.0})])
        assert frame.hints["src"].values[0] == 1.0

        # Below range
        frame = collector.collect([FakeObservation("src", {"x": -1.0})])
        assert frame.hints["src"].values[0] == 0.0

    def test_range_required(self):
        with pytest.raises(ValueError, match="range"):
            SourceSpec(signals=("x",), normalize="minmax")


class TestNormalizationSigmoid:
    def test_center_maps_to_half(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("x",),
                normalize="sigmoid",
                center=0.0,
                scale=1.0,
            ),
        })
        frame = collector.collect([FakeObservation("src", {"x": 0.0})])
        assert abs(frame.hints["src"].values[0] - 0.5) < 1e-8

    def test_high_value_approaches_one(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("x",),
                normalize="sigmoid",
                center=0.0,
                scale=1.0,
            ),
        })
        frame = collector.collect([FakeObservation("src", {"x": 10.0})])
        assert frame.hints["src"].values[0] > 0.99

    def test_custom_center(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("x",),
                normalize="sigmoid",
                center=5.0,
                scale=1.0,
            ),
        })
        frame = collector.collect([FakeObservation("src", {"x": 5.0})])
        assert abs(frame.hints["src"].values[0] - 0.5) < 1e-8


class TestNormalizationSoftmax:
    def test_sums_to_one(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("a", "b", "c"),
                normalize="softmax",
            ),
        })
        obs = FakeObservation("src", {"a": 1.0, "b": 2.0, "c": 3.0})
        frame = collector.collect([obs])

        values = frame.hints["src"].values
        assert abs(sum(values) - 1.0) < 1e-8

    def test_highest_gets_largest_share(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("a", "b"),
                normalize="softmax",
            ),
        })
        obs = FakeObservation("src", {"a": 0.1, "b": 5.0})
        frame = collector.collect([obs])

        values = frame.hints["src"].values
        assert values[1] > values[0]

    def test_equal_values(self):
        collector = HintCollector({
            "src": SourceSpec(
                signals=("a", "b"),
                normalize="softmax",
            ),
        })
        obs = FakeObservation("src", {"a": 1.0, "b": 1.0})
        frame = collector.collect([obs])

        values = frame.hints["src"].values
        assert abs(values[0] - 0.5) < 1e-8
        assert abs(values[1] - 0.5) < 1e-8


class TestCollectFromSignals:
    def test_basic(self):
        collector = HintCollector({
            "face.au": SourceSpec(
                signals=("AU6", "AU12"),
                normalize="minmax",
                range=(0.0, 5.0),
            ),
        })
        frame = collector.collect_from_signals({
            "face.au": {"AU6": 2.5, "AU12": 3.0},
        })

        assert "face.au" in frame.hints
        assert abs(frame.hints["face.au"].values[0] - 0.5) < 1e-8
        assert abs(frame.hints["face.au"].values[1] - 0.6) < 1e-8


class TestHintFrame:
    def test_flat_vector(self):
        collector = HintCollector({
            "a": SourceSpec(signals=("x",)),
            "b": SourceSpec(signals=("y", "z")),
        })
        frame = collector.collect_from_signals({
            "a": {"x": 1.0},
            "b": {"y": 2.0, "z": 3.0},
        })
        # Sorted by source name: a.x, b.y, b.z
        assert frame.flat_vector() == (1.0, 2.0, 3.0)
        assert frame.flat_names() == ("a.x", "b.y", "b.z")

    def test_get_signal_missing(self):
        frame = HintCollector({"a": SourceSpec(signals=("x",))}).collect([])
        assert frame.get_signal("a", "x") == 0.0
        assert frame.get_signal("missing", "y", default=-1.0) == -1.0


class TestHintVector:
    def test_as_dict(self):
        from visualbind.types import HintVector

        hv = HintVector(
            source="test",
            names=("a", "b"),
            values=(0.5, 0.8),
            raw_values=(1.0, 2.0),
        )
        assert hv.as_dict() == {"a": 0.5, "b": 0.8}

    def test_length_mismatch_raises(self):
        from visualbind.types import HintVector

        with pytest.raises(ValueError, match="same length"):
            HintVector(
                source="test",
                names=("a", "b"),
                values=(0.5,),
                raw_values=(1.0,),
            )
