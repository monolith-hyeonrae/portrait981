"""Core data types for visualbind.

HintVector: normalized signal vector from a single observer.
HintFrame: collection of HintVectors from all observers for one frame.
CrossCheck: domain-knowledge rule linking two signals.
AgreementResult: consensus result from cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence


@dataclass(frozen=True)
class SourceSpec:
    """Specification for collecting signals from one observer.

    Attributes:
        signals: Signal names to collect from this source.
        normalize: Normalization strategy ("none", "minmax", "sigmoid", "softmax").
        range: (min, max) for minmax normalization. Required when normalize="minmax".
        center: Sigmoid center value. Default 0.0.
        scale: Sigmoid steepness. Default 1.0.
    """

    signals: tuple[str, ...]
    normalize: str = "none"
    range: Optional[tuple[float, float]] = None
    center: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        if self.normalize == "minmax" and self.range is None:
            raise ValueError("minmax normalization requires 'range' parameter")


@dataclass
class HintVector:
    """Normalized signal vector from a single observer.

    Attributes:
        source: Observer name (e.g. "face.expression").
        names: Signal names in order.
        values: Normalized signal values (same length as names).
        raw_values: Original pre-normalization values.
        confidence: Observer confidence (0-1). Default 1.0.
    """

    source: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    raw_values: tuple[float, ...]
    confidence: float = 1.0

    def __post_init__(self):
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names ({len(self.names)}) and values ({len(self.values)}) "
                f"must have the same length"
            )

    def get(self, name: str, default: float = 0.0) -> float:
        """Get a signal value by name."""
        try:
            idx = self.names.index(name)
            return self.values[idx]
        except ValueError:
            return default

    def get_raw(self, name: str, default: float = 0.0) -> float:
        """Get a raw (pre-normalization) signal value by name."""
        try:
            idx = self.names.index(name)
            return self.raw_values[idx]
        except ValueError:
            return default

    def as_dict(self) -> Dict[str, float]:
        """Return values as {name: value} dict."""
        return dict(zip(self.names, self.values))


@dataclass
class HintFrame:
    """Collection of HintVectors from all observers for one frame.

    Attributes:
        frame_id: Frame identifier.
        hints: HintVectors keyed by source name.
    """

    frame_id: int = 0
    hints: Dict[str, HintVector] = field(default_factory=dict)

    def get_signal(self, source: str, signal: str, default: float = 0.0) -> float:
        """Get a specific signal value from a specific source."""
        hint = self.hints.get(source)
        if hint is None:
            return default
        return hint.get(signal, default)

    @property
    def sources(self) -> List[str]:
        """Active source names."""
        return list(self.hints.keys())

    def flat_vector(self) -> tuple[float, ...]:
        """Concatenate all hint vectors into a single flat vector.

        Order is determined by sorted source names, then signal order within each.
        """
        values: list[float] = []
        for source in sorted(self.hints.keys()):
            values.extend(self.hints[source].values)
        return tuple(values)

    def flat_names(self) -> tuple[str, ...]:
        """Signal names corresponding to flat_vector()."""
        names: list[str] = []
        for source in sorted(self.hints.keys()):
            hint = self.hints[source]
            names.extend(f"{source}.{n}" for n in hint.names)
        return tuple(names)


@dataclass(frozen=True)
class CrossCheck:
    """Domain-knowledge rule linking two signals from different observers.

    Attributes:
        source_a: First observer (e.g. "face.expression").
        signal_a: Signal name from source_a (e.g. "em_happy").
        source_b: Second observer (e.g. "face.au").
        signal_b: Signal name from source_b (e.g. "AU12").
        relation: Expected correlation direction.
        weight: Importance weight for this check. Default 1.0.
        description: Human-readable description.
    """

    source_a: str
    signal_a: str
    source_b: str
    signal_b: str
    relation: Literal["positive", "negative"] = "positive"
    weight: float = 1.0
    description: str = ""


@dataclass
class CheckResult:
    """Result of a single CrossCheck evaluation.

    Attributes:
        check: The CrossCheck that was evaluated.
        value_a: Signal A value (normalized).
        value_b: Signal B value (normalized).
        agreement: Agreement score for this check (-1 to 1).
            Positive = signals agree with expected relation.
            Negative = signals disagree.
        available: Whether both signals were present.
    """

    check: CrossCheck
    value_a: float = 0.0
    value_b: float = 0.0
    agreement: float = 0.0
    available: bool = True


@dataclass
class AgreementResult:
    """Aggregate consensus result from cross-validation.

    Attributes:
        score: Overall agreement score (0 to 1).
            0 = complete disagreement, 1 = full consensus.
        details: Per-check results.
        n_available: Number of checks that had both signals present.
        n_total: Total number of checks.
    """

    score: float = 0.0
    details: List[CheckResult] = field(default_factory=list)
    n_available: int = 0
    n_total: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of checks that were evaluable."""
        return self.n_available / self.n_total if self.n_total > 0 else 0.0
