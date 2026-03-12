"""HintCollector — declarative signal collection and normalization.

Replaces hardcoded per-module extraction functions with a declarative config:

    collector = HintCollector({
        "face.expression": SourceSpec(
            signals=("em_happy", "em_neutral", "em_surprise"),
            normalize="softmax",
        ),
        "face.au": SourceSpec(
            signals=("AU6", "AU12", "AU25", "AU26"),
            normalize="minmax",
            range=(0.0, 5.0),
        ),
    })

    hint_frame = collector.collect(observations, frame_id=42)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

from visualbind.types import HintFrame, HintVector, SourceSpec


def _normalize_none(values: Sequence[float], spec: SourceSpec) -> tuple[float, ...]:
    """No normalization — pass through."""
    return tuple(values)


def _normalize_minmax(values: Sequence[float], spec: SourceSpec) -> tuple[float, ...]:
    """Min-max normalization to [0, 1]."""
    lo, hi = spec.range  # type: ignore[misc]
    span = hi - lo
    if span <= 0:
        return tuple(0.0 for _ in values)
    return tuple(max(0.0, min(1.0, (v - lo) / span)) for v in values)


def _normalize_sigmoid(values: Sequence[float], spec: SourceSpec) -> tuple[float, ...]:
    """Sigmoid normalization centered at spec.center with spec.scale."""
    center = spec.center
    scale = spec.scale
    result = []
    for v in values:
        x = (v - center) * scale
        # Clamp to avoid overflow
        x = max(-500.0, min(500.0, x))
        result.append(1.0 / (1.0 + math.exp(-x)))
    return tuple(result)


def _normalize_softmax(values: Sequence[float], spec: SourceSpec) -> tuple[float, ...]:
    """Softmax normalization across the signal group."""
    if not values:
        return ()
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]  # numerically stable
    total = sum(exps)
    if total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(e / total for e in exps)


_NORMALIZERS = {
    "none": _normalize_none,
    "minmax": _normalize_minmax,
    "sigmoid": _normalize_sigmoid,
    "softmax": _normalize_softmax,
}


class HintCollector:
    """Collects and normalizes signals from multiple observers.

    Args:
        specs: Mapping of source name to SourceSpec.
            Can also accept raw dicts for convenience.
    """

    def __init__(self, specs: Dict[str, Union[SourceSpec, dict]]):
        self._specs: Dict[str, SourceSpec] = {}
        for source, spec in specs.items():
            if isinstance(spec, dict):
                signals = spec.get("signals", ())
                if isinstance(signals, list):
                    signals = tuple(signals)
                self._specs[source] = SourceSpec(
                    signals=signals,
                    normalize=spec.get("normalize", "none"),
                    range=spec.get("range"),
                    center=spec.get("center", 0.0),
                    scale=spec.get("scale", 1.0),
                )
            else:
                self._specs[source] = spec

    @property
    def sources(self) -> List[str]:
        """Registered source names."""
        return list(self._specs.keys())

    @property
    def specs(self) -> Dict[str, SourceSpec]:
        """Source specifications."""
        return dict(self._specs)

    def total_signals(self) -> int:
        """Total number of signals across all sources."""
        return sum(len(s.signals) for s in self._specs.values())

    def collect(
        self,
        observations: Sequence[Any],
        frame_id: int = 0,
    ) -> HintFrame:
        """Collect signals from observations and produce a HintFrame.

        Observations are matched by their ``source`` attribute.
        Each observation's ``signals`` dict is read for the configured signal names.

        Args:
            observations: Sequence of Observation-like objects.
                Each must have ``source: str`` and ``signals: Dict[str, float]``.
            frame_id: Frame identifier to attach.

        Returns:
            HintFrame with normalized HintVectors.
        """
        # Index observations by source
        obs_by_source: Dict[str, Any] = {}
        for obs in observations:
            source = getattr(obs, "source", None)
            if source and source in self._specs:
                obs_by_source[source] = obs

        frame = HintFrame(frame_id=frame_id)

        for source, spec in self._specs.items():
            obs = obs_by_source.get(source)
            if obs is None:
                continue

            signals = getattr(obs, "signals", None) or {}
            confidence = float(signals.get("confidence", 1.0))

            # Extract raw values in spec order
            raw_values = tuple(float(signals.get(name, 0.0)) for name in spec.signals)

            # Normalize
            normalizer = _NORMALIZERS.get(spec.normalize, _normalize_none)
            normalized = normalizer(raw_values, spec)

            frame.hints[source] = HintVector(
                source=source,
                names=spec.signals,
                values=normalized,
                raw_values=raw_values,
                confidence=confidence,
            )

        return frame

    def collect_from_signals(
        self,
        signals_by_source: Dict[str, Dict[str, float]],
        frame_id: int = 0,
    ) -> HintFrame:
        """Collect from pre-extracted signal dicts (no Observation objects needed).

        Args:
            signals_by_source: {source_name: {signal_name: value}}.
            frame_id: Frame identifier.

        Returns:
            HintFrame with normalized HintVectors.
        """
        frame = HintFrame(frame_id=frame_id)

        for source, spec in self._specs.items():
            signals = signals_by_source.get(source)
            if signals is None:
                continue

            raw_values = tuple(float(signals.get(name, 0.0)) for name in spec.signals)

            normalizer = _NORMALIZERS.get(spec.normalize, _normalize_none)
            normalized = normalizer(raw_values, spec)

            frame.hints[source] = HintVector(
                source=source,
                names=spec.signals,
                values=normalized,
                raw_values=raw_values,
                confidence=1.0,
            )

        return frame
