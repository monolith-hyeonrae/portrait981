"""Observation dataclass for analyzer outputs."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Observation:
    """Observation output from an analyzer.

    Observations are timestamped feature analyses that flow from
    B modules (analyzers) to C module (fusion).

    For trigger modules, set trigger info in signals:
    - signals["should_trigger"]: Whether to fire a trigger
    - signals["trigger_score"]: Confidence score [0, 1]
    - signals["trigger_reason"]: Reason for the trigger
    - metadata["trigger"]: Trigger object

    Attributes:
        source: Name of the analyzer that produced this observation.
        frame_id: Frame identifier from the source video.
        t_ns: Timestamp in nanoseconds (source timeline).
        signals: Dictionary of extracted signals/features.
        data: Type-safe output data (e.g., PoseOutput, FaceClassifierOutput).
        metadata: Additional metadata about the observation.
        timing: Optional per-component timing in milliseconds.
    """

    source: str
    frame_id: int
    t_ns: int
    signals: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Any] = None  # Type-safe output (PoseOutput, FaceDetectOutput, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Dict[str, float]] = None  # {"detect_ms": 42.3, "expression_ms": 28.1}

    # Trigger helper properties (matches visualpath.core.Observation)

    @property
    def should_trigger(self) -> bool:
        """Check if this observation indicates a trigger should fire."""
        return bool(self.signals.get("should_trigger", False))

    @property
    def trigger_score(self) -> float:
        """Get the trigger confidence score."""
        return float(self.signals.get("trigger_score", 0.0))

    @property
    def trigger_reason(self) -> str:
        """Get the trigger reason."""
        return str(self.signals.get("trigger_reason", ""))

    @property
    def trigger(self) -> Optional[Any]:
        """Get the Trigger object if present."""
        return self.metadata.get("trigger")

    # Backwards-compatible aliases
    @property
    def score(self) -> float:
        """Alias for trigger_score."""
        return self.trigger_score

    @property
    def reason(self) -> str:
        """Alias for trigger_reason."""
        return self.trigger_reason
