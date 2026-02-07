"""Trigger message definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

from visualbase.core.timestamp import seconds_to_ns, ns_to_seconds


class TriggerType(Enum):
    """Type of trigger event."""

    POINT = "POINT"  # Single moment event
    RANGE = "RANGE"  # Time range event


@dataclass(frozen=True, slots=True)
class Trigger:
    """Trigger message for clip extraction.

    Attributes:
        type: POINT for single moment, RANGE for time span.
        event_time_ns: Event timestamp in nanoseconds (required for POINT).
        start_time_ns: Start timestamp in nanoseconds (required for RANGE).
        end_time_ns: End timestamp in nanoseconds (required for RANGE).
        pre_sec: Seconds to include before the event.
        post_sec: Seconds to include after the event.
        label: Optional label for the clip.
        score: Optional score (0.0 to 1.0).
        metadata: Optional additional metadata.

    Examples:
        # Point event: capture 3s before and 2s after event at t=10s
        >>> trig = Trigger.point(event_time_ns=10_000_000_000, pre_sec=3.0, post_sec=2.0)
        >>> trig.clip_start_ns  # 7s
        7000000000
        >>> trig.clip_end_ns    # 12s
        12000000000

        # Range event: capture around a 5s highlight window
        >>> trig = Trigger.range(
        ...     start_time_ns=10_000_000_000,
        ...     end_time_ns=15_000_000_000,
        ...     pre_sec=3.0,
        ...     post_sec=1.0
        ... )
        >>> trig.clip_start_ns  # 7s
        7000000000
        >>> trig.clip_end_ns    # 16s
        16000000000
    """

    type: TriggerType
    event_time_ns: Optional[int] = None
    start_time_ns: Optional[int] = None
    end_time_ns: Optional[int] = None
    pre_sec: float = 3.0
    post_sec: float = 2.0
    label: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trigger parameters."""
        if self.type == TriggerType.POINT:
            if self.event_time_ns is None:
                raise ValueError("POINT trigger requires event_time_ns")
        elif self.type == TriggerType.RANGE:
            if self.start_time_ns is None or self.end_time_ns is None:
                raise ValueError("RANGE trigger requires start_time_ns and end_time_ns")
            if self.start_time_ns > self.end_time_ns:
                raise ValueError("start_time_ns must be <= end_time_ns")

    @classmethod
    def point(
        cls,
        event_time_ns: int,
        pre_sec: float = 3.0,
        post_sec: float = 2.0,
        label: str = "",
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Trigger":
        """Create a POINT trigger for a single moment event."""
        return cls(
            type=TriggerType.POINT,
            event_time_ns=event_time_ns,
            pre_sec=pre_sec,
            post_sec=post_sec,
            label=label,
            score=score,
            metadata=metadata or {},
        )

    @classmethod
    def range(
        cls,
        start_time_ns: int,
        end_time_ns: int,
        pre_sec: float = 3.0,
        post_sec: float = 2.0,
        label: str = "",
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Trigger":
        """Create a RANGE trigger for a time span event."""
        return cls(
            type=TriggerType.RANGE,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            pre_sec=pre_sec,
            post_sec=post_sec,
            label=label,
            score=score,
            metadata=metadata or {},
        )

    @property
    def clip_start_ns(self) -> int:
        """Calculate clip start timestamp in nanoseconds."""
        pre_ns = seconds_to_ns(self.pre_sec)
        if self.type == TriggerType.POINT:
            return max(0, self.event_time_ns - pre_ns)
        else:
            return max(0, self.start_time_ns - pre_ns)

    @property
    def clip_end_ns(self) -> int:
        """Calculate clip end timestamp in nanoseconds."""
        post_ns = seconds_to_ns(self.post_sec)
        if self.type == TriggerType.POINT:
            return self.event_time_ns + post_ns
        else:
            return self.end_time_ns + post_ns

    @property
    def clip_start_sec(self) -> float:
        """Clip start in seconds."""
        return ns_to_seconds(self.clip_start_ns)

    @property
    def clip_end_sec(self) -> float:
        """Clip end in seconds."""
        return ns_to_seconds(self.clip_end_ns)

    @property
    def clip_duration_sec(self) -> float:
        """Clip duration in seconds."""
        return self.clip_end_sec - self.clip_start_sec
