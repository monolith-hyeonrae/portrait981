from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..types import AssetRef, CustomerId, TimeRange


@dataclass(frozen=True)
class MomentSelection:
    time_range: TimeRange
    keyframe_timestamps_ms: Sequence[int]
    metadata: dict[str, object] | None = None


class MomentService(Protocol):
    def select_moments(
        self, state_timeline_ref: AssetRef, customer_id: CustomerId | None
    ) -> Sequence[MomentSelection]:
        """Select moments with dedupe/diversity rules applied."""


class StubMomentService:
    def select_moments(
        self, state_timeline_ref: AssetRef, customer_id: CustomerId | None
    ) -> Sequence[MomentSelection]:
        raise NotImplementedError("MomentService.select_moments is not implemented")
