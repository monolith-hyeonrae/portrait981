"""모먼트 후보/선정 도메인 서비스 인터페이스를 정의한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..types import AssetRef, FrameSource, MemberId, TimeRange


@dataclass(frozen=True)
class MomentSelection:
    time_range: TimeRange
    keyframe_timestamps_ms: Sequence[int]
    metadata: dict[str, object] | None = None


class MomentService(Protocol):
    def select_moments(
        self,
        state_timeline_ref: AssetRef,
        member_id: MemberId | None,
        frame_source: FrameSource | None = None,
    ) -> Sequence[MomentSelection]:
        """중복 제거/다양성 규칙을 적용해 모먼트를 선택한다."""


class StubMomentService:
    def select_moments(
        self,
        state_timeline_ref: AssetRef,
        member_id: MemberId | None,
        frame_source: FrameSource | None = None,
    ) -> Sequence[MomentSelection]:
        raise NotImplementedError("MomentService.select_moments is not implemented")
