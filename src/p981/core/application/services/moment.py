"""모먼트 후보/선정 애플리케이션 서비스 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol, Sequence

from p981.core.domain.moment import MomentSelection
from p981.core.types import AssetRef, FrameSource, MemberId


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
