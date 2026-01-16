"""상태 타임라인 생성 애플리케이션 서비스 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol

from p981.core.types import AssetRef, FrameSource


class StateService(Protocol):
    def build_state_timeline(self, frame_source: FrameSource) -> AssetRef:
        """프레임을 분석해 state_timeline_ref를 반환한다."""


class StubStateService:
    def build_state_timeline(self, frame_source: FrameSource) -> AssetRef:
        raise NotImplementedError("StateService.build_state_timeline is not implemented")
