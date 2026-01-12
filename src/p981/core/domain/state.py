from __future__ import annotations

from typing import Protocol

from ..types import AssetRef, VideoRef


class StateService(Protocol):
    def build_state_timeline(self, video_ref: VideoRef) -> AssetRef:
        """Analyze video and return a state_timeline_ref."""


class StubStateService:
    def build_state_timeline(self, video_ref: VideoRef) -> AssetRef:
        raise NotImplementedError("StateService.build_state_timeline is not implemented")
