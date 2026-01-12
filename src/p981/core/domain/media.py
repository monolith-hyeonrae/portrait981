from __future__ import annotations

from typing import Protocol, Sequence

from ..types import AssetRef, TimeRange, VideoRef


class MediaService(Protocol):
    def register_video(self, video_ref: VideoRef) -> AssetRef:
        """Prepare access to a raw video and return a handle."""

    def extract_keyframes(self, video_ref: VideoRef, timestamps_ms: Sequence[int]) -> AssetRef:
        """Extract keyframes and return a keyframe_pack_ref."""

    def extract_clip(self, video_ref: VideoRef, time_range: TimeRange) -> AssetRef:
        """Extract a clip and return a moment_clip_ref."""


class StubMediaService:
    def register_video(self, video_ref: VideoRef) -> AssetRef:
        raise NotImplementedError("MediaService.register_video is not implemented")

    def extract_keyframes(self, video_ref: VideoRef, timestamps_ms: Sequence[int]) -> AssetRef:
        raise NotImplementedError("MediaService.extract_keyframes is not implemented")

    def extract_clip(self, video_ref: VideoRef, time_range: TimeRange) -> AssetRef:
        raise NotImplementedError("MediaService.extract_clip is not implemented")
