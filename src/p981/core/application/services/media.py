"""미디어 애플리케이션 서비스 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol, Sequence

from p981.core.types import AssetRef, FrameSource, MediaHandle, TimeRange, VideoRef


class MediaService(Protocol):
    def register_video(self, video_ref: VideoRef) -> MediaHandle:
        """원본 비디오 접근을 준비하고 media_handle을 반환한다."""

    def open_frame_source(
        self,
        media_handle: MediaHandle,
        fps: int,
        time_range: TimeRange | None = None,
        max_frames: int | None = None,
    ) -> FrameSource:
        """요청 fps/구간에 대한 frame_source를 제공한다."""

    def extract_keyframes(
        self, media_handle: MediaHandle, timestamps_ms: Sequence[int]
    ) -> AssetRef:
        """키프레임을 추출해 keyframe_pack_ref를 반환한다."""

    def extract_clip(self, media_handle: MediaHandle, time_range: TimeRange) -> AssetRef:
        """클립을 추출해 moment_clip_ref를 반환한다."""


class StubMediaService:
    def register_video(self, video_ref: VideoRef) -> MediaHandle:
        raise NotImplementedError("MediaService.register_video is not implemented")

    def open_frame_source(
        self,
        media_handle: MediaHandle,
        fps: int,
        time_range: TimeRange | None = None,
        max_frames: int | None = None,
    ) -> FrameSource:
        raise NotImplementedError("MediaService.open_frame_source is not implemented")

    def extract_keyframes(
        self, media_handle: MediaHandle, timestamps_ms: Sequence[int]
    ) -> AssetRef:
        raise NotImplementedError("MediaService.extract_keyframes is not implemented")

    def extract_clip(self, media_handle: MediaHandle, time_range: TimeRange) -> AssetRef:
        raise NotImplementedError("MediaService.extract_clip is not implemented")
