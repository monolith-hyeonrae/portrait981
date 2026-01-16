"""스켈레톤 실행을 위한 인메모리 도메인 구현체 모음."""

from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from p981.core.application.protocols import (
    AssetProtocols,
    MediaProtocols,
    MomentProtocols,
    StateProtocols,
    SynthesisProtocols,
)
from p981.core.application.protocols.observer import ObserverEvent
from p981.core.application.services.asset import AssetService
from p981.core.application.services.media import MediaService
from p981.core.application.services.moment import MomentService
from p981.core.application.services.state import StateService
from p981.core.application.services.synthesis import SynthesisService
from p981.core.domain.moment import MomentSelection
from p981.core.types import AssetRef, FrameSample, FrameSource, MediaHandle, MemberId, TimeRange, VideoRef


class _InMemoryFrameSource:
    """스켈레톤 실행용 결정적 프레임 샘플러."""
    def __init__(
        self,
        media_handle: MediaHandle,
        fps: int,
        time_range: TimeRange | None,
        max_frames: int | None,
    ) -> None:
        self.media_handle = media_handle
        self.fps = fps
        self._time_range = time_range
        self._max_frames = max_frames
        self._frames: list[FrameSample] | None = None

    def iter_frames(self) -> Sequence[FrameSample]:
        if self._frames is not None:
            return list(self._frames)

        # 스켈레톤 실행을 위한 결정적 샘플링.
        start_ms = self._time_range.start_ms if self._time_range else 0
        end_ms = self._time_range.end_ms if self._time_range else start_ms + 3000
        step_ms = max(1, int(1000 / self.fps)) if self.fps > 0 else 1000
        total = max(0, (end_ms - start_ms) // step_ms)
        if total == 0:
            total = 1
        if self._max_frames is not None:
            total = min(total, self._max_frames)

        frames = [
            FrameSample(frame_index=index, timestamp_ms=start_ms + index * step_ms)
            for index in range(total)
        ]
        self._frames = frames
        return list(frames)


class InMemoryMediaService(MediaService):
    """합성 이벤트를 발행하는 인메모리 미디어 서비스."""
    def __init__(self, protocols: MediaProtocols) -> None:
        self._blob_store = protocols.blob_store
        self._observer = protocols.observer

    def register_video(self, video_ref: VideoRef) -> MediaHandle:
        return video_ref

    def open_frame_source(
        self,
        media_handle: MediaHandle,
        fps: int,
        time_range: TimeRange | None = None,
        max_frames: int | None = None,
    ) -> FrameSource:
        return _InMemoryFrameSource(media_handle, fps, time_range, max_frames)

    def extract_keyframes(self, media_handle: MediaHandle, timestamps_ms: Sequence[int]) -> AssetRef:
        # 스텁 프레임에 대한 합성 관측 이벤트를 발행한다.
        for index, timestamp_ms in enumerate(timestamps_ms):
            self._observer.emit(
                ObserverEvent(
                    kind="media.frame",
                    payload={
                        "video_ref": media_handle,
                        "frame_index": index,
                        "timestamp_ms": timestamp_ms,
                        "avg_luma": 0.0,
                    },
                    timestamp_ms=timestamp_ms,
                )
            )

        payload = f"keyframes:{media_handle}:{list(timestamps_ms)}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def extract_clip(self, media_handle: MediaHandle, time_range: TimeRange) -> AssetRef:
        payload = f"clip:{media_handle}:{time_range.start_ms}-{time_range.end_ms}".encode(
            "ascii", "ignore"
        )
        return self._blob_store.put(payload)


class InMemoryStateService(StateService):
    """기본 프레임 메타를 기록하는 인메모리 상태 서비스."""
    def __init__(self, protocols: StateProtocols) -> None:
        self._meta_store = protocols.meta_store
        self._observer = protocols.observer

    def build_state_timeline(self, frame_source: FrameSource) -> AssetRef:
        state_timeline_ref = f"state_{uuid4().hex}"
        frames = list(frame_source.iter_frames())

        for frame in frames:
            self._observer.emit(
                ObserverEvent(
                    kind="state.frame",
                    payload={
                        "video_ref": frame_source.media_handle,
                        "frame_index": frame.frame_index,
                        "timestamp_ms": frame.timestamp_ms,
                        "frame_path": frame.frame_path,
                    },
                    timestamp_ms=frame.timestamp_ms,
                )
            )

        self._meta_store.save(
            state_timeline_ref,
            {
                "media_handle": frame_source.media_handle,
                "fps": frame_source.fps,
                "frame_count": len(frames),
                "tracks": [],
            },
        )
        return state_timeline_ref


class InMemoryMomentService(MomentService):
    """결정적 결과를 반환하는 인메모리 모먼트 서비스."""
    def __init__(self, protocols: MomentProtocols) -> None:
        self._observer = protocols.observer

    def select_moments(
        self,
        state_timeline_ref: AssetRef,
        member_id: MemberId | None,
        frame_source: FrameSource | None = None,
    ) -> Sequence[MomentSelection]:
        frames: list[FrameSample] = []
        if frame_source is not None:
            frames = list(frame_source.iter_frames())
            for frame in frames:
                self._observer.emit(
                    ObserverEvent(
                        kind="moment.frame",
                        payload={
                            "video_ref": frame_source.media_handle,
                            "frame_index": frame.frame_index,
                            "timestamp_ms": frame.timestamp_ms,
                            "frame_path": frame.frame_path,
                        },
                        timestamp_ms=frame.timestamp_ms,
                    )
                )

        if frames:
            start_ms = frames[0].timestamp_ms
            end_ms = frames[-1].timestamp_ms + 1000
        else:
            start_ms = 0
            end_ms = 3000

        # 스켈레톤: 단일 결정적 선택지를 반환.
        metadata = {
            "label": "neutral",
            "score": 0.5,
            "dedupe_hash": f"stub-{state_timeline_ref}",
            "diversity_key": "default",
        }
        selection = MomentSelection(
            time_range=TimeRange(start_ms=start_ms, end_ms=end_ms),
            keyframe_timestamps_ms=[0, 1000, 2000],
            metadata=metadata,
        )
        return [selection]


class InMemoryAssetService(AssetService):
    """인덱스/히스토리 옵션을 가진 인메모리 자산 스토어."""
    def __init__(self, protocols: AssetProtocols) -> None:
        self._meta_store = protocols.meta_store
        self._asset_index = protocols.asset_index
        self._history: dict[MemberId, set[AssetRef]] = {}

    def save_asset(
        self,
        asset_type: str,
        member_id: MemberId | None,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        asset_ref = f"asset_{uuid4().hex}"
        stored_meta = dict(meta)
        stored_meta.setdefault("asset_type", asset_type)
        if member_id is not None:
            stored_meta.setdefault("member_id", member_id)
        stored_meta.setdefault("source_ref", source_ref)
        if blob_ref is not None:
            stored_meta.setdefault("blob_ref", blob_ref)

        self._meta_store.save(asset_ref, stored_meta)
        if self._asset_index is not None:
            self._asset_index.index(asset_ref, stored_meta)
        return asset_ref

    def get_asset_meta(self, asset_ref: AssetRef) -> dict[str, object]:
        return self._meta_store.load(asset_ref)

    def update_history(self, member_id: MemberId, moment_refs: Sequence[AssetRef]) -> bool:
        history = self._history.setdefault(member_id, set())
        before = len(history)
        history.update(moment_refs)
        return len(history) != before


class InMemorySynthesisService(SynthesisService):
    """더미 blob을 생성하는 인메모리 합성 서비스."""
    def __init__(self, protocols: SynthesisProtocols) -> None:
        self._blob_store = protocols.blob_store

    def synthesize_base(self, keyframe_pack_ref: AssetRef) -> AssetRef:
        payload = f"base:{keyframe_pack_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def synthesize_closeup(self, base_portrait_ref: AssetRef) -> AssetRef:
        payload = f"closeup:{base_portrait_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def synthesize_fullbody(self, base_portrait_ref: AssetRef) -> AssetRef:
        payload = f"fullbody:{base_portrait_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def synthesize_cinematic(
        self, closeup_image_ref: AssetRef, fullbody_image_ref: AssetRef
    ) -> AssetRef:
        payload = f"cinematic:{closeup_image_ref}:{fullbody_image_ref}".encode(
            "ascii", "ignore"
        )
        return self._blob_store.put(payload)
