from __future__ import annotations

"""In-memory domain services for single-process skeleton runs."""

from typing import Sequence
from uuid import uuid4

from ...common import ObservationEvent
from ...ports import AssetIndex, BlobStore, MetaStore, NoopObservationPort, ObservationPort
from ...types import AssetRef, CustomerId, TimeRange, VideoRef
from ..asset import AssetService
from ..media import MediaService
from ..moment import MomentSelection, MomentService
from ..state import StateService
from ..synthesis import SynthesisService


class InMemoryMediaService(MediaService):
    def __init__(self, blob_store: BlobStore, observer: ObservationPort | None = None) -> None:
        self._blob_store = blob_store
        self._observer = observer or NoopObservationPort()

    def register_video(self, video_ref: VideoRef) -> AssetRef:
        return video_ref

    def extract_keyframes(self, video_ref: VideoRef, timestamps_ms: Sequence[int]) -> AssetRef:
        for index, timestamp_ms in enumerate(timestamps_ms):
            self._observer.emit(
                ObservationEvent(
                    kind="media.frame",
                    payload={
                        "video_ref": video_ref,
                        "frame_index": index,
                        "timestamp_ms": timestamp_ms,
                        "avg_luma": 0.0,
                    },
                    timestamp_ms=timestamp_ms,
                )
            )
        payload = f"keyframes:{video_ref}:{list(timestamps_ms)}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def extract_clip(self, video_ref: VideoRef, time_range: TimeRange) -> AssetRef:
        payload = f"clip:{video_ref}:{time_range.start_ms}-{time_range.end_ms}".encode(
            "ascii", "ignore"
        )
        return self._blob_store.put(payload)


class InMemoryStateService(StateService):
    def __init__(self, meta_store: MetaStore) -> None:
        self._meta_store = meta_store

    def build_state_timeline(self, video_ref: VideoRef) -> AssetRef:
        state_timeline_ref = f"state_{uuid4().hex}"
        self._meta_store.save(
            state_timeline_ref,
            {
                "video_ref": video_ref,
                "tracks": [],
            },
        )
        return state_timeline_ref


class InMemoryMomentService(MomentService):
    def select_moments(
        self, state_timeline_ref: AssetRef, customer_id: CustomerId | None
    ) -> Sequence[MomentSelection]:
        selection = MomentSelection(
            time_range=TimeRange(start_ms=0, end_ms=3000),
            keyframe_timestamps_ms=[0, 1000, 2000],
            metadata={
                "label": "neutral",
                "score": 0.5,
                "dedupe_hash": f"stub-{state_timeline_ref}",
                "diversity_key": "default",
            },
        )
        return [selection]


class InMemoryAssetService(AssetService):
    def __init__(self, meta_store: MetaStore, asset_index: AssetIndex | None = None) -> None:
        self._meta_store = meta_store
        self._asset_index = asset_index
        self._history: dict[CustomerId, set[AssetRef]] = {}

    def save_asset(
        self,
        asset_type: str,
        customer_id: CustomerId | None,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        asset_ref = f"asset_{uuid4().hex}"
        stored_meta = dict(meta)
        stored_meta.setdefault("asset_type", asset_type)
        if customer_id is not None:
            stored_meta.setdefault("customer_id", customer_id)
        stored_meta.setdefault("source_ref", source_ref)
        if blob_ref is not None:
            stored_meta.setdefault("blob_ref", blob_ref)
        self._meta_store.save(asset_ref, stored_meta)
        if self._asset_index is not None:
            self._asset_index.index(asset_ref, stored_meta)
        return asset_ref

    def get_asset_meta(self, asset_ref: AssetRef) -> dict[str, object]:
        return self._meta_store.load(asset_ref)

    def update_history(self, customer_id: CustomerId, moment_refs: Sequence[AssetRef]) -> bool:
        history = self._history.setdefault(customer_id, set())
        before = len(history)
        history.update(moment_refs)
        return len(history) != before


class InMemorySynthesisService(SynthesisService):
    def __init__(self, blob_store: BlobStore) -> None:
        self._blob_store = blob_store

    def synthesize_base(self, keyframe_pack_ref: AssetRef) -> AssetRef:
        payload = f"base:{keyframe_pack_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def synthesize_closeup(self, base_portrait_ref: AssetRef) -> AssetRef:
        payload = f"closeup:{base_portrait_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def synthesize_fullbody(self, base_portrait_ref: AssetRef) -> AssetRef:
        payload = f"fullbody:{base_portrait_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)

    def synthesize_cinematic(self, closeup_image_ref: AssetRef, fullbody_image_ref: AssetRef) -> AssetRef:
        payload = f"cinematic:{closeup_image_ref}:{fullbody_image_ref}".encode("ascii", "ignore")
        return self._blob_store.put(payload)
