"""Discover 스테이지 전용 스텝 구성 요소."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...domain import AssetService, MediaService, MomentSelection, MomentService, StateService
from ...executor.step_runner import StageRunner, StageStep
from ...types import AssetRef, FrameSource, MediaHandle, MemberId, VideoRef


class DiscoverDeps(Protocol):
    """Discover 스텝 실행에 필요한 의존성."""

    media: MediaService
    state: StateService
    moment: MomentService
    asset: AssetService


@dataclass
class DiscoverContext:
    """Discover 실행 컨텍스트."""

    video_ref: VideoRef
    member_id: MemberId | None
    media_handle: MediaHandle | None = None
    state_frame_source: FrameSource | None = None
    moment_frame_source: FrameSource | None = None
    state_timeline_ref: AssetRef | None = None
    selections: list[MomentSelection] = field(default_factory=list)


@dataclass
class DiscoverStepBuilder:
    """Discover 스텝 구성과 실행을 돕는다."""

    deps: DiscoverDeps
    state_fps: int
    moment_fps: int
    frame_limit: int | None

    def run_initial_steps(self, runner: StageRunner, context: DiscoverContext) -> None:
        runner.run(
            [
                StageStep(
                    name="media.prepare",
                    message="prepare media_handle",
                    progress=0.05,
                    action=lambda: self._prepare_media(context),
                ),
                StageStep(
                    name="media.open_state_frames",
                    message="open frame_source for state",
                    progress=0.15,
                    action=lambda: self._open_state_frames(context),
                ),
                StageStep(
                    name="state.build_state_timeline",
                    message="build state timeline",
                    progress=0.25,
                    action=lambda: self._build_state(context),
                ),
                StageStep(
                    name="media.open_moment_frames",
                    message="open frame_source for moment",
                    progress=0.35,
                    action=lambda: self._open_moment_frames(context),
                ),
                StageStep(
                    name="moment.select_moments",
                    message="select moments",
                    progress=0.45,
                    action=lambda: self._select_moments(context),
                ),
            ]
        )

    def persist_selections(
        self, runner: StageRunner, context: DiscoverContext
    ) -> tuple[list[AssetRef], list[AssetRef], list[AssetRef], list[AssetRef]]:
        media_handle = self._require_media_handle(context)

        moment_refs: list[AssetRef] = []
        keyframe_pack_refs: list[AssetRef] = []
        moment_clip_refs: list[AssetRef] = []
        moment_metadata_refs: list[AssetRef] = []

        total = max(1, len(context.selections))
        for index, selection in enumerate(context.selections):
            keyframe_pack_ref, moment_clip_ref, moment_ref = self._persist_selection(
                runner=runner,
                selection=selection,
                index=index,
                total=total,
                media_handle=media_handle,
                member_id=context.member_id,
                source_ref=context.video_ref,
            )
            moment_refs.append(moment_ref)
            keyframe_pack_refs.append(keyframe_pack_ref)
            moment_clip_refs.append(moment_clip_ref)
            moment_metadata_refs.append(moment_ref)

        return moment_refs, keyframe_pack_refs, moment_clip_refs, moment_metadata_refs

    def update_history(
        self, runner: StageRunner, context: DiscoverContext, moment_refs: list[AssetRef]
    ) -> bool:
        if context.member_id is None:
            return False

        return runner.run_step(
            StageStep(
                name="asset.update_history",
                message="update member history",
                progress=0.95,
                action=lambda: self.deps.asset.update_history(context.member_id, moment_refs),
            )
        )

    def _prepare_media(self, context: DiscoverContext) -> None:
        context.media_handle = self.deps.media.register_video(context.video_ref)

    def _open_state_frames(self, context: DiscoverContext) -> None:
        media_handle = self._require_media_handle(context)
        context.state_frame_source = self.deps.media.open_frame_source(
            media_handle,
            fps=self.state_fps,
            max_frames=self.frame_limit,
        )

    def _build_state(self, context: DiscoverContext) -> None:
        state_frame_source = self._require_state_frame_source(context)
        context.state_timeline_ref = self.deps.state.build_state_timeline(state_frame_source)

    def _open_moment_frames(self, context: DiscoverContext) -> None:
        media_handle = self._require_media_handle(context)
        context.moment_frame_source = self.deps.media.open_frame_source(
            media_handle,
            fps=self.moment_fps,
            max_frames=self.frame_limit,
        )

    def _select_moments(self, context: DiscoverContext) -> None:
        state_timeline_ref = self._require_state_timeline_ref(context)
        context.selections = list(
            self.deps.moment.select_moments(
                state_timeline_ref, context.member_id, frame_source=context.moment_frame_source
            )
        )

    def _persist_selection(
        self,
        runner: StageRunner,
        selection: MomentSelection,
        index: int,
        total: int,
        media_handle: MediaHandle,
        member_id: MemberId | None,
        source_ref: VideoRef,
    ) -> tuple[AssetRef, AssetRef, AssetRef]:
        position = index + 1
        timestamps_ms = list(selection.keyframe_timestamps_ms)
        time_range = selection.time_range

        keyframe_blob_ref = runner.run_step(
            StageStep(
                name="media.extract_keyframes",
                message=f"extract keyframes ({position}/{total})",
                progress=0.55,
                action=lambda: self.deps.media.extract_keyframes(media_handle, timestamps_ms),
            )
        )
        keyframe_pack_ref = runner.run_step(
            StageStep(
                name="asset.save_keyframe_pack",
                message=f"save keyframe pack ({position}/{total})",
                progress=0.65,
                action=lambda: self.deps.asset.save_asset(
                    asset_type="keyframe_pack",
                    member_id=member_id,
                    source_ref=source_ref,
                    blob_ref=keyframe_blob_ref,
                    meta={"timestamps_ms": timestamps_ms},
                ),
            )
        )

        clip_blob_ref = runner.run_step(
            StageStep(
                name="media.extract_clip",
                message=f"extract moment clip ({position}/{total})",
                progress=0.75,
                action=lambda: self.deps.media.extract_clip(media_handle, time_range),
            )
        )
        moment_clip_ref = runner.run_step(
            StageStep(
                name="asset.save_moment_clip",
                message=f"save moment clip ({position}/{total})",
                progress=0.85,
                action=lambda: self.deps.asset.save_asset(
                    asset_type="moment_clip",
                    member_id=member_id,
                    source_ref=source_ref,
                    blob_ref=clip_blob_ref,
                    meta={
                        "time_range": {
                            "start_ms": time_range.start_ms,
                            "end_ms": time_range.end_ms,
                        }
                    },
                ),
            )
        )

        # 선택 메타를 기본값과 합쳐 moment_meta를 구성한다.
        selection_meta = selection.metadata or {}
        moment_meta = self._build_moment_meta(
            selection_meta=selection_meta,
            time_range=time_range,
            keyframe_pack_ref=keyframe_pack_ref,
            moment_clip_ref=moment_clip_ref,
            index=index,
        )
        moment_ref = runner.run_step(
            StageStep(
                name="asset.save_moment",
                message=f"save moment ({position}/{total})",
                progress=0.9,
                action=lambda: self.deps.asset.save_asset(
                    asset_type="moment",
                    member_id=member_id,
                    source_ref=source_ref,
                    blob_ref=None,
                    meta=moment_meta,
                ),
            )
        )

        return keyframe_pack_ref, moment_clip_ref, moment_ref

    def _build_moment_meta(
        self,
        selection_meta: dict[str, object],
        time_range,
        keyframe_pack_ref: AssetRef,
        moment_clip_ref: AssetRef,
        index: int,
    ) -> dict[str, object]:
        moment_meta = {
            "start_ms": time_range.start_ms,
            "end_ms": time_range.end_ms,
            "label": selection_meta.get("label", "neutral"),
            "score": selection_meta.get("score", 0.0),
            "dedupe_hash": selection_meta.get("dedupe_hash", f"stub-{index}"),
            "diversity_key": selection_meta.get("diversity_key", "default"),
            "keyframe_pack_ref": keyframe_pack_ref,
            "moment_clip_ref": moment_clip_ref,
        }
        if "reason" in selection_meta:
            moment_meta["reason"] = selection_meta["reason"]
        return moment_meta

    @staticmethod
    def _require_media_handle(context: DiscoverContext) -> MediaHandle:
        if context.media_handle is None:
            raise ValueError("media_handle is not initialized")
        return context.media_handle

    @staticmethod
    def _require_state_frame_source(context: DiscoverContext) -> FrameSource:
        if context.state_frame_source is None:
            raise ValueError("state_frame_source is not initialized")
        return context.state_frame_source

    @staticmethod
    def _require_state_timeline_ref(context: DiscoverContext) -> AssetRef:
        if context.state_timeline_ref is None:
            raise ValueError("state_timeline_ref is not initialized")
        return context.state_timeline_ref
