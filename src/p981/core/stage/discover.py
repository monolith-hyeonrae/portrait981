from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..common import ProgressHandler, ProgressReporter
from ..domain import AssetService, MediaService, MomentService, StateService
from ..types import DiscoverInput, DiscoverOutput
from .runner import StageRunner, StageStep


@dataclass(frozen=True)
class DiscoverStageDeps:
    media: MediaService
    state: StateService
    moment: MomentService
    asset: AssetService


class DiscoverStage(Protocol):
    def run(self, request: DiscoverInput, progress: ProgressHandler | None = None) -> DiscoverOutput:
        """Execute discover stage and return its outputs."""


class StubDiscoverStage:
    def __init__(self, deps: DiscoverStageDeps) -> None:
        self._deps = deps

    def run(self, request: DiscoverInput, progress: ProgressHandler | None = None) -> DiscoverOutput:
        raise NotImplementedError("DiscoverStage.run is not implemented")


class SimpleDiscoverStage:
    def __init__(self, deps: DiscoverStageDeps) -> None:
        self._deps = deps

    def run(self, request: DiscoverInput, progress: ProgressHandler | None = None) -> DiscoverOutput:
        reporter = ProgressReporter("discover", progress)
        runner = StageRunner(reporter)

        customer_id = request.customer_id
        state_timeline_ref: str | None = None
        selections: list = []

        def register_video() -> None:
            self._deps.media.register_video(request.video_ref)

        def build_state() -> None:
            nonlocal state_timeline_ref
            state_timeline_ref = self._deps.state.build_state_timeline(request.video_ref)

        def select_moments() -> None:
            nonlocal selections
            if state_timeline_ref is None:
                raise ValueError("state_timeline_ref is not initialized")
            selections = list(
                self._deps.moment.select_moments(state_timeline_ref, customer_id)
            )

        runner.run(
            [
                StageStep(
                    name="media.register_video",
                    message="register video_ref",
                    progress=0.05,
                    action=register_video,
                ),
                StageStep(
                    name="state.build_state_timeline",
                    message="build state timeline",
                    progress=0.2,
                    action=build_state,
                ),
                StageStep(
                    name="moment.select_moments",
                    message="select moments",
                    progress=0.35,
                    action=select_moments,
                ),
            ]
        )

        moment_refs: list[str] = []
        keyframe_pack_refs: list[str] = []
        moment_clip_refs: list[str] = []
        moment_metadata_refs: list[str] = []

        total = max(1, len(selections))
        for idx, selection in enumerate(selections):
            keyframe_blob_ref = runner.run_step(
                StageStep(
                    name="media.extract_keyframes",
                    message=f"extract keyframes ({idx + 1}/{total})",
                    progress=0.45,
                    action=lambda: self._deps.media.extract_keyframes(
                        request.video_ref, selection.keyframe_timestamps_ms
                    ),
                )
            )
            keyframe_pack_ref = runner.run_step(
                StageStep(
                    name="asset.save_keyframe_pack",
                    message=f"save keyframe pack ({idx + 1}/{total})",
                    progress=0.55,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="keyframe_pack",
                        customer_id=customer_id,
                        source_ref=request.video_ref,
                        blob_ref=keyframe_blob_ref,
                        meta={"timestamps_ms": list(selection.keyframe_timestamps_ms)},
                    ),
                )
            )
            clip_blob_ref = runner.run_step(
                StageStep(
                    name="media.extract_clip",
                    message=f"extract moment clip ({idx + 1}/{total})",
                    progress=0.65,
                    action=lambda: self._deps.media.extract_clip(
                        request.video_ref, selection.time_range
                    ),
                )
            )
            moment_clip_ref = runner.run_step(
                StageStep(
                    name="asset.save_moment_clip",
                    message=f"save moment clip ({idx + 1}/{total})",
                    progress=0.75,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="moment_clip",
                        customer_id=customer_id,
                        source_ref=request.video_ref,
                        blob_ref=clip_blob_ref,
                        meta={
                            "time_range": {
                                "start_ms": selection.time_range.start_ms,
                                "end_ms": selection.time_range.end_ms,
                            }
                        },
                    ),
                )
            )
            selection_meta = selection.metadata or {}
            moment_meta = {
                "start_ms": selection.time_range.start_ms,
                "end_ms": selection.time_range.end_ms,
                "label": selection_meta.get("label", "neutral"),
                "score": selection_meta.get("score", 0.0),
                "dedupe_hash": selection_meta.get("dedupe_hash", f"stub-{idx}"),
                "diversity_key": selection_meta.get("diversity_key", "default"),
                "keyframe_pack_ref": keyframe_pack_ref,
                "moment_clip_ref": moment_clip_ref,
            }
            if "reason" in selection_meta:
                moment_meta["reason"] = selection_meta["reason"]
            moment_ref = runner.run_step(
                StageStep(
                    name="asset.save_moment",
                    message=f"save moment ({idx + 1}/{total})",
                    progress=0.85,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="moment",
                        customer_id=customer_id,
                        source_ref=request.video_ref,
                        blob_ref=None,
                        meta=moment_meta,
                    ),
                )
            )
            moment_refs.append(moment_ref)
            keyframe_pack_refs.append(keyframe_pack_ref)
            moment_clip_refs.append(moment_clip_ref)
            moment_metadata_refs.append(moment_ref)

        history_updated = False
        if customer_id is not None:
            history_updated = runner.run_step(
                StageStep(
                    name="asset.update_history",
                    message="update customer history",
                    progress=0.9,
                    action=lambda: self._deps.asset.update_history(customer_id, moment_refs),
                )
            )
        reporter.emit("discover.complete", "discover complete", 1.0)
        return DiscoverOutput(
            moment_refs=moment_refs,
            keyframe_pack_refs=keyframe_pack_refs,
            moment_clip_refs=moment_clip_refs,
            moment_metadata_refs=moment_metadata_refs,
            history_updated=history_updated,
        )
