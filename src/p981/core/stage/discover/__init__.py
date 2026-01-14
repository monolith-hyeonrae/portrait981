"""Discover 스테이지의 오케스트레이션 흐름을 구현한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...common import ProgressHandler, ProgressReporter
from ...domain import AssetService, MediaService, MomentService, StateService
from ...executor.step_runner import StageRunner
from ...types import AssetRef, MemberId, VideoRef
from .steps import DiscoverContext, DiscoverStepBuilder


@dataclass(frozen=True)
class DiscoverInput:
    """Discover 스테이지 입력."""

    video_ref: VideoRef
    member_id: MemberId | None = None


@dataclass(frozen=True)
class DiscoverOutput:
    """Discover 스테이지 출력."""

    moment_refs: list[AssetRef]
    keyframe_pack_refs: list[AssetRef]
    moment_clip_refs: list[AssetRef]
    moment_metadata_refs: list[AssetRef]
    history_updated: bool


@dataclass(frozen=True)
class DiscoverStageDeps:
    """Discover 스테이지에 필요한 의존성 묶음."""
    media: MediaService
    state: StateService
    moment: MomentService
    asset: AssetService


class DiscoverStage(Protocol):
    def run(self, request: DiscoverInput, progress: ProgressHandler | None = None) -> DiscoverOutput:
        """Discover 스테이지를 실행하고 결과를 반환한다."""


class StubDiscoverStage:
    def __init__(self, deps: DiscoverStageDeps) -> None:
        self._deps = deps

    def run(self, request: DiscoverInput, progress: ProgressHandler | None = None) -> DiscoverOutput:
        raise NotImplementedError("DiscoverStage.run is not implemented")


class SimpleDiscoverStage:
    """오케스트레이션 전용 Discover 스테이지 구현."""
    def __init__(
        self,
        deps: DiscoverStageDeps,
        state_fps: int = 10,
        moment_fps: int = 5,
        frame_limit: int | None = 30,
    ) -> None:
        self._steps = DiscoverStepBuilder(
            deps=deps,
            state_fps=state_fps,
            moment_fps=moment_fps,
            frame_limit=frame_limit,
        )

    def run(self, request: DiscoverInput, progress: ProgressHandler | None = None) -> DiscoverOutput:
        reporter = ProgressReporter("discover", progress)
        runner = StageRunner(reporter)
        context = DiscoverContext(video_ref=request.video_ref, member_id=request.member_id)

        # 1단계: 상태 타임라인과 모먼트 선택을 생성한다.
        self._steps.run_initial_steps(runner, context)

        # 2단계: 선택 결과를 자산으로 저장한다.
        (
            moment_refs,
            keyframe_pack_refs,
            moment_clip_refs,
            moment_metadata_refs,
        ) = self._steps.persist_selections(runner, context)

        history_updated = self._steps.update_history(runner, context, moment_refs)
        reporter.emit("discover.complete", "discover complete", 1.0)
        return DiscoverOutput(
            moment_refs=moment_refs,
            keyframe_pack_refs=keyframe_pack_refs,
            moment_clip_refs=moment_clip_refs,
            moment_metadata_refs=moment_metadata_refs,
            history_updated=history_updated,
        )
