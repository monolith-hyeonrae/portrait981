"""Synthesize 스테이지의 오케스트레이션 흐름을 구현한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...common import ProgressHandler, ProgressReporter
from ...domain import AssetService, SynthesisService
from ...executor.step_runner import StageRunner
from ...types import AssetRef, Style
from .steps import SynthesizeStepBuilder


@dataclass(frozen=True)
class SynthesizeInput:
    """Synthesize 스테이지 입력."""

    style: Style
    moment_ref: AssetRef | None = None
    base_portrait_ref: AssetRef | None = None
    closeup_image_ref: AssetRef | None = None
    fullbody_image_ref: AssetRef | None = None


@dataclass(frozen=True)
class SynthesizeOutput:
    """Synthesize 스테이지 출력."""

    generated_asset_ref: AssetRef
    reused_existing: bool


@dataclass(frozen=True)
class SynthesizeStageDeps:
    """Synthesize 스테이지에 필요한 의존성 묶음."""
    asset: AssetService
    synthesis: SynthesisService


class SynthesizeStage(Protocol):
    def run(self, request: SynthesizeInput, progress: ProgressHandler | None = None) -> SynthesizeOutput:
        """Synthesize 스테이지를 실행하고 결과를 반환한다."""


class StubSynthesizeStage:
    def __init__(self, deps: SynthesizeStageDeps) -> None:
        self._deps = deps

    def run(self, request: SynthesizeInput, progress: ProgressHandler | None = None) -> SynthesizeOutput:
        raise NotImplementedError("SynthesizeStage.run is not implemented")


class SimpleSynthesizeStage:
    """오케스트레이션 전용 Synthesize 스테이지 구현."""
    def __init__(self, deps: SynthesizeStageDeps) -> None:
        self._steps = SynthesizeStepBuilder(deps=deps)

    def run(self, request: SynthesizeInput, progress: ProgressHandler | None = None) -> SynthesizeOutput:
        reporter = ProgressReporter("synthesize", progress)
        runner = StageRunner(reporter)

        generated_ref = self._steps.run(request, runner)
        reporter.emit("synthesize.complete", "synthesize complete", 1.0)
        return SynthesizeOutput(generated_asset_ref=generated_ref, reused_existing=False)
