from __future__ import annotations

from dataclasses import dataclass

from ..stage import DiscoverStage, SynthesizeStage
from ..common import ProgressHandler
from ..types import DiscoverInput, DiscoverOutput, SynthesizeInput, SynthesizeOutput


@dataclass(frozen=True)
class StageExecutor:
    discover_stage: DiscoverStage
    synthesize_stage: SynthesizeStage

    def run_discover(
        self, request: DiscoverInput, progress: ProgressHandler | None = None
    ) -> DiscoverOutput:
        return self.discover_stage.run(request, progress=progress)

    def run_synthesize(
        self, request: SynthesizeInput, progress: ProgressHandler | None = None
    ) -> SynthesizeOutput:
        return self.synthesize_stage.run(request, progress=progress)
