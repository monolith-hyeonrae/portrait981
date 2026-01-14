"""스테이지 실행을 위한 공통 엔트리포인트를 제공한다."""

from __future__ import annotations

from dataclasses import dataclass

from ..stage import DiscoverStage, SynthesizeStage
from ..common import ProgressHandler
from ..stage.discover import DiscoverInput, DiscoverOutput
from ..stage.synthesize import SynthesizeInput, SynthesizeOutput


@dataclass(frozen=True)
class StageExecutor:
    """공통 계약으로 스테이지 구현을 실행하는 파사드."""

    discover_stage: DiscoverStage
    synthesize_stage: SynthesizeStage

    def run_discover(
        self, request: DiscoverInput, progress: ProgressHandler | None = None
    ) -> DiscoverOutput:
        """Discover 스테이지를 실행한다."""
        return self.discover_stage.run(request, progress=progress)

    def run_synthesize(
        self, request: SynthesizeInput, progress: ProgressHandler | None = None
    ) -> SynthesizeOutput:
        """Synthesize 스테이지를 실행한다."""
        return self.synthesize_stage.run(request, progress=progress)
