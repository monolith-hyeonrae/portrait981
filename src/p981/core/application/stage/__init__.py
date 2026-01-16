"""스테이지 인터페이스와 기본 구현을 내보낸다."""

from p981.core.application.stage.discover import (
    DiscoverStage,
    DiscoverStageDeps,
    SimpleDiscoverStage,
    StubDiscoverStage,
)
from p981.core.application.stage.synthesize import (
    SynthesizeStage,
    SynthesizeStageDeps,
    SimpleSynthesizeStage,
    StubSynthesizeStage,
)

__all__ = [
    "DiscoverStage",
    "DiscoverStageDeps",
    "SimpleDiscoverStage",
    "SynthesizeStage",
    "SynthesizeStageDeps",
    "SimpleSynthesizeStage",
    "StubDiscoverStage",
    "StubSynthesizeStage",
]
