"""스테이지 실행을 지원하는 구성요소를 내보낸다."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stage_executor import StageExecutor
    from .step_runner import StageRunner, StageStep

__all__ = ["StageExecutor", "StageRunner", "StageStep"]


def __getattr__(name: str):
    """순환 임포트를 피하기 위해 필요 시에만 로딩한다."""
    if name == "StageExecutor":
        from .stage_executor import StageExecutor

        return StageExecutor
    if name == "StageRunner":
        from .step_runner import StageRunner

        return StageRunner
    if name == "StageStep":
        from .step_runner import StageStep

        return StageStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
