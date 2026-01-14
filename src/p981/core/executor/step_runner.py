"""스테이지 스텝 실행과 진행률 리포팅을 담당한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..common import ProgressReporter


@dataclass(frozen=True)
class StageStep:
    """스테이지 실행의 단일 스텝 정의."""

    name: str
    action: Callable[[], object]
    message: str | None = None
    progress: float | None = None


class StageRunner:
    """스텝 실행과 진행률 리포팅을 담당한다."""
    def __init__(self, reporter: ProgressReporter) -> None:
        self._reporter = reporter

    def run(self, steps: Sequence[StageStep]) -> list[object]:
        """순차 스텝을 실행하고 결과 목록을 반환한다."""
        total = max(1, len(steps))
        results: list[object] = []

        for index, step in enumerate(steps, start=1):
            progress = step.progress
            if progress is None:
                # 스텝이 progress를 주지 않으면 균등 분배한다.
                progress = index / (total + 1)

            self._reporter.emit(step.name, step.message, progress)
            results.append(step.action())

            self._reporter.emit_complete(step.name, step.message)
        return results

    def run_step(self, step: StageStep) -> object:
        """단일 스텝 실행을 위한 래퍼."""
        return self.run([step])[0]
