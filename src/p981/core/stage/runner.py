from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..common import ProgressReporter


@dataclass(frozen=True)
class StageStep:
    name: str
    action: Callable[[], object]
    message: str | None = None
    progress: float | None = None


class StageRunner:
    def __init__(self, reporter: ProgressReporter) -> None:
        self._reporter = reporter

    def run(self, steps: Sequence[StageStep]) -> list[object]:
        total = max(1, len(steps))
        results: list[object] = []
        for index, step in enumerate(steps, start=1):
            progress = step.progress
            if progress is None:
                progress = index / (total + 1)
            self._reporter.emit(step.name, step.message, progress)
            results.append(step.action())
            self._reporter.emit_complete(step.name, step.message)
        return results

    def run_step(self, step: StageStep) -> object:
        return self.run([step])[0]
