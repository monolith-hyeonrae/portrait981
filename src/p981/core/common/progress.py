from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class ProgressUpdate:
    stage: str
    step: str
    message: str | None = None
    progress: float | None = None


class ProgressSink(Protocol):
    def emit(self, update: ProgressUpdate) -> None:
        """Receive progress updates from stages."""


ProgressCallback = Callable[[ProgressUpdate], None]
ProgressHandler = ProgressCallback | ProgressSink | None


def emit_progress(handler: ProgressHandler, update: ProgressUpdate) -> None:
    if handler is None:
        return
    if hasattr(handler, "emit"):
        handler.emit(update)  # type: ignore[call-arg]
        return
    handler(update)  # type: ignore[misc]


class ProgressReporter:
    def __init__(self, stage: str, handler: ProgressHandler | None) -> None:
        self._stage = stage
        self._handler = handler

    def emit(self, step: str, message: str | None = None, progress: float | None = None) -> None:
        emit_progress(
            self._handler,
            ProgressUpdate(stage=self._stage, step=step, message=message, progress=progress),
        )

    def emit_complete(
        self, step: str, message: str | None = None, progress: float | None = None
    ) -> None:
        complete_message = "complete"
        if message:
            complete_message = f"{message} complete"
        self.emit(f"{step}.complete", complete_message, progress)
