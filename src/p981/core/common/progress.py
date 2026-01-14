"""스테이지 진행률 업데이트 타입과 전송 로직을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class ProgressUpdate:
    """스텝 실행 중 발생하는 단일 진행률 업데이트."""

    stage: str
    step: str
    message: str | None = None
    progress: float | None = None


class ProgressSink(Protocol):
    def emit(self, update: ProgressUpdate) -> None:
        """스테이지 진행률 업데이트를 수신한다."""


ProgressCallback = Callable[[ProgressUpdate], None]
ProgressHandler = ProgressCallback | ProgressSink | None


def emit_progress(handler: ProgressHandler, update: ProgressUpdate) -> None:
    """지정된 핸들러로 진행률 업데이트를 전달한다."""
    if handler is None:
        return

    if hasattr(handler, "emit"):
        # Protocol 호환 sink.
        handler.emit(update)  # type: ignore[call-arg]
        return

    # 일반 콜백.
    handler(update)  # type: ignore[misc]


class ProgressReporter:
    """스텝 진행률 업데이트를 구성해 발행한다."""
    def __init__(self, stage: str, handler: ProgressHandler | None) -> None:
        self._stage = stage
        self._handler = handler

    def emit(self, step: str, message: str | None = None, progress: float | None = None) -> None:
        """스텝 진행 중 업데이트를 발행한다."""
        emit_progress(
            self._handler,
            ProgressUpdate(stage=self._stage, step=step, message=message, progress=progress),
        )

    def emit_complete(
        self, step: str, message: str | None = None, progress: float | None = None
    ) -> None:
        """스텝 완료 업데이트를 발행한다."""
        complete_message = "complete"
        if message:
            complete_message = f"{message} complete"
        self.emit(f"{step}.complete", complete_message, progress)
