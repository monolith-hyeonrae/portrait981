"""진행률 업데이트를 observer 이벤트로 전달한다."""

from __future__ import annotations

from p981.core.application.executor.progress import ProgressSink, ProgressUpdate
from p981.core.application.protocols.observer import ObserverEvent, ObserverProtocol


class ObserverProgressSink(ProgressSink):
    """진행률 업데이트를 observer 이벤트로 전달한다."""

    def __init__(self, observer: ObserverProtocol, level: str = "INFO") -> None:
        self._observer = observer
        self._level = level

    def emit(self, update: ProgressUpdate) -> None:
        self._observer.emit(
            ObserverEvent(
                kind="progress",
                payload={
                    "level": self._level,
                    "stage": update.stage,
                    "step": update.step,
                    "message": update.message,
                    "progress": update.progress,
                },
            )
        )
