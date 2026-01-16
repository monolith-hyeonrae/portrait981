"""여러 observer 백엔드를 합치는 멀티 백엔드."""

from __future__ import annotations

from loguru import logger

from p981.core.application.protocols.observer import ObserverEvent, ObserverProtocol


class MultiObserverBackend:
    def __init__(self, backends: list[ObserverProtocol]) -> None:
        self._backends = backends

    def emit(self, event: ObserverEvent) -> None:
        # 관측 백엔드에 최대한 전달하고 실패는 경고만 남긴다.
        for backend in self._backends:
            try:
                backend.emit(event)
            except Exception as exc:
                logger.warning("Observer backend emit failed: {}", exc)
