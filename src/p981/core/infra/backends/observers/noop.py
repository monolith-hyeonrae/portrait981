"""관측 이벤트를 무시하는 noop 백엔드."""

from __future__ import annotations

from p981.core.application.protocols.observer import ObserverEvent


class NoopObserverBackend:
    def emit(self, event: ObserverEvent) -> None:
        return
