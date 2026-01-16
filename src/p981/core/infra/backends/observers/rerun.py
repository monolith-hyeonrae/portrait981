"""Rerun 기반 observer 백엔드 (스켈레톤 경고)."""

from __future__ import annotations

from loguru import logger

from p981.core.application.protocols.observer import ObserverEvent


class _WarningObserverBackend:
    """미구현 백엔드에 대해 1회 경고를 출력한다."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._warned = False

    def emit(self, event: ObserverEvent) -> None:
        if not self._warned:
            logger.warning("{} observer backend is not implemented yet.", self._name)
            self._warned = True


class RerunObserverBackend(_WarningObserverBackend):
    def __init__(self) -> None:
        super().__init__("rerun")
