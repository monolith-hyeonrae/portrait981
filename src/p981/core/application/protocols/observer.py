"""관측 이벤트 프로토콜 정의."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ObserverEvent:
    kind: str
    payload: dict[str, object]
    asset_ref: str | None = None
    tags: dict[str, str] | None = None
    timestamp_ms: int | None = None


class ObserverProtocol(Protocol):
    def emit(self, event: ObserverEvent) -> None:
        """코어 도메인에서 발생한 관측 이벤트를 처리한다."""
