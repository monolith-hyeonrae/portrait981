from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationEvent:
    kind: str
    payload: dict[str, object]
    asset_ref: str | None = None
    tags: dict[str, str] | None = None
    timestamp_ms: int | None = None
