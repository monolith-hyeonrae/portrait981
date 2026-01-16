"""모먼트 도메인 모델을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from p981.core.types import TimeRange


@dataclass(frozen=True)
class MomentSelection:
    time_range: TimeRange
    keyframe_timestamps_ms: Sequence[int]
    metadata: dict[str, object] | None = None
