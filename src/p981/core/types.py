"""p981-core에서 사용하는 공통 타입과 데이터 계약을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

AssetRef = str
VideoRef = str
MediaHandle = str
MemberId = str
Style = Literal["base", "closeup", "fullbody", "cinematic"]


@dataclass(frozen=True)
class TimeRange:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    timestamp_ms: int
    frame_path: str | None = None


class FrameSource(Protocol):
    media_handle: MediaHandle
    fps: int

    def iter_frames(self) -> Iterable[FrameSample]:
        """Yield sampled frames for analysis."""

