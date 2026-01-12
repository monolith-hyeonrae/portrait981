from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AssetRef = str
VideoRef = str
CustomerId = str
Style = Literal["base", "closeup", "fullbody", "cinematic"]


@dataclass(frozen=True)
class TimeRange:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class DiscoverInput:
    video_ref: VideoRef
    customer_id: CustomerId | None = None


@dataclass(frozen=True)
class DiscoverOutput:
    moment_refs: list[AssetRef]
    keyframe_pack_refs: list[AssetRef]
    moment_clip_refs: list[AssetRef]
    moment_metadata_refs: list[AssetRef]
    history_updated: bool


@dataclass(frozen=True)
class SynthesizeInput:
    style: Style
    moment_ref: AssetRef | None = None
    base_portrait_ref: AssetRef | None = None
    closeup_image_ref: AssetRef | None = None
    fullbody_image_ref: AssetRef | None = None


@dataclass(frozen=True)
class SynthesizeOutput:
    generated_asset_ref: AssetRef
    reused_existing: bool
