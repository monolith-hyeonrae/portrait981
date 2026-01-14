"""도메인 서비스별 포트 의존성을 명시적으로 묶는다."""

from __future__ import annotations

from dataclasses import dataclass

from .asset_index import AssetIndex
from .blob_store import BlobStore
from .meta_store import MetaStore
from .observation import ObservationPort


@dataclass(frozen=True)
class MediaPorts:
    """미디어 도메인 서비스가 사용하는 포트 묶음."""

    blob_store: BlobStore
    observer: ObservationPort


@dataclass(frozen=True)
class StatePorts:
    """상태 도메인 서비스가 사용하는 포트 묶음."""

    meta_store: MetaStore
    observer: ObservationPort


@dataclass(frozen=True)
class MomentPorts:
    """모먼트 도메인 서비스가 사용하는 포트 묶음."""

    observer: ObservationPort


@dataclass(frozen=True)
class AssetPorts:
    """자산 도메인 서비스가 사용하는 포트 묶음."""

    meta_store: MetaStore
    asset_index: AssetIndex | None = None


@dataclass(frozen=True)
class SynthesisPorts:
    """합성 도메인 서비스가 사용하는 포트 묶음."""

    blob_store: BlobStore
