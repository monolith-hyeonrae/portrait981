"""도메인 서비스별 프로토콜 의존성을 명시적으로 묶는다."""

from __future__ import annotations

from dataclasses import dataclass

from p981.core.application.protocols.asset_index import AssetIndex
from p981.core.application.protocols.blob_store import BlobStore
from p981.core.application.protocols.meta_store import MetaStore
from p981.core.application.protocols.observer import ObserverProtocol


@dataclass(frozen=True)
class MediaProtocols:
    """미디어 도메인 서비스가 사용하는 프로토콜 묶음."""

    blob_store: BlobStore
    observer: ObserverProtocol


@dataclass(frozen=True)
class StateProtocols:
    """상태 도메인 서비스가 사용하는 프로토콜 묶음."""

    meta_store: MetaStore
    observer: ObserverProtocol


@dataclass(frozen=True)
class MomentProtocols:
    """모먼트 도메인 서비스가 사용하는 프로토콜 묶음."""

    observer: ObserverProtocol


@dataclass(frozen=True)
class AssetProtocols:
    """자산 도메인 서비스가 사용하는 프로토콜 묶음."""

    meta_store: MetaStore
    asset_index: AssetIndex | None = None


@dataclass(frozen=True)
class SynthesisProtocols:
    """합성 도메인 서비스가 사용하는 프로토콜 묶음."""

    blob_store: BlobStore
