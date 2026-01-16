"""단일 프로세스 스켈레톤 실행용 인메모리 백엔드 구현체."""

from __future__ import annotations

from uuid import uuid4

from p981.core.application.protocols.asset_index import AssetIndex
from p981.core.application.protocols.blob_store import BlobStore
from p981.core.application.protocols.cache import Cache
from p981.core.application.protocols.meta_store import MetaStore
from p981.core.types import AssetRef, MemberId


class InMemoryBlobStore(BlobStore):
    def __init__(self) -> None:
        self._blobs: dict[AssetRef, bytes] = {}

    def put(self, blob: bytes) -> AssetRef:
        blob_ref = f"blob_{uuid4().hex}"
        self._blobs[blob_ref] = blob
        return blob_ref

    def get(self, blob_ref: AssetRef) -> bytes:
        return self._blobs[blob_ref]


class InMemoryMetaStore(MetaStore):
    def __init__(self) -> None:
        self._meta: dict[AssetRef, dict[str, object]] = {}

    def save(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        self._meta[asset_ref] = dict(meta)

    def load(self, asset_ref: AssetRef) -> dict[str, object]:
        return dict(self._meta[asset_ref])


class InMemoryAssetIndex(AssetIndex):
    def __init__(self) -> None:
        self._by_member: dict[MemberId, list[AssetRef]] = {}

    def index(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        member_id = meta.get("member_id")
        if isinstance(member_id, str):
            self._by_member.setdefault(member_id, []).append(asset_ref)

    def search(self, member_id: MemberId, query: dict[str, object]) -> list[AssetRef]:
        return list(self._by_member.get(member_id, []))


class InMemoryCache(Cache):
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def get(self, key: str) -> object | None:
        return self._values.get(key)

    def set(self, key: str, value: object, ttl_sec: int | None = None) -> None:
        self._values[key] = value
