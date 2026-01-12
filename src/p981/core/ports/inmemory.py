from __future__ import annotations

"""In-memory ports for single-process skeleton runs."""

from uuid import uuid4

from .asset_index import AssetIndex
from .blob_store import BlobStore
from .cache import Cache
from .meta_store import MetaStore
from ..types import AssetRef, CustomerId


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
        self._by_customer: dict[CustomerId, list[AssetRef]] = {}

    def index(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        customer_id = meta.get("customer_id")
        if isinstance(customer_id, str):
            self._by_customer.setdefault(customer_id, []).append(asset_ref)

    def search(self, customer_id: CustomerId, query: dict[str, object]) -> list[AssetRef]:
        return list(self._by_customer.get(customer_id, []))


class InMemoryCache(Cache):
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def get(self, key: str) -> object | None:
        return self._values.get(key)

    def set(self, key: str, value: object, ttl_sec: int | None = None) -> None:
        self._values[key] = value
