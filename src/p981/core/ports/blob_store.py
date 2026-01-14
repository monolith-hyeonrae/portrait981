"""미디어 blob 저장/조회 포트를 정의한다."""

from __future__ import annotations

from typing import Protocol

from ..types import AssetRef


class BlobStore(Protocol):
    def put(self, blob: bytes) -> AssetRef:
        """바이너리를 저장하고 blob_ref를 반환한다."""

    def get(self, blob_ref: AssetRef) -> bytes:
        """blob_ref로 바이너리를 조회한다."""


class StubBlobStore:
    def put(self, blob: bytes) -> AssetRef:
        raise NotImplementedError("BlobStore.put is not implemented")

    def get(self, blob_ref: AssetRef) -> bytes:
        raise NotImplementedError("BlobStore.get is not implemented")
