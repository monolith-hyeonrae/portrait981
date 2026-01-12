from __future__ import annotations

from typing import Protocol

from ..types import AssetRef


class BlobStore(Protocol):
    def put(self, blob: bytes) -> AssetRef:
        """Store raw bytes and return a blob_ref."""

    def get(self, blob_ref: AssetRef) -> bytes:
        """Load raw bytes by blob_ref."""


class StubBlobStore:
    def put(self, blob: bytes) -> AssetRef:
        raise NotImplementedError("BlobStore.put is not implemented")

    def get(self, blob_ref: AssetRef) -> bytes:
        raise NotImplementedError("BlobStore.get is not implemented")
