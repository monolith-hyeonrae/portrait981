from __future__ import annotations

from typing import Protocol

from ..types import AssetRef


class MetaStore(Protocol):
    def save(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        """Persist metadata for an asset."""

    def load(self, asset_ref: AssetRef) -> dict[str, object]:
        """Load metadata for an asset."""


class StubMetaStore:
    def save(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        raise NotImplementedError("MetaStore.save is not implemented")

    def load(self, asset_ref: AssetRef) -> dict[str, object]:
        raise NotImplementedError("MetaStore.load is not implemented")
