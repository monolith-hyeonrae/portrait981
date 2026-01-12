from __future__ import annotations

from typing import Protocol

from ..types import AssetRef, CustomerId


class AssetIndex(Protocol):
    def index(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        """Index asset metadata for search/dedupe."""

    def search(self, customer_id: CustomerId, query: dict[str, object]) -> list[AssetRef]:
        """Search assets by query for a customer."""


class StubAssetIndex:
    def index(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        raise NotImplementedError("AssetIndex.index is not implemented")

    def search(self, customer_id: CustomerId, query: dict[str, object]) -> list[AssetRef]:
        raise NotImplementedError("AssetIndex.search is not implemented")
