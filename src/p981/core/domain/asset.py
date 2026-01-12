from __future__ import annotations

from typing import Protocol, Sequence

from ..types import AssetRef, CustomerId


class AssetService(Protocol):
    def save_asset(
        self,
        asset_type: str,
        customer_id: CustomerId | None,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        """Persist an asset record and return its asset_ref."""

    def get_asset_meta(self, asset_ref: AssetRef) -> dict[str, object]:
        """Load metadata for an asset."""

    def update_history(self, customer_id: CustomerId, moment_refs: Sequence[AssetRef]) -> bool:
        """Update customer history and report whether it changed."""


class StubAssetService:
    def save_asset(
        self,
        asset_type: str,
        customer_id: CustomerId | None,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        raise NotImplementedError("AssetService.save_asset is not implemented")

    def get_asset_meta(self, asset_ref: AssetRef) -> dict[str, object]:
        raise NotImplementedError("AssetService.get_asset_meta is not implemented")

    def update_history(self, customer_id: CustomerId, moment_refs: Sequence[AssetRef]) -> bool:
        raise NotImplementedError("AssetService.update_history is not implemented")
