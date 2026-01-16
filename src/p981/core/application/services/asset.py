"""자산 저장/조회 및 히스토리 관리 애플리케이션 서비스 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol, Sequence

from p981.core.types import AssetRef, MemberId


class AssetService(Protocol):
    def save_asset(
        self,
        asset_type: str,
        member_id: MemberId | None,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        """자산 레코드를 저장하고 asset_ref를 반환한다."""

    def get_asset_meta(self, asset_ref: AssetRef) -> dict[str, object]:
        """자산 메타데이터를 조회한다."""

    def update_history(self, member_id: MemberId, moment_refs: Sequence[AssetRef]) -> bool:
        """멤버 히스토리를 갱신하고 변경 여부를 반환한다."""


class StubAssetService:
    def save_asset(
        self,
        asset_type: str,
        member_id: MemberId | None,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        raise NotImplementedError("AssetService.save_asset is not implemented")

    def get_asset_meta(self, asset_ref: AssetRef) -> dict[str, object]:
        raise NotImplementedError("AssetService.get_asset_meta is not implemented")

    def update_history(self, member_id: MemberId, moment_refs: Sequence[AssetRef]) -> bool:
        raise NotImplementedError("AssetService.update_history is not implemented")
