"""자산 검색/중복 판별을 위한 인덱스 프로토콜을 정의한다."""

from __future__ import annotations

from typing import Protocol

from p981.core.types import AssetRef, MemberId


class AssetIndex(Protocol):
    def index(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        """검색/중복 판별을 위해 자산 메타를 인덱싱한다."""

    def search(self, member_id: MemberId, query: dict[str, object]) -> list[AssetRef]:
        """멤버 기준으로 자산을 검색한다."""


class StubAssetIndex:
    def index(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        raise NotImplementedError("AssetIndex.index is not implemented")

    def search(self, member_id: MemberId, query: dict[str, object]) -> list[AssetRef]:
        raise NotImplementedError("AssetIndex.search is not implemented")
