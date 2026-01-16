"""자산 메타데이터 저장/조회 프로토콜을 정의한다."""

from __future__ import annotations

from typing import Protocol

from p981.core.types import AssetRef


class MetaStore(Protocol):
    def save(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        """자산 메타데이터를 저장한다."""

    def load(self, asset_ref: AssetRef) -> dict[str, object]:
        """자산 메타데이터를 조회한다."""


class StubMetaStore:
    def save(self, asset_ref: AssetRef, meta: dict[str, object]) -> None:
        raise NotImplementedError("MetaStore.save is not implemented")

    def load(self, asset_ref: AssetRef) -> dict[str, object]:
        raise NotImplementedError("MetaStore.load is not implemented")
