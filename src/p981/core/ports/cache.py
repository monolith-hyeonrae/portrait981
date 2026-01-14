"""캐시 포트 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol


class Cache(Protocol):
    def get(self, key: str) -> object | None:
        """캐시된 값을 조회한다."""

    def set(self, key: str, value: object, ttl_sec: int | None = None) -> None:
        """캐시 값을 저장한다."""


class StubCache:
    def get(self, key: str) -> object | None:
        raise NotImplementedError("Cache.get is not implemented")

    def set(self, key: str, value: object, ttl_sec: int | None = None) -> None:
        raise NotImplementedError("Cache.set is not implemented")
