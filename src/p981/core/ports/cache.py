from __future__ import annotations

from typing import Protocol


class Cache(Protocol):
    def get(self, key: str) -> object | None:
        """Fetch a cached value."""

    def set(self, key: str, value: object, ttl_sec: int | None = None) -> None:
        """Store a cached value."""


class StubCache:
    def get(self, key: str) -> object | None:
        raise NotImplementedError("Cache.get is not implemented")

    def set(self, key: str, value: object, ttl_sec: int | None = None) -> None:
        raise NotImplementedError("Cache.set is not implemented")
