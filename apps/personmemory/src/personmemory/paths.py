"""Bank path utilities with SHA1-based directory sharding.

Member banks are stored under ``~/.portrait981/personmemory/{shard}/{member_id}/``
where *shard* is the first two hex digits of ``SHA1(member_id)``, distributing
entries across 256 buckets for filesystem scalability.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from vpx.sdk.paths import get_home_dir


def _shard(member_id: str) -> str:
    """SHA1 hash prefix (2 hex chars) -> 256 buckets."""
    return hashlib.sha1(member_id.encode()).hexdigest()[:2]


def get_bank_base_dir() -> Path:
    """Return ``~/.portrait981/personmemory/``."""
    return get_home_dir() / "personmemory"


def get_bank_dir(member_id: str) -> Path:
    """Return ``~/.portrait981/personmemory/{shard}/{member_id}/``."""
    return get_bank_base_dir() / _shard(member_id) / member_id


def get_bank_path(member_id: str) -> Path:
    """Return ``~/.portrait981/personmemory/{shard}/{member_id}/memory_bank.json``."""
    return get_bank_dir(member_id) / "memory_bank.json"


def list_member_ids() -> list[str]:
    """List all member_ids that have a saved memory.

    Scans shard directories under the bank base dir.
    """
    base = get_bank_base_dir()
    if not base.exists():
        return []
    ids: list[str] = []
    for shard_dir in sorted(base.iterdir()):
        if shard_dir.is_dir() and len(shard_dir.name) == 2:
            for member_dir in sorted(shard_dir.iterdir()):
                if (member_dir / "memory.json").exists() or (member_dir / "memory_bank.json").exists():
                    ids.append(member_dir.name)
    return ids
