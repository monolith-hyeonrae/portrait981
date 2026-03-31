"""MomentBank — member 단위 person memory store.

전체 member를 관리하는 최상위 인터페이스.

Usage:
    bank = MomentBank()
    bank.ingest("test_3", workflow_id="ride_1", frames=results)
    refs = bank.get_reference("test_3", expression="cheese")
    profile = bank.profile("test_3")
    bank.rename_member("test_3", "member_042")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from personmemory.member import MemberMemory, MemberProfile

logger = logging.getLogger("personmemory.store")


class MomentBank:
    """전체 member를 관리하는 person memory store."""

    def __init__(self, root_dir: str | Path = "data/personmemory/members"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MemberMemory] = {}

    def _get_member(self, member_id: str) -> MemberMemory:
        """Get or load MemberMemory (cached)."""
        if member_id not in self._cache:
            store_dir = self.root_dir / member_id
            self._cache[member_id] = MemberMemory.load(member_id, store_dir=store_dir)
        return self._cache[member_id]

    # ── Ingest ──

    def ingest(
        self,
        member_id: str,
        workflow_id: str,
        frames: list,
        summary=None,
        timestamp: str = "",
        auto_save: bool = True,
    ) -> dict:
        """Workflow 결과를 member 기억에 축적.

        Args:
            member_id: 고객 식별자
            workflow_id: 탑승 식별자
            frames: list[FrameResult] from momentscan
            summary: SignalSummary (optional)
            timestamp: ISO timestamp
            auto_save: 자동 저장 여부

        Returns:
            dict with ingest stats
        """
        mem = self._get_member(member_id)
        mem.store_dir = self.root_dir / member_id
        result = mem.ingest(workflow_id=workflow_id, frames=frames,
                           summary=summary, timestamp=timestamp)

        if auto_save:
            mem.save()

        return result

    # ── Retrieval ──

    def get_reference(
        self,
        member_id: str,
        expression: Optional[str] = None,
        pose: Optional[str] = None,
        top_k: int = 3,
    ) -> list[str]:
        """참조 이미지 경로 반환."""
        return self._get_member(member_id).get_reference(
            expression=expression, pose=pose, top_k=top_k,
        )

    def get_profile_reference(self, member_id: str, top_k: int = 1) -> list[str]:
        """프로필 대표 이미지 반환."""
        return self._get_member(member_id).get_profile_reference(top_k=top_k)

    # ── Profile ──

    def profile(self, member_id: str) -> MemberProfile:
        """Member 프로필."""
        return self._get_member(member_id).profile()

    # ── Marginal Value ──

    def marginal_value(self, member_id: str, signal) -> float:
        """새 프레임의 marginal value."""
        import numpy as np
        return self._get_member(member_id).marginal_value(np.asarray(signal))

    # ── Management ──

    def list_members(self) -> list[str]:
        """등록된 전체 member 목록."""
        members = []
        if self.root_dir.exists():
            for d in sorted(self.root_dir.iterdir()):
                if d.is_dir() and (d / "memory.json").exists():
                    members.append(d.name)
        return members

    def rename_member(self, old_id: str, new_id: str) -> None:
        """Member ID 변경."""
        old_dir = self.root_dir / old_id
        new_dir = self.root_dir / new_id
        if not old_dir.exists():
            raise FileNotFoundError(f"Member not found: {old_id}")
        if new_dir.exists():
            raise FileExistsError(f"Member already exists: {new_id}")

        old_dir.rename(new_dir)

        # Update memory.json
        mem = MemberMemory.load(new_id, store_dir=new_dir)
        mem.member_id = new_id
        mem.save()

        # Clear cache
        self._cache.pop(old_id, None)
        self._cache.pop(new_id, None)

        logger.info("Renamed member: %s → %s", old_id, new_id)

    def delete_member(self, member_id: str) -> None:
        """Member 기억 삭제."""
        import shutil
        member_dir = self.root_dir / member_id
        if member_dir.exists():
            shutil.rmtree(member_dir)
        self._cache.pop(member_id, None)
        logger.info("Deleted member: %s", member_id)
