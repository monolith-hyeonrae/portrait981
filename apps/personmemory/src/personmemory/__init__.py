"""personmemory — Person memory system for portrait981.

Per-member conditional distribution store.
서비스 인프라이자 person-conditioned distribution의 구현체.

Usage:
    from personmemory import PersonMemory

    memory = PersonMemory("test_3")
    memory.ingest(workflow_id="ride_1", frames=shoot_results)
    memory.profile()
    memory.get_reference(expression="cheese")

    PersonMemory.list_all()
    PersonMemory.rename("test_3", "member_042")
"""

from personmemory.memory import PersonMemory, MemoryNode, Profile

# Legacy re-exports (기존 코드 호환)
from personmemory.types import MemoryNode as _LegacyNode, MatchResult, RefQuery, RefSelection
from personmemory.bank import MemoryBank
from personmemory.persistence import save_bank, load_bank

__all__ = [
    "PersonMemory", "MemoryNode", "Profile",
    # Legacy
    "MemoryBank", "MatchResult", "RefQuery", "RefSelection",
    "save_bank", "load_bank",
]
