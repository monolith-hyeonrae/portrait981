"""momentbank - Identity memory bank for portrait981.

Stores per-person Face-ID embeddings as multi-centroid memory nodes,
provides identity matching and context-based reference image selection.

Quick Start:
    >>> from momentbank import MemoryBank
    >>> bank = MemoryBank(person_id=0)
    >>> bank.update(embedding, quality=0.9, meta={"yaw": "[-5,5]"}, image_path="img.jpg")
    >>> result = bank.match(embedding)
    >>> print(f"Stable score: {result.stable_score:.2f}")
"""

from momentbank.types import (
    MemoryNode,
    NodeMeta,
    MatchResult,
    RefQuery,
    RefSelection,
)
from momentbank.bank import MemoryBank
from momentbank.persistence import save_bank, load_bank

__all__ = [
    "MemoryBank",
    "MemoryNode",
    "NodeMeta",
    "MatchResult",
    "RefQuery",
    "RefSelection",
    "save_bank",
    "load_bank",
]
