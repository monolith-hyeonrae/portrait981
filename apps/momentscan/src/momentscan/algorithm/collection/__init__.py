"""Unified collection engine for momentscan.

Replaces the dual BatchHighlightEngine + IdentityBuilder paradigm
with a single identity-centric collection engine.
"""

from momentscan.algorithm.collection.engine import CollectionEngine
from momentscan.algorithm.collection.types import (
    CollectionConfig,
    CollectionRecord,
    CollectionResult,
    PersonCollection,
    SelectedFrame,
)

__all__ = [
    "CollectionEngine",
    "CollectionConfig",
    "CollectionRecord",
    "CollectionResult",
    "PersonCollection",
    "SelectedFrame",
]
