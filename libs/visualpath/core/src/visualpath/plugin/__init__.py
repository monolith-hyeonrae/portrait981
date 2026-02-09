"""Plugin discovery and loading system.

This module provides infrastructure for discovering and loading
analyzer plugins via Python entry points.
"""

from visualpath.plugin.discovery import (
    discover_analyzers,
    discover_fusions,
    load_analyzer,
    load_fusion,
    create_analyzer,
    PluginRegistry,
    ANALYZERS_GROUP,
    FUSIONS_GROUP,
)

__all__ = [
    "discover_analyzers",
    "discover_fusions",
    "load_analyzer",
    "load_fusion",
    "create_analyzer",
    "PluginRegistry",
    "ANALYZERS_GROUP",
    "FUSIONS_GROUP",
]
