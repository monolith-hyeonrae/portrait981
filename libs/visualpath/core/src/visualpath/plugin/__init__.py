"""Plugin discovery and loading system.

This module provides infrastructure for discovering and loading
module plugins via Python entry points.
"""

from visualpath.plugin.discovery import (
    # Unified API
    discover_modules,
    load_module,
    create_module,
    MODULES_GROUP,
    # Aliases (backward compatibility)
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
    # Unified API
    "discover_modules",
    "load_module",
    "create_module",
    "MODULES_GROUP",
    # Aliases (backward compatibility)
    "discover_analyzers",
    "discover_fusions",
    "load_analyzer",
    "load_fusion",
    "create_analyzer",
    "PluginRegistry",
    "ANALYZERS_GROUP",
    "FUSIONS_GROUP",
]
