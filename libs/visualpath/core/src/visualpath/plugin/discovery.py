"""Plugin discovery system for visualpath.

This module provides infrastructure for discovering and loading
analyzer plugins via Python entry points.

Plugins register themselves using the `visualpath.analyzers` entry point
group in their pyproject.toml:

```toml
[project.entry-points."visualpath.analyzers"]
face = "myplugin.analyzers:FaceAnalyzer"
pose = "myplugin.analyzers:PoseAnalyzer"
```

Example:
    >>> from visualpath.plugin import discover_analyzers, load_analyzer
    >>>
    >>> # Discover all available analyzer plugins
    >>> analyzers = discover_analyzers()
    >>> for name, entry_point in analyzers.items():
    ...     print(f"Found analyzer: {name}")
    >>>
    >>> # Load a specific analyzer
    >>> FaceAnalyzer = load_analyzer("face")
    >>> analyzer = FaceAnalyzer()
"""

import sys
from typing import Dict, Optional, Type, Any

from visualpath.core.module import Module

# Entry point group names
ANALYZERS_GROUP = "visualpath.analyzers"
FUSIONS_GROUP = "visualpath.fusions"


def _get_entry_points(group: str) -> Dict[str, Any]:
    """Get entry points for a group.

    Uses importlib.metadata (Python 3.10+) for entry point discovery.

    Args:
        group: The entry point group name.

    Returns:
        Dict mapping entry point names to entry point objects.
    """
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        eps = entry_points(group=group)
        return {ep.name: ep for ep in eps}
    else:
        # Fallback for Python 3.9
        from importlib.metadata import entry_points as get_eps
        all_eps = get_eps()
        if hasattr(all_eps, 'select'):
            eps = all_eps.select(group=group)
        else:
            eps = all_eps.get(group, [])
        return {ep.name: ep for ep in eps}


def discover_analyzers() -> Dict[str, Any]:
    """Discover all available analyzer plugins.

    Scans the `visualpath.analyzers` entry point group for registered
    analyzers.

    Returns:
        Dict mapping analyzer names to their entry points.

    Example:
        >>> analyzers = discover_analyzers()
        >>> print(list(analyzers.keys()))
        ['face', 'pose', 'gesture', 'quality']
    """
    return _get_entry_points(ANALYZERS_GROUP)


def discover_fusions() -> Dict[str, Any]:
    """Discover all available fusion plugins.

    Scans the `visualpath.fusions` entry point group for registered
    fusion modules.

    Returns:
        Dict mapping fusion names to their entry points.
    """
    return _get_entry_points(FUSIONS_GROUP)


def load_analyzer(name: str) -> Type[Module]:
    """Load an analyzer class by name.

    Args:
        name: The registered name of the analyzer.

    Returns:
        The analyzer class.

    Raises:
        KeyError: If no analyzer with the given name is registered.
        ImportError: If the analyzer cannot be loaded.

    Example:
        >>> FaceAnalyzer = load_analyzer("face")
        >>> analyzer = FaceAnalyzer()
    """
    analyzers = discover_analyzers()
    if name not in analyzers:
        raise KeyError(
            f"No analyzer registered with name '{name}'. "
            f"Available: {list(analyzers.keys())}"
        )
    entry_point = analyzers[name]
    return entry_point.load()


def load_fusion(name: str) -> Type:
    """Load a fusion class by name.

    Args:
        name: The registered name of the fusion module.

    Returns:
        The fusion class.

    Raises:
        KeyError: If no fusion with the given name is registered.
        ImportError: If the fusion cannot be loaded.
    """
    fusions = discover_fusions()
    if name not in fusions:
        raise KeyError(
            f"No fusion registered with name '{name}'. "
            f"Available: {list(fusions.keys())}"
        )
    entry_point = fusions[name]
    return entry_point.load()


def create_analyzer(name: str, **kwargs) -> Module:
    """Create an analyzer instance by name.

    Convenience function that loads and instantiates an analyzer.

    Args:
        name: The registered name of the analyzer.
        **kwargs: Arguments passed to the analyzer constructor.

    Returns:
        An initialized analyzer instance.

    Example:
        >>> analyzer = create_analyzer("face", device="cuda:0")
    """
    AnalyzerClass = load_analyzer(name)
    return AnalyzerClass(**kwargs)


class PluginRegistry:
    """Registry for managing loaded plugins.

    Provides caching and lifecycle management for plugins.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_analyzer("face", FaceAnalyzer)
        >>> analyzer = registry.create_analyzer("face")
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._analyzers: Dict[str, Type[Module]] = {}
        self._fusions: Dict[str, Type] = {}
        self._instances: Dict[str, Module] = {}

    def register_analyzer(
        self,
        name: str,
        analyzer_class: Type[Module],
    ) -> None:
        """Register an analyzer class.

        Args:
            name: Name to register under.
            analyzer_class: The analyzer class.
        """
        self._analyzers[name] = analyzer_class

    def register_fusion(self, name: str, fusion_class: Type) -> None:
        """Register a fusion class.

        Args:
            name: Name to register under.
            fusion_class: The fusion class.
        """
        self._fusions[name] = fusion_class

    def get_analyzer_class(self, name: str) -> Type[Module]:
        """Get a registered analyzer class.

        Args:
            name: The analyzer name.

        Returns:
            The analyzer class.

        Raises:
            KeyError: If not registered.
        """
        if name in self._analyzers:
            return self._analyzers[name]
        # Fall back to entry point discovery
        return load_analyzer(name)

    def create_analyzer(
        self,
        name: str,
        singleton: bool = False,
        **kwargs,
    ) -> Module:
        """Create an analyzer instance.

        Args:
            name: The analyzer name.
            singleton: If True, return cached instance.
            **kwargs: Constructor arguments.

        Returns:
            Analyzer instance.
        """
        if singleton and name in self._instances:
            return self._instances[name]

        AnalyzerClass = self.get_analyzer_class(name)
        instance = AnalyzerClass(**kwargs)

        if singleton:
            self._instances[name] = instance

        return instance

    def list_analyzers(self) -> list[str]:
        """List all available analyzer names.

        Returns:
            List of analyzer names (registered + discovered).
        """
        discovered = set(discover_analyzers().keys())
        registered = set(self._analyzers.keys())
        return sorted(discovered | registered)

    def list_fusions(self) -> list[str]:
        """List all available fusion names.

        Returns:
            List of fusion names (registered + discovered).
        """
        discovered = set(discover_fusions().keys())
        registered = set(self._fusions.keys())
        return sorted(discovered | registered)

    def cleanup(self) -> None:
        """Clean up all cached instances."""
        for instance in self._instances.values():
            try:
                instance.cleanup()
            except Exception:
                pass
        self._instances.clear()
