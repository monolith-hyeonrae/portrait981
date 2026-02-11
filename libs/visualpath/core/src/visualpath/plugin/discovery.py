"""Plugin discovery system for visualpath.

This module provides infrastructure for discovering and loading
module plugins via Python entry points.

Plugins register themselves using the `visualpath.modules` entry point
group in their pyproject.toml:

```toml
[project.entry-points."visualpath.modules"]
"face.detect" = "myplugin.analyzers:FaceDetectionAnalyzer"
"highlight" = "myplugin.fusion:HighlightFusion"
```

Example:
    >>> from visualpath.plugin import discover_modules, load_module
    >>>
    >>> # Discover all available module plugins
    >>> modules = discover_modules()
    >>> for name, entry_point in modules.items():
    ...     print(f"Found module: {name}")
    >>>
    >>> # Load a specific module
    >>> FaceDetector = load_module("face.detect")
    >>> detector = FaceDetector()
"""

import sys
from typing import Dict, Optional, Type, Any

from visualpath.core.module import Module

# Entry point group name (unified)
MODULES_GROUP = "visualpath.modules"

# Legacy group names (kept for backward compatibility during transition)
ANALYZERS_GROUP = MODULES_GROUP
FUSIONS_GROUP = MODULES_GROUP


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


def discover_modules() -> Dict[str, Any]:
    """Discover all available module plugins (analyzers + fusions).

    Scans the `visualpath.modules` entry point group for registered modules.

    Returns:
        Dict mapping module names to their entry points.

    Example:
        >>> modules = discover_modules()
        >>> print(list(modules.keys()))
        ['face.detect', 'face.expression', 'highlight']
    """
    return _get_entry_points(MODULES_GROUP)


def load_module(name: str) -> Type[Module]:
    """Load a module class by name.

    Args:
        name: The registered name of the module.

    Returns:
        The module class.

    Raises:
        KeyError: If no module with the given name is registered.
        ImportError: If the module cannot be loaded.

    Example:
        >>> FaceDetector = load_module("face.detect")
        >>> detector = FaceDetector()
    """
    modules = discover_modules()
    if name not in modules:
        raise KeyError(
            f"Unknown module: '{name}'. "
            f"Available: {list(modules.keys())}"
        )
    return modules[name].load()


def create_module(name: str, **kwargs) -> Module:
    """Create a module instance by name.

    Convenience function that loads and instantiates a module.

    Args:
        name: The registered name of the module.
        **kwargs: Arguments passed to the module constructor.

    Returns:
        An initialized module instance.

    Example:
        >>> detector = create_module("face.detect", device="cuda:0")
    """
    ModuleClass = load_module(name)
    return ModuleClass(**kwargs)


# =============================================================================
# Aliases (backward compatibility)
# =============================================================================

def discover_analyzers() -> Dict[str, Any]:
    """Discover all available modules. Alias for discover_modules()."""
    return discover_modules()


def discover_fusions() -> Dict[str, Any]:
    """Discover all available modules. Alias for discover_modules()."""
    return discover_modules()


def load_analyzer(name: str) -> Type[Module]:
    """Load a module class by name. Alias for load_module()."""
    return load_module(name)


def load_fusion(name: str) -> Type:
    """Load a module class by name. Alias for load_module()."""
    return load_module(name)


def create_analyzer(name: str, **kwargs) -> Module:
    """Create a module instance by name. Alias for create_module()."""
    return create_module(name, **kwargs)


class PluginRegistry:
    """Registry for managing loaded plugins.

    Provides caching and lifecycle management for plugins.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register("face.detect", FaceDetectionAnalyzer)
        >>> module = registry.create("face.detect")
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._modules: Dict[str, Type[Module]] = {}
        self._instances: Dict[str, Module] = {}

    def register(self, name: str, module_class: Type[Module]) -> None:
        """Register a module class.

        Args:
            name: Name to register under.
            module_class: The module class.
        """
        self._modules[name] = module_class

    # Aliases for backward compatibility
    def register_analyzer(self, name: str, analyzer_class: Type[Module]) -> None:
        """Register a module class. Alias for register()."""
        self.register(name, analyzer_class)

    def register_fusion(self, name: str, fusion_class: Type) -> None:
        """Register a module class. Alias for register()."""
        self._modules[name] = fusion_class

    def get_class(self, name: str) -> Type[Module]:
        """Get a registered module class.

        Args:
            name: The module name.

        Returns:
            The module class.

        Raises:
            KeyError: If not registered.
        """
        if name in self._modules:
            return self._modules[name]
        # Fall back to entry point discovery
        return load_module(name)

    # Alias
    def get_analyzer_class(self, name: str) -> Type[Module]:
        """Get a registered module class. Alias for get_class()."""
        return self.get_class(name)

    def create(
        self,
        name: str,
        singleton: bool = False,
        **kwargs,
    ) -> Module:
        """Create a module instance.

        Args:
            name: The module name.
            singleton: If True, return cached instance.
            **kwargs: Constructor arguments.

        Returns:
            Module instance.
        """
        if singleton and name in self._instances:
            return self._instances[name]

        ModuleClass = self.get_class(name)
        instance = ModuleClass(**kwargs)

        if singleton:
            self._instances[name] = instance

        return instance

    # Alias
    def create_analyzer(
        self,
        name: str,
        singleton: bool = False,
        **kwargs,
    ) -> Module:
        """Create a module instance. Alias for create()."""
        return self.create(name, singleton=singleton, **kwargs)

    def list_modules(self) -> list[str]:
        """List all available module names.

        Returns:
            List of module names (registered + discovered).
        """
        discovered = set(discover_modules().keys())
        registered = set(self._modules.keys())
        return sorted(discovered | registered)

    # Aliases
    def list_analyzers(self) -> list[str]:
        """List all available module names. Alias for list_modules()."""
        return self.list_modules()

    def list_fusions(self) -> list[str]:
        """List all available module names. Alias for list_modules()."""
        return self.list_modules()

    def cleanup(self) -> None:
        """Clean up all cached instances."""
        for instance in self._instances.values():
            try:
                instance.cleanup()
            except Exception:
                pass
        self._instances.clear()
