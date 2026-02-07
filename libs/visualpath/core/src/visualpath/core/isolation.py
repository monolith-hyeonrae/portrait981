"""Isolation levels for module execution.

IsolationLevel controls how modules are executed, from inline (same process)
to fully isolated containers. Each level offers different trade-offs between
performance and isolation.

Example:
    >>> from visualpath.core import IsolationLevel, IsolationConfig
    >>>
    >>> # Plugin declares recommended isolation
    >>> class HeavyMLModule(Module):
    ...     # Some ML modules may prefer VENV isolation
    ...     pass
    >>>
    >>> # Config can override
    >>> config = IsolationConfig(
    ...     default_level=IsolationLevel.PROCESS,
    ...     overrides={"heavy_ml": IsolationLevel.INLINE},  # For debugging
    ... )
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional


class IsolationLevel(IntEnum):
    """Isolation level for module execution.

    Higher levels provide more isolation but incur more overhead.
    Choose the appropriate level based on the module's requirements.

    Levels:
        INLINE: Same process, same thread. Best for simple/fast modules.
        THREAD: Same process, different thread. Good for I/O-bound work.
        PROCESS: Same venv, different process. Provides memory isolation.
        VENV: Different venv, different process. Resolves dependency conflicts.
        CONTAINER: Full container isolation. Maximum isolation.
    """
    INLINE = 0      # Same process, same thread
    THREAD = 1      # Same process, different thread
    PROCESS = 2     # Same venv, different process
    VENV = 3        # Different venv, different process
    CONTAINER = 4   # Full container isolation

    @classmethod
    def from_string(cls, s: str) -> "IsolationLevel":
        """Parse isolation level from string.

        Args:
            s: String like "inline", "thread", "process", "venv", "container"

        Returns:
            Corresponding IsolationLevel.

        Raises:
            ValueError: If string is not a valid level name.
        """
        mapping = {
            "inline": cls.INLINE,
            "thread": cls.THREAD,
            "process": cls.PROCESS,
            "venv": cls.VENV,
            "container": cls.CONTAINER,
        }
        s_lower = s.lower()
        if s_lower not in mapping:
            raise ValueError(
                f"Unknown isolation level: {s}. "
                f"Valid levels: {', '.join(mapping.keys())}"
            )
        return mapping[s_lower]


@dataclass
class IsolationConfig:
    """Configuration for isolation levels.

    Allows setting a default isolation level and per-module overrides.
    Overrides take precedence over both the default and the module's
    recommended level.

    Attributes:
        default_level: Default isolation level for all modules.
        overrides: Per-module isolation level overrides.
        venv_paths: Mapping from module name to venv path (for VENV level).
        container_images: Mapping from module name to container image (for CONTAINER level).

    Example:
        >>> config = IsolationConfig(
        ...     default_level=IsolationLevel.PROCESS,
        ...     overrides={
        ...         "face": IsolationLevel.VENV,  # Heavy ML
        ...         "quality": IsolationLevel.INLINE,  # Simple OpenCV
        ...     },
        ...     venv_paths={
        ...         "face": "/opt/venvs/face",
        ...     },
        ... )
    """

    default_level: IsolationLevel = IsolationLevel.INLINE
    overrides: Dict[str, IsolationLevel] = field(default_factory=dict)
    venv_paths: Dict[str, str] = field(default_factory=dict)
    container_images: Dict[str, str] = field(default_factory=dict)

    def get_level(
        self,
        module_name: str,
        recommended: Optional[IsolationLevel] = None,
    ) -> IsolationLevel:
        """Get the effective isolation level for a module.

        Priority (highest to lowest):
        1. Override from config
        2. Default from config
        3. Recommended from module (if higher than default)

        Args:
            module_name: Name of the module.
            recommended: Module's recommended isolation level.

        Returns:
            Effective isolation level to use.
        """
        # Check for override first
        if module_name in self.overrides:
            return self.overrides[module_name]

        # Use default, but respect recommended if higher
        if recommended is not None and recommended > self.default_level:
            return recommended

        return self.default_level

    def get_venv_path(self, module_name: str) -> Optional[str]:
        """Get the venv path for a module.

        Args:
            module_name: Name of the module.

        Returns:
            Path to venv, or None if not configured.
        """
        return self.venv_paths.get(module_name)

    def get_container_image(self, module_name: str) -> Optional[str]:
        """Get the container image for a module.

        Args:
            module_name: Name of the module.

        Returns:
            Container image name, or None if not configured.
        """
        return self.container_images.get(module_name)
