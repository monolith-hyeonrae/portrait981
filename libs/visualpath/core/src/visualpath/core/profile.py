"""Execution profiles for lite vs platform configurations.

Profiles are configuration presets — they don't create new execution paths.
The same FlowGraph runs through the same backends; profiles simply
configure isolation, observability, and warmup defaults.

Example:
    >>> from visualpath.core.profile import ExecutionProfile, ProfileName
    >>> profile = ExecutionProfile.lite()
    >>> # or
    >>> profile = ExecutionProfile.platform()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

from visualpath.core.capabilities import Capability
from visualpath.core.isolation import IsolationConfig, IsolationLevel

if TYPE_CHECKING:
    from visualpath.core.module import Module


class ProfileName(str, Enum):
    """Named execution profiles."""

    LITE = "lite"
    PLATFORM = "platform"


@dataclass(frozen=True)
class ExecutionProfile:
    """Configuration preset for pipeline execution.

    Attributes:
        name: Profile identifier.
        isolation_default: Default isolation level for modules.
        enable_observability: Whether to enable tracing/observability.
        enable_warmup: Whether to call warmup() on modules.
        enable_compat_check: Whether to run compatibility checks.
        backend: Default backend name ("simple", "pathway", "worker").
        daemon: Whether to run in daemon mode (long-lived process).
    """

    name: ProfileName
    isolation_default: IsolationLevel = IsolationLevel.INLINE
    enable_observability: bool = False
    enable_warmup: bool = True
    enable_compat_check: bool = True
    backend: str = "simple"
    daemon: bool = False

    @classmethod
    def lite(cls) -> "ExecutionProfile":
        """Lite profile: minimal overhead, inline execution."""
        return cls(
            name=ProfileName.LITE,
            isolation_default=IsolationLevel.INLINE,
            enable_observability=False,
            enable_warmup=True,
            enable_compat_check=True,
            backend="simple",
            daemon=False,
        )

    @classmethod
    def platform(cls) -> "ExecutionProfile":
        """Platform profile: full isolation, observability, streaming."""
        return cls(
            name=ProfileName.PLATFORM,
            isolation_default=IsolationLevel.PROCESS,
            enable_observability=True,
            enable_warmup=True,
            enable_compat_check=True,
            backend="pathway",
            daemon=False,
        )

    @classmethod
    def from_name(cls, name: str) -> "ExecutionProfile":
        """Create a profile by name.

        Args:
            name: "lite" or "platform".

        Returns:
            ExecutionProfile preset.

        Raises:
            ValueError: If name is unknown.
        """
        if name == "lite":
            return cls.lite()
        elif name == "platform":
            return cls.platform()
        else:
            raise ValueError(
                f"Unknown profile: {name}. Use 'lite' or 'platform'."
            )


def resolve_profile(
    profile: ExecutionProfile,
    modules: List["Module"],
) -> IsolationConfig:
    """Resolve a profile into an IsolationConfig for the given modules.

    Uses module capabilities to determine per-module overrides:
    - Platform profile: GPU modules get PROCESS isolation automatically.
    - Lite profile: all modules stay INLINE.

    Args:
        profile: Execution profile to resolve.
        modules: List of Module instances.

    Returns:
        IsolationConfig with appropriate overrides.
    """
    overrides: Dict[str, IsolationLevel] = {}

    if profile.name == ProfileName.PLATFORM:
        for mod in modules:
            caps = mod.capabilities
            if Capability.GPU in caps.flags:
                overrides[mod.name] = IsolationLevel.PROCESS

    return IsolationConfig(
        default_level=profile.isolation_default,
        overrides=overrides,
    )


__all__ = ["ProfileName", "ExecutionProfile", "resolve_profile"]
