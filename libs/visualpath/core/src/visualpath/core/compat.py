"""Compatibility checker for module combinations.

Checks resource group conflicts, GPU memory budgets, and
distributed-execution readiness before graph execution.

Results are always warnings, never errors — existing pipelines
continue to work even if capabilities are not declared.

Example:
    >>> from visualpath.core.compat import check_compatibility
    >>> report = check_compatibility(modules)
    >>> if not report.valid:
    ...     for w in report.warnings:
    ...         logger.warning(w)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from visualpath.core.capabilities import Capability
from visualpath.core.isolation import IsolationConfig, IsolationLevel

if TYPE_CHECKING:
    from visualpath.core.module import Module

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityReport:
    """Result of a compatibility check.

    Attributes:
        valid: True if no errors were found.
        warnings: Actionable issues (e.g. resource group conflicts).
        info: Informational messages (e.g. GPU memory estimates).
        errors: Fatal issues (currently unused — kept for future use).
        resource_conflicts: Groups with >1 group active → list of module names.
        estimated_gpu_mb: Sum of declared gpu_memory_mb across all GPU modules.
    """

    valid: bool = True
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    resource_conflicts: Dict[str, List[str]] = field(default_factory=dict)
    estimated_gpu_mb: int = 0


def check_compatibility(modules: List["Module"]) -> CompatibilityReport:
    """Check compatibility of a module combination.

    Inspects ``module.capabilities`` for each module and reports:
    - Resource group conflicts (e.g. onnxruntime + torch in same process)
    - Total GPU memory estimate
    - Distributed-execution readiness

    Modules without a ``capabilities`` property are silently skipped.

    Args:
        modules: List of Module instances to check.

    Returns:
        CompatibilityReport with warnings (never raises).
    """
    report = CompatibilityReport()

    # Collect per-group module names
    group_members: Dict[str, List[str]] = defaultdict(list)
    total_gpu_mb = 0

    for mod in modules:
        if not hasattr(mod, "capabilities"):
            continue
        caps = mod.capabilities
        total_gpu_mb += caps.gpu_memory_mb

        for group in caps.resource_groups:
            group_members[group].append(mod.name)

    report.estimated_gpu_mb = total_gpu_mb

    # Detect resource group conflicts: 2+ groups with members
    active_groups = {g: members for g, members in group_members.items() if members}
    if len(active_groups) >= 2:
        report.resource_conflicts = dict(active_groups)
        groups_str = ", ".join(
            f"{g}=[{', '.join(ms)}]" for g, ms in active_groups.items()
        )
        report.warnings.append(
            f"Resource group conflict: {groups_str}. "
            "Consider process isolation for minority group."
        )

    # GPU memory estimate (informational, not actionable)
    if total_gpu_mb > 0:
        report.info.append(
            f"Estimated GPU memory: {total_gpu_mb}MB across "
            f"{sum(1 for m in modules if hasattr(m, 'capabilities') and Capability.GPU in m.capabilities.flags)} modules."
        )

    # Distributed readiness check
    non_distributable = [
        mod.name
        for mod in modules
        if hasattr(mod, "capabilities")
        and Capability.NEEDS_ZERO_COPY in mod.capabilities.flags
    ]
    if non_distributable:
        report.warnings.append(
            f"Modules require zero-copy (not distributable): {non_distributable}"
        )

    # valid stays True — compatibility issues are warnings, not errors
    return report


def build_conflict_isolation(
    modules: List["Module"],
) -> Optional[IsolationConfig]:
    """Build IsolationConfig from resource group conflicts.

    Uses :func:`check_compatibility` to detect conflicts between
    resource groups (e.g. onnxruntime vs torch), then configures
    the minority group for PROCESS isolation.

    Requires pyzmq for IPC — returns None if unavailable.

    Args:
        modules: List of Module instances to check.

    Returns:
        IsolationConfig if isolation is needed, None otherwise.
    """
    try:
        import zmq  # noqa: F401
    except ImportError:
        logger.debug("pyzmq not available, skipping conflict isolation")
        return None

    report = check_compatibility(modules)
    if not report.resource_conflicts:
        return None

    # Isolate the smallest group (fewest modules)
    minority_group = min(
        report.resource_conflicts,
        key=lambda g: len(report.resource_conflicts[g]),
    )
    isolated_names = set(report.resource_conflicts[minority_group])

    logger.info(
        "Resource conflict detected: groups %s. Isolating %s to subprocess.",
        list(report.resource_conflicts.keys()), isolated_names,
    )

    return IsolationConfig(
        default_level=IsolationLevel.INLINE,
        overrides={name: IsolationLevel.PROCESS for name in isolated_names},
    )


def build_distributed_config(
    modules: List["Module"],
    *,
    venv_paths: Optional[Dict[str, str]] = None,
) -> IsolationConfig:
    """Build IsolationConfig for distributed execution.

    GPU-capable modules get PROCESS isolation by default.
    When ``venv_paths`` provides a path for a module, it uses VENV
    isolation instead. Non-GPU modules stay INLINE.

    Args:
        modules: List of Module instances.
        venv_paths: Optional mapping from module name to venv path.

    Returns:
        IsolationConfig for distributed execution.
    """
    venv_paths = venv_paths or {}
    overrides: Dict[str, IsolationLevel] = {}
    resolved_venvs: Dict[str, str] = {}

    for mod in modules:
        if not hasattr(mod, "capabilities"):
            continue
        if Capability.GPU not in mod.capabilities.flags:
            continue
        venv = venv_paths.get(mod.name)
        if venv:
            overrides[mod.name] = IsolationLevel.VENV
            resolved_venvs[mod.name] = venv
        else:
            overrides[mod.name] = IsolationLevel.PROCESS

    return IsolationConfig(
        default_level=IsolationLevel.INLINE,
        overrides=overrides,
        venv_paths=resolved_venvs,
    )


__all__ = [
    "CompatibilityReport",
    "check_compatibility",
    "build_conflict_isolation",
    "build_distributed_config",
]
