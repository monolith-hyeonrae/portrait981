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
from typing import Dict, List, TYPE_CHECKING

from visualpath.core.capabilities import Capability

if TYPE_CHECKING:
    from visualpath.core.module import Module

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityReport:
    """Result of a compatibility check.

    Attributes:
        valid: True if no errors were found.
        warnings: Non-fatal issues (e.g. GPU memory close to limit).
        errors: Fatal issues (currently unused — kept for future use).
        resource_conflicts: Groups with >1 group active → list of module names.
        estimated_gpu_mb: Sum of declared gpu_memory_mb across all GPU modules.
    """

    valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    resource_conflicts: Dict[str, List[str]] = field(default_factory=dict)
    estimated_gpu_mb: int = 0


def check_compatibility(modules: List["Module"]) -> CompatibilityReport:
    """Check compatibility of a module combination.

    Inspects ``module.capabilities`` for each module and reports:
    - Resource group conflicts (e.g. onnxruntime + torch in same process)
    - Total GPU memory estimate
    - Distributed-execution readiness

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

    # GPU memory warning
    if total_gpu_mb > 0:
        report.warnings.append(
            f"Estimated GPU memory: {total_gpu_mb}MB across "
            f"{sum(1 for m in modules if Capability.GPU in m.capabilities.flags)} modules."
        )

    # Distributed readiness check
    non_distributable = [
        mod.name
        for mod in modules
        if Capability.NEEDS_ZERO_COPY in mod.capabilities.flags
    ]
    if non_distributable:
        report.warnings.append(
            f"Modules require zero-copy (not distributable): {non_distributable}"
        )

    # valid stays True — compatibility issues are warnings, not errors
    return report


__all__ = ["CompatibilityReport", "check_compatibility"]
