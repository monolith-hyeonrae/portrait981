"""Module dependency graph utilities."""

from collections import defaultdict, deque
from typing import List, Sequence


def toposort_modules(modules: Sequence) -> List:
    """Topologically sort modules based on their depends/optional_depends.

    Uses Kahn's algorithm. Only considers dependencies that are
    present within the given set of modules (external deps are ignored).
    Falls back to original order for any unresolvable cycles.

    Args:
        modules: Sequence of module-like objects with ``name``,
                 ``depends`` and optionally ``optional_depends`` attributes.

    Returns:
        List of modules in dependency order.
    """
    if not modules:
        return []

    name_to_mod = {m.name: m for m in modules}
    available = set(name_to_mod.keys())
    in_degree = {m.name: 0 for m in modules}
    dependents: dict[str, list[str]] = defaultdict(list)

    for m in modules:
        for dep in list(getattr(m, "depends", [])) + list(
            getattr(m, "optional_depends", [])
        ):
            if dep in available:
                in_degree[m.name] += 1
                dependents[dep].append(m.name)

    queue = deque(n for n, d in in_degree.items() if d == 0)
    result = []
    while queue:
        name = queue.popleft()
        result.append(name_to_mod[name])
        for dep_name in dependents[name]:
            in_degree[dep_name] -= 1
            if in_degree[dep_name] == 0:
                queue.append(dep_name)

    if len(result) != len(modules):
        # Cycle detected - append remaining modules in original order
        seen = {m.name for m in result}
        result.extend(m for m in modules if m.name not in seen)

    return result
