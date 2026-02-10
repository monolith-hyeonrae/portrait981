"""Pathway operators for analyzer execution.

This module provides operator wrappers that integrate visualpath's
analyzers with Pathway's streaming operators.

Pure Python functions (create_analyzer_udf, create_multi_analyzer_udf)
can be used independently.

Pathway-specific functions (apply_analyzers) require Pathway installed
and operate on pw.Table with PyObjectWrapper columns.
"""

import time as _time_mod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.observation import Observation
    from visualpath.core.module import Module

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


@dataclass
class AnalyzerResult:
    """Result from analyzer execution in Pathway.

    Attributes:
        frame_id: Frame identifier.
        t_ns: Timestamp in nanoseconds.
        source: Analyzer name.
        observation: The Observation, or None if filtered.
        elapsed_ms: Wall-clock time for this analyzer invocation.
    """
    frame_id: int
    t_ns: int
    source: str
    observation: Optional["Observation"]
    elapsed_ms: float = 0.0


def create_analyzer_udf(
    analyzer: "Module",
    deps: Optional[Dict[str, "Observation"]] = None,
):
    """Create a callable for a single analyzer.

    Args:
        analyzer: The analyzer to wrap.
        deps: Optional pre-built deps for this analyzer.

    Returns:
        A function that takes a Frame and returns list of AnalyzerResult.
    """
    def analyze_fn(frame: "Frame") -> List[AnalyzerResult]:
        """Analyze observations from a frame."""
        try:
            analyzer_deps = None
            all_dep_names = list(getattr(analyzer, 'depends', [])) + list(getattr(analyzer, 'optional_depends', []))
            if all_dep_names and deps:
                analyzer_deps = {
                    name: deps[name]
                    for name in all_dep_names
                    if name in deps
                }
            observation = analyzer.process(frame, analyzer_deps)
            return [AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=observation,
            )]
        except Exception:
            return []

    return analyze_fn


def _toposort_modules(modules: List["Module"]) -> List["Module"]:
    """Topologically sort modules by depends/optional_depends."""
    from collections import defaultdict, deque

    name_to_mod = {m.name: m for m in modules}
    available = set(name_to_mod.keys())
    in_degree = {m.name: 0 for m in modules}
    dependents = defaultdict(list)

    for m in modules:
        for dep in list(getattr(m, 'depends', [])) + list(getattr(m, 'optional_depends', [])):
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
        seen = {m.name for m in result}
        result.extend(m for m in modules if m.name not in seen)

    return result


def _group_by_dependency_level(sorted_modules: List["Module"]) -> List[List["Module"]]:
    """Group topologically sorted modules by dependency level.

    Level 0: modules with no dependencies (in the current set)
    Level N: modules whose dependencies are all in levels 0..N-1

    Returns:
        List of lists — each inner list is a set of independent modules
        that can execute in parallel.
    """
    available = {m.name for m in sorted_modules}
    name_to_mod = {m.name: m for m in sorted_modules}
    levels_map: Dict[str, int] = {}

    for m in sorted_modules:
        all_deps = list(getattr(m, 'depends', [])) + list(getattr(m, 'optional_depends', []))
        relevant_deps = [d for d in all_deps if d in available]
        if not relevant_deps:
            levels_map[m.name] = 0
        else:
            levels_map[m.name] = max(levels_map.get(d, 0) for d in relevant_deps) + 1

    max_level = max(levels_map.values()) if levels_map else 0
    groups: List[List["Module"]] = []
    for level in range(max_level + 1):
        group = [name_to_mod[n] for n, lv in levels_map.items() if lv == level]
        if group:
            groups.append(group)
    return groups


def create_single_analyzer_udf(analyzer: "Module"):
    """Create a callable for an independent analyzer (no deps).

    Returns:
        A function that takes a Frame and returns list of AnalyzerResult (1 item).
    """
    def analyze_fn(frame: "Frame") -> List[AnalyzerResult]:
        t0 = _time_mod.perf_counter()
        try:
            observation = analyzer.process(frame, None)
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
            return [AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=observation,
                elapsed_ms=elapsed_ms,
            )]
        except Exception:
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
            return [AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=None,
                elapsed_ms=elapsed_ms,
            )]

    return analyze_fn


def create_dep_analyzer_udf(analyzer: "Module"):
    """Create a callable for an analyzer with dependencies.

    The UDF receives a frame and upstream_results (list of AnalyzerResult
    from parent UDFs). It extracts deps from upstream_results and calls
    analyzer.process(frame, deps).

    Returns:
        A function(frame, upstream_results) -> list of AnalyzerResult (1 item).
        upstream_results is a list of AnalyzerResult from upstream tables.
    """
    all_dep_names = list(getattr(analyzer, 'depends', []) or []) + \
                    list(getattr(analyzer, 'optional_depends', []) or [])

    def analyze_fn(
        frame: "Frame",
        upstream_results: List[AnalyzerResult],
    ) -> List[AnalyzerResult]:
        t0 = _time_mod.perf_counter()
        try:
            # Build deps dict from upstream results
            deps = {}
            if all_dep_names and upstream_results:
                for r in upstream_results:
                    if r.observation is not None and r.source in all_dep_names:
                        deps[r.source] = r.observation
            analyzer_deps = deps if deps else None
            observation = analyzer.process(frame, analyzer_deps)
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
            return [AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=observation,
                elapsed_ms=elapsed_ms,
            )]
        except Exception:
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
            return [AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=None,
                elapsed_ms=elapsed_ms,
            )]

    return analyze_fn


def create_multi_analyzer_udf(analyzers: List["Module"]):
    """Create a callable that runs multiple analyzers on each frame.

    Analyzers are topologically sorted by dependencies, grouped by
    dependency level, then run level-by-level. Independent analyzers
    within the same level execute in parallel via ThreadPoolExecutor.

    Args:
        analyzers: List of analyzers to run.

    Returns:
        A function that takes a Frame and returns list of AnalyzerResults.
    """
    sorted_analyzers = _toposort_modules(analyzers)
    levels = _group_by_dependency_level(sorted_analyzers)

    max_parallel = max(len(level) for level in levels) if levels else 1
    pool = ThreadPoolExecutor(max_workers=max_parallel) if max_parallel > 1 else None

    def _run_one(analyzer, frame, deps_snapshot):
        """Run a single analyzer (may execute in a worker thread)."""
        t0 = _time_mod.perf_counter()
        try:
            analyzer_deps = None
            all_dep_names = list(getattr(analyzer, 'depends', [])) + \
                            list(getattr(analyzer, 'optional_depends', []))
            if all_dep_names:
                analyzer_deps = {
                    name: deps_snapshot[name]
                    for name in all_dep_names
                    if name in deps_snapshot
                }
            observation = analyzer.process(frame, analyzer_deps)
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
            return AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=observation,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
            return AnalyzerResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=analyzer.name,
                observation=None,
                elapsed_ms=elapsed_ms,
            )

    def analyze_all(frame: "Frame") -> List[AnalyzerResult]:
        """Run all analyzers on a frame with level-based parallelism."""
        results: List[AnalyzerResult] = []
        deps: Dict[str, "Observation"] = {}
        for level in levels:
            if len(level) == 1 or pool is None:
                for analyzer in level:
                    r = _run_one(analyzer, frame, deps)
                    results.append(r)
                    if r.observation is not None:
                        deps[analyzer.name] = r.observation
            else:
                deps_snapshot = dict(deps)
                futures = {pool.submit(_run_one, a, frame, deps_snapshot): a for a in level}
                for future in futures:
                    r = future.result()
                    results.append(r)
                    if r.observation is not None:
                        deps[r.source] = r.observation
        return results

    return analyze_all


if PATHWAY_AVAILABLE:
    def apply_analyzers(
        frames_table: "pw.Table",
        analyzers: List["Module"],
    ) -> "pw.Table":
        """Apply analyzers to a Pathway frames table.

        Runs all analyzers on each frame via a @pw.udf, wrapping
        results in PyObjectWrapper for Pathway transport.

        Args:
            frames_table: Table with FrameSchema (frame column is PyObjectWrapper).
            analyzers: List of analyzers.

        Returns:
            Table with columns: frame_id, t_ns, observations (PyObjectWrapper).
        """
        raw_udf = create_multi_analyzer_udf(analyzers)

        @pw.udf
        def analyze_all_udf(
            frame_wrapped: pw.PyObjectWrapper,
        ) -> pw.PyObjectWrapper:
            frame = frame_wrapped.value
            results = raw_udf(frame)
            return pw.PyObjectWrapper(results)

        return frames_table.select(
            frame_id=pw.this.frame_id,
            t_ns=pw.this.t_ns,
            observations=analyze_all_udf(pw.this.frame),
        )


__all__ = [
    "AnalyzerResult",
    "create_analyzer_udf",
    "create_single_analyzer_udf",
    "create_dep_analyzer_udf",
    "create_multi_analyzer_udf",
    "_toposort_modules",
    "_group_by_dependency_level",
]

if PATHWAY_AVAILABLE:
    __all__.append("apply_analyzers")
