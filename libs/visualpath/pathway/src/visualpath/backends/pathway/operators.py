"""Pathway operators for analyzer execution.

This module provides operator wrappers that integrate visualpath's
analyzers with Pathway's streaming operators.

Pure Python functions (create_analyzer_udf, create_multi_analyzer_udf)
can be used independently.

Pathway-specific functions (apply_analyzers) require Pathway installed
and operate on pw.Table with PyObjectWrapper columns.
"""

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
    """
    frame_id: int
    t_ns: int
    source: str
    observation: Optional["Observation"]


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


def create_multi_analyzer_udf(analyzers: List["Module"]):
    """Create a callable that runs multiple analyzers on each frame.

    Analyzers are topologically sorted by dependencies, then run
    in order with dependency resolution: each analyzer receives
    observations from its dependencies via the deps parameter.

    Args:
        analyzers: List of analyzers to run.

    Returns:
        A function that takes a Frame and returns list of AnalyzerResults.
    """
    sorted_analyzers = _toposort_modules(analyzers)

    def analyze_all(frame: "Frame") -> List[AnalyzerResult]:
        """Run all analyzers on a frame with deps accumulation."""
        results = []
        deps: Dict[str, "Observation"] = {}
        for analyzer in sorted_analyzers:
            try:
                analyzer_deps = None
                all_dep_names = list(getattr(analyzer, 'depends', [])) + list(getattr(analyzer, 'optional_depends', []))
                if all_dep_names:
                    analyzer_deps = {
                        name: deps[name]
                        for name in all_dep_names
                        if name in deps
                    }
                observation = analyzer.process(frame, analyzer_deps)
                results.append(AnalyzerResult(
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    source=analyzer.name,
                    observation=observation,
                ))
                if observation is not None:
                    deps[analyzer.name] = observation
            except Exception:
                pass
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
    "create_multi_analyzer_udf",
]

if PATHWAY_AVAILABLE:
    __all__.append("apply_analyzers")
