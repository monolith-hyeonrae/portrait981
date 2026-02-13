"""WorkerBackend — FlowGraph execution with isolation support.

WorkerBackend reads ModuleSpec.isolation from the FlowGraph and
wraps modules that need isolation in WorkerModule before delegating
to SimpleBackend for actual execution.

The key insight: SimpleInterpreter calls module.process() on each
module. WorkerModule implements Module, so the interpreter doesn't
need to know about isolation. WorkerBackend just swaps the modules
before execution starts.
"""

import logging
from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

from visualpath.backends.base import ExecutionBackend, PipelineResult
from visualpath.core.isolation import IsolationConfig, IsolationLevel

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.module import Module
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData

logger = logging.getLogger(__name__)


class WorkerBackend(ExecutionBackend):
    """Execution backend that handles module isolation transparently.

    WorkerBackend inspects each PathNode's ModuleSpec for isolation
    configuration. Modules that need PROCESS or VENV isolation are
    wrapped in WorkerModule (which delegates to BaseWorker via ZMQ).
    The wrapped graph is then executed by SimpleBackend.

    This allows the single execution path:
        ms.run() → build_graph(isolation) → WorkerBackend.execute()

    Example:
        >>> from visualpath.backends.worker import WorkerBackend
        >>> backend = WorkerBackend()
        >>> result = backend.execute(frames, graph)
    """

    def __init__(self, batch_size: int = 1):
        self._batch_size = max(1, batch_size)

    @property
    def name(self) -> str:
        return "WorkerBackend"

    def execute(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        *,
        on_frame: Optional[Callable[["Frame", List["FlowData"]], bool]] = None,
    ) -> PipelineResult:
        """Execute a FlowGraph with isolation support.

        1. Scan graph for ModuleSpec nodes with isolation config
        2. Wrap isolated modules in WorkerModule
        3. Build a new graph with wrapped modules
        4. Delegate to SimpleBackend for execution

        Args:
            frames: Iterator of Frame objects.
            graph: FlowGraph (may contain ModuleSpec.isolation).
            on_frame: Optional per-frame callback. See ExecutionBackend.execute().

        Returns:
            PipelineResult with triggers and frame count.
        """
        wrapped_graph = self._wrap_isolated_modules(graph)

        from visualpath.backends.simple import SimpleBackend
        return SimpleBackend(batch_size=self._batch_size).execute(
            frames, wrapped_graph, on_frame=on_frame,
        )

    def _wrap_isolated_modules(self, graph: "FlowGraph") -> "FlowGraph":
        """Create a new graph with isolated modules replaced by WorkerModule.

        Scans PathNodes for ModuleSpec.isolation. For each module whose
        isolation level is > INLINE, creates a WorkerModule wrapper.

        Args:
            graph: Original FlowGraph.

        Returns:
            New FlowGraph with wrapped modules (or the original if no wrapping needed).
        """
        from visualpath.flow.specs import ModuleSpec
        from visualpath.flow.nodes.path import PathNode
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.graph import FlowGraph

        needs_wrapping = False

        # Check if any node needs wrapping
        for node in graph.nodes.values():
            spec = node.spec
            if isinstance(spec, ModuleSpec) and spec.isolation is not None:
                needs_wrapping = True
                break

        if not needs_wrapping:
            return graph

        # Build new graph with wrapped modules
        new_graph = FlowGraph(entry_node=graph.entry_node)

        for node_name, node in graph.nodes.items():
            spec = node.spec
            if isinstance(spec, ModuleSpec) and spec.isolation is not None:
                wrapped_modules = self._wrap_modules(
                    list(spec.modules), spec.isolation
                )
                new_node = PathNode(
                    name=node_name,
                    modules=wrapped_modules,
                    parallel=spec.parallel,
                    join_window_ns=spec.join_window_ns,
                    # No isolation on new node — already wrapped
                )
                new_graph.add_node(new_node)
            else:
                new_graph.add_node(node)

        # Copy edges
        for edge in graph.edges:
            new_graph.add_edge(edge.source, edge.target, edge.path_filter)

        # Copy trigger callbacks
        for callback in graph._trigger_callbacks:
            new_graph.on_trigger(callback)

        return new_graph

    def _wrap_modules(
        self,
        modules: List["Module"],
        isolation: IsolationConfig,
    ) -> List["Module"]:
        """Wrap modules that need isolation in WorkerModule.

        Modules at INLINE level are kept as-is.
        Higher levels are wrapped in WorkerModule.

        Args:
            modules: Original module list.
            isolation: Isolation configuration.

        Returns:
            List of modules (some may be WorkerModule wrappers).
        """
        from visualpath.process.launcher import WorkerLauncher
        from visualpath.process.worker_module import WorkerModule

        result: List["Module"] = []

        for module in modules:
            level = isolation.get_level(module.name)

            if level == IsolationLevel.INLINE:
                result.append(module)
                continue

            # Create worker for this module
            venv_path = isolation.get_venv_path(module.name)

            try:
                worker = WorkerLauncher.create(
                    level=level,
                    analyzer=module if level <= IsolationLevel.THREAD else None,
                    venv_path=venv_path,
                    analyzer_name=module.name,
                )
                wrapped = WorkerModule(
                    name=module.name,
                    worker=worker,
                    depends=list(module.depends) if module.depends else [],
                    optional_depends=list(module.optional_depends) if module.optional_depends else [],
                    stateful=module.stateful,
                    capabilities=module.capabilities,
                )
                result.append(wrapped)
                logger.info(
                    "Wrapped module '%s' with %s isolation",
                    module.name, level.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to wrap module '%s' for %s isolation: %s. "
                    "Using inline fallback.",
                    module.name, level.name, e,
                )
                result.append(module)

        return result


__all__ = ["WorkerBackend"]
