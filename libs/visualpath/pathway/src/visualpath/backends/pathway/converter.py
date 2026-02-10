"""FlowGraph to Pathway dataflow converter.

This module converts visualpath FlowGraph structures into Pathway
streaming dataflows using declarative NodeSpec dispatch.

Conversion mapping (spec-based):
- SourceSpec -> pw.io.python.read()
- ModuleSpec -> @pw.udf with PyObjectWrapper (per-module parallel branches)
- JoinSpec -> interval_join() with spec-defined window/lateness
- FilterSpec/ObservationFilterSpec/SignalFilterSpec -> pw.filter()
- SampleSpec -> frame_id modulo filter
- RateLimitSpec/TimestampSampleSpec -> pass-through (wall-clock based)
- BranchSpec/FanOutSpec/MultiBranchSpec/ConditionalFanOutSpec -> table replication
- CascadeFusionSpec/CollectorSpec -> pass-through
- spec=None -> pass-through (fallback to process())
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowNode

logger = logging.getLogger(__name__)

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


class FlowGraphConverter:
    """Converts FlowGraph to Pathway dataflow.

    FlowGraphConverter traverses the FlowGraph in topological order
    and creates corresponding Pathway operators based on each node's
    ``spec`` property.  Nodes without a spec (``spec is None``) are
    treated as pass-through, relying on ``process()`` for execution
    in non-Pathway backends.

    When ``ModuleSpec.parallel=True`` and there are independent
    analyzer groups (no cross-dependencies), each group gets its
    own Pathway UDF so the engine can schedule them in parallel.
    The branches are then rejoined via ``interval_join``.

    Example:
        >>> converter = FlowGraphConverter()
        >>> pw_dataflow = converter.convert(graph, frames_table)
    """

    def __init__(
        self,
        window_ns: int = 100_000_000,  # 100ms default
        allowed_lateness_ns: int = 50_000_000,  # 50ms default
    ) -> None:
        """Initialize the converter.

        Args:
            window_ns: Default window size for joins (nanoseconds).
            allowed_lateness_ns: Allowed late arrival time (nanoseconds).
        """
        self._window_ns = window_ns
        self._allowed_lateness_ns = allowed_lateness_ns
        self._tables: Dict[str, "pw.Table"] = {}
        self._deferred_modules: list = []

    def convert(
        self,
        graph: "FlowGraph",
        frames_table: "pw.Table",
    ) -> "pw.Table":
        """Convert a FlowGraph to Pathway dataflow.

        Args:
            graph: The FlowGraph to convert.
            frames_table: Initial Pathway table of frames.

        Returns:
            Final Pathway table representing the output.

        Raises:
            ValueError: If graph contains unsupported node types.
        """
        if not PATHWAY_AVAILABLE:
            raise ImportError(
                "Pathway is not installed. Install with: pip install visualpath[pathway]"
            )

        # Get topological order
        order = graph.topological_order()

        # Process nodes in order
        for node_name in order:
            node = graph.nodes[node_name]
            self._convert_node(node, graph, frames_table)

        # Return the last table
        if order:
            return self._tables.get(order[-1], frames_table)
        return frames_table

    @property
    def deferred_modules(self) -> list:
        """Stateful modules deferred from UDF to ordered subscribe callback.

        Pathway UDFs execute in arbitrary order across rows. Stateful
        modules (``is_trigger=True``) require temporal ordering, so they
        are excluded from the UDF and returned here for the backend to
        run in the subscribe callback, which Pathway delivers in order.
        """
        return list(self._deferred_modules)

    def _convert_node(
        self,
        node: "FlowNode",
        graph: "FlowGraph",
        frames_table: "pw.Table",
    ) -> None:
        """Convert a single node to Pathway operators using its spec."""
        from visualpath.flow.specs import (
            SourceSpec,
            ModuleSpec,
            JoinSpec,
            FilterSpec,
            ObservationFilterSpec,
            SignalFilterSpec,
            SampleSpec,
            RateLimitSpec,
            TimestampSampleSpec,
            BranchSpec,
            FanOutSpec,
            MultiBranchSpec,
            ConditionalFanOutSpec,
            CascadeFusionSpec,
            CollectorSpec,
        )

        spec = node.spec

        if spec is None:
            # No spec — pass through from predecessor
            self._convert_passthrough_node(node, graph)
            return

        if isinstance(spec, SourceSpec):
            self._tables[node.name] = frames_table

        elif isinstance(spec, ModuleSpec):
            self._convert_module_spec(node.name, spec, graph)

        elif isinstance(spec, JoinSpec):
            self._convert_join_spec(node.name, spec, graph)

        elif isinstance(spec, (FilterSpec, ObservationFilterSpec, SignalFilterSpec)):
            self._convert_filter_spec(node.name, spec, graph)

        elif isinstance(spec, SampleSpec):
            self._convert_sample_spec(node.name, spec, graph)

        elif isinstance(spec, (RateLimitSpec, TimestampSampleSpec)):
            self._convert_passthrough_node(node, graph)

        elif isinstance(spec, (BranchSpec, FanOutSpec, MultiBranchSpec, ConditionalFanOutSpec)):
            self._convert_branch_node(node, graph)

        elif isinstance(spec, (CascadeFusionSpec, CollectorSpec)):
            self._convert_passthrough_node(node, graph)

        else:
            self._convert_passthrough_node(node, graph)

    # ------------------------------------------------------------------
    # Helper: get input table from predecessor
    # ------------------------------------------------------------------

    def _get_input_table(self, node_name: str, graph: "FlowGraph"):
        """Get the Pathway table from the first predecessor of a node."""
        predecessors = graph.get_incoming_edges(node_name)
        if not predecessors:
            return None
        pred_name = predecessors[0].source
        return self._tables.get(pred_name)

    # ------------------------------------------------------------------
    # Pass-through / branch
    # ------------------------------------------------------------------

    def _convert_passthrough_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert a node as pass-through from predecessor."""
        input_table = self._get_input_table(node.name, graph)
        if input_table is not None:
            self._tables[node.name] = input_table

    def _convert_branch_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert branching nodes (pass-through — table replication)."""
        input_table = self._get_input_table(node.name, graph)
        if input_table is not None:
            self._tables[node.name] = input_table

    # ------------------------------------------------------------------
    # ModuleSpec — per-module parallel branches
    # ------------------------------------------------------------------

    def _convert_module_spec(
        self,
        node_name: str,
        spec: "ModuleSpec",
        graph: "FlowGraph",
    ) -> None:
        """Convert ModuleSpec to Pathway UDF(s).

        When ``spec.isolation`` is set, modules needing isolation are
        wrapped in WorkerModule before being converted to UDFs.

        When ``spec.parallel=True``, each analyzer gets its own Pathway
        UDF so the Rust engine can schedule independent branches in
        parallel.  Dependent analyzers chain from their upstream table.
        All per-analyzer tables are joined via ``interval_join``.
        """
        input_table = self._get_input_table(node_name, graph)
        if input_table is None:
            return

        modules = list(spec.modules)
        if not modules:
            self._tables[node_name] = input_table
            return

        # Defer stateful trigger modules — Pathway UDFs execute in
        # arbitrary row order, but subscribe delivers rows in temporal
        # order.  Stateful modules (is_trigger=True) need ordering, so
        # they are excluded from the UDF and run in the subscribe callback.
        analyzers = [m for m in modules if not getattr(m, 'is_trigger', False)]
        deferred = [m for m in modules if getattr(m, 'is_trigger', False)]
        if deferred:
            self._deferred_modules.extend(deferred)
            logger.info(
                "Deferred %d stateful module(s) to ordered subscribe: %s",
                len(deferred),
                [m.name for m in deferred],
            )
        modules = analyzers

        if not modules:
            self._tables[node_name] = input_table
            return

        # Wrap isolated modules in WorkerModule if isolation config is present
        if spec.isolation is not None:
            modules = self._wrap_isolated_modules(modules, spec.isolation)

        # Single UDF path: no parallelism or single module
        if not spec.parallel or len(modules) == 1:
            self._tables[node_name] = self._build_single_udf(
                node_name, modules, input_table,
            )
            return

        # Per-analyzer DAG: each analyzer gets its own UDF + table
        branch_tables = self._build_per_analyzer_dag(
            modules, input_table, spec.join_window_ns,
        )

        if len(branch_tables) == 1:
            self._tables[node_name] = branch_tables[0]
            return

        # Merge all per-analyzer tables via interval_join
        joined = self._auto_join(branch_tables, spec.join_window_ns)
        self._tables[node_name] = joined

    def _build_single_udf(
        self,
        name: str,
        analyzers: list,
        input_table: "pw.Table",
    ) -> "pw.Table":
        """Build a single Pathway UDF that runs analyzers sequentially."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        raw_udf = create_multi_analyzer_udf(analyzers)

        @pw.udf
        def analyze_udf(
            frame_wrapped: pw.PyObjectWrapper,
        ) -> pw.PyObjectWrapper:
            frame = frame_wrapped.value
            results = raw_udf(frame)
            return pw.PyObjectWrapper(results)

        return input_table.select(
            frame_id=pw.this.frame_id,
            t_ns=pw.this.t_ns,
            frame=pw.this.frame,
            results=analyze_udf(pw.this.frame),
        )

    def _build_per_analyzer_dag(
        self,
        analyzers: list,
        input_table: "pw.Table",
        window_ns: int,
    ) -> List["pw.Table"]:
        """Build per-analyzer DAG: each analyzer → individual UDF + table.

        Independent analyzers branch from input_table (Rust engine
        parallelizes them automatically). Dependent analyzers chain
        from their upstream table(s).

        Returns:
            List of per-analyzer tables (leaf tables only — those not
            depended on by other analyzers in this set, plus independent
            analyzers).
        """
        from visualpath.backends.pathway.operators import (
            create_single_analyzer_udf,
            create_dep_analyzer_udf,
            _toposort_modules,
        )

        sorted_analyzers = _toposort_modules(analyzers)
        analyzer_names = {a.name for a in analyzers}
        analyzer_tables: Dict[str, "pw.Table"] = {}

        for analyzer in sorted_analyzers:
            dep_names = [
                d for d in (getattr(analyzer, 'depends', None) or [])
                if d in analyzer_tables
            ]

            if not dep_names:
                # Independent: branch from input_table
                raw_udf = create_single_analyzer_udf(analyzer)

                @pw.udf
                def _single_udf(
                    frame_wrapped: pw.PyObjectWrapper,
                    _fn=raw_udf,
                ) -> pw.PyObjectWrapper:
                    frame = frame_wrapped.value
                    return pw.PyObjectWrapper(_fn(frame))

                table = input_table.select(
                    frame_id=pw.this.frame_id,
                    t_ns=pw.this.t_ns,
                    frame=pw.this.frame,
                    results=_single_udf(pw.this.frame),
                )
            elif len(dep_names) == 1:
                # Single dependency: chain from upstream table
                dep_table = analyzer_tables[dep_names[0]]
                raw_udf = create_dep_analyzer_udf(analyzer)

                @pw.udf
                def _dep_udf(
                    frame_wrapped: pw.PyObjectWrapper,
                    results_wrapped: pw.PyObjectWrapper,
                    _fn=raw_udf,
                ) -> pw.PyObjectWrapper:
                    frame = frame_wrapped.value
                    upstream = results_wrapped.value if hasattr(results_wrapped, 'value') else results_wrapped
                    new_results = _fn(frame, upstream)
                    # Combine upstream + own results
                    return pw.PyObjectWrapper(list(upstream) + new_results)

                table = dep_table.select(
                    frame_id=pw.this.frame_id,
                    t_ns=pw.this.t_ns,
                    frame=pw.this.frame,
                    results=_dep_udf(pw.this.frame, pw.this.results),
                )
            else:
                # Multiple dependencies: join upstream tables, then chain
                dep_tables = [analyzer_tables[n] for n in dep_names]
                joined_deps = self._auto_join(dep_tables, window_ns)
                raw_udf = create_dep_analyzer_udf(analyzer)

                @pw.udf
                def _multi_dep_udf(
                    frame_wrapped: pw.PyObjectWrapper,
                    results_wrapped: pw.PyObjectWrapper,
                    _fn=raw_udf,
                ) -> pw.PyObjectWrapper:
                    frame = frame_wrapped.value
                    upstream = results_wrapped.value if hasattr(results_wrapped, 'value') else results_wrapped
                    new_results = _fn(frame, upstream)
                    return pw.PyObjectWrapper(list(upstream) + new_results)

                table = joined_deps.select(
                    frame_id=pw.this.frame_id,
                    t_ns=pw.this.t_ns,
                    frame=pw.this.frame,
                    results=_multi_dep_udf(pw.this.frame, pw.this.results),
                )

            analyzer_tables[analyzer.name] = table

        # Return leaf tables: analyzers not depended on by others in this set
        depended_on = set()
        for a in analyzers:
            for d in (getattr(a, 'depends', None) or []):
                if d in analyzer_names:
                    depended_on.add(d)
        leaf_tables = [
            analyzer_tables[a.name]
            for a in sorted_analyzers
            if a.name not in depended_on
        ]
        return leaf_tables if leaf_tables else list(analyzer_tables.values())

    def _auto_join(
        self,
        tables: List["pw.Table"],
        window_ns: int,
    ) -> "pw.Table":
        """Join multiple branch tables, merging results.

        Per-analyzer tables derive from the same input_table (same
        universe), so we join on ``frame_id`` for exact 1:1 matching.
        This avoids the cross-product problem of ``interval_join``
        where nearby frames would spuriously match.
        """
        if len(tables) == 1:
            return tables[0]

        @pw.udf
        def merge_results_udf(
            left: pw.PyObjectWrapper,
            right: pw.PyObjectWrapper,
        ) -> pw.PyObjectWrapper:
            left_list = left.value if hasattr(left, 'value') else left
            right_list = right.value if hasattr(right, 'value') else right
            return pw.PyObjectWrapper(list(left_list) + list(right_list))

        joined = tables[0]
        for table in tables[1:]:
            joined = joined.join(
                table,
                pw.left.frame_id == pw.right.frame_id,
            ).select(
                frame_id=pw.left.frame_id,
                t_ns=pw.left.t_ns,
                frame=pw.left.frame,
                results=merge_results_udf(pw.left.results, pw.right.results),
            )

        return joined

    # ------------------------------------------------------------------
    # Isolation — wrap modules in WorkerModule
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_isolated_modules(modules: list, isolation) -> list:
        """Wrap modules needing isolation in WorkerModule.

        Modules at INLINE level are kept as-is. Higher levels are
        wrapped in WorkerModule so Pathway UDFs execute them via
        subprocess workers.

        Args:
            modules: List of Module instances.
            isolation: IsolationConfig with per-module overrides.

        Returns:
            List of modules (some may be WorkerModule wrappers).
        """
        from visualpath.core.isolation import IsolationLevel

        try:
            from visualpath.process.launcher import WorkerLauncher
            from visualpath.process.worker_module import WorkerModule
        except ImportError:
            logger.warning(
                "visualpath-isolation not available, skipping module wrapping"
            )
            return modules

        result = []
        for module in modules:
            level = isolation.get_level(module.name)

            if level == IsolationLevel.INLINE:
                result.append(module)
                continue

            if level == IsolationLevel.PROCESS:
                logger.info(
                    "Skipping PROCESS isolation for '%s' — "
                    "Pathway UDF already provides parallelism",
                    module.name,
                )
                result.append(module)
                continue

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
                )
                result.append(wrapped)
                logger.info(
                    "Wrapped module '%s' with %s isolation for Pathway",
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

    # ------------------------------------------------------------------
    # JoinSpec — temporal config from graph
    # ------------------------------------------------------------------

    def _convert_join_spec(
        self,
        node_name: str,
        spec: "JoinSpec",
        graph: "FlowGraph",
    ) -> None:
        """Convert JoinSpec to Pathway interval_join.

        Uses ``spec.window_ns`` (from graph) as the window size.
        Falls back to ``self._window_ns`` only if spec has no value.
        """
        input_paths = list(spec.input_paths)
        if len(input_paths) < 2:
            if input_paths:
                self._tables[node_name] = self._tables.get(input_paths[0])
            return

        tables = [self._tables.get(path) for path in input_paths]
        if None in tables:
            return

        # Prefer spec window, fall back to converter default
        window = spec.window_ns if spec.window_ns > 0 else self._window_ns

        left = tables[0]
        right = tables[1]

        joined = left.interval_join(
            right,
            pw.left.t_ns,
            pw.right.t_ns,
            pw.temporal.interval(-window, window),
        ).select(
            frame_id=pw.left.frame_id,
            t_ns=pw.left.t_ns,
            frame=pw.left.frame,
        )

        # Join remaining tables
        for table in tables[2:]:
            joined = joined.interval_join(
                table,
                pw.left.t_ns,
                pw.right.t_ns,
                pw.temporal.interval(-window, window),
            ).select(
                frame_id=pw.left.frame_id,
                t_ns=pw.left.t_ns,
                frame=pw.left.frame,
            )

        self._tables[node_name] = joined

    # ------------------------------------------------------------------
    # FilterSpec variants
    # ------------------------------------------------------------------

    def _convert_filter_spec(
        self,
        node_name: str,
        spec,
        graph: "FlowGraph",
    ) -> None:
        """Convert filter specs to Pathway filter."""
        from visualpath.flow.specs import FilterSpec, ObservationFilterSpec, SignalFilterSpec

        input_table = self._get_input_table(node_name, graph)
        if input_table is None:
            return

        if isinstance(spec, FilterSpec):
            condition = spec.condition

            @pw.udf
            def filter_udf(frame_wrapped: pw.PyObjectWrapper) -> bool:
                return condition(frame_wrapped.value)

            self._tables[node_name] = input_table.filter(
                filter_udf(pw.this.frame)
            )
        else:
            # ObservationFilterSpec, SignalFilterSpec — pass through in Pathway
            # (these operate on FlowData which is not available in the
            # Pathway table; they are handled by process() fallback)
            self._tables[node_name] = input_table

    # ------------------------------------------------------------------
    # SampleSpec
    # ------------------------------------------------------------------

    def _convert_sample_spec(
        self,
        node_name: str,
        spec: "SampleSpec",
        graph: "FlowGraph",
    ) -> None:
        """Convert SampleSpec to Pathway sampling via frame_id modulo."""
        input_table = self._get_input_table(node_name, graph)
        if input_table is None:
            return

        every_nth = spec.every_nth
        sampled = input_table.filter(
            pw.this.frame_id % every_nth == 0
        )
        self._tables[node_name] = sampled


__all__ = ["FlowGraphConverter"]
