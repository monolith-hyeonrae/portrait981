"""SimpleInterpreter — spec-based execution engine.

The interpreter reads NodeSpec from each FlowNode and executes accordingly.
This is the "interpreter" half of the AST/interpreter pattern:

    FlowGraph = AST
    NodeSpec  = token
    Interpreter = this module

Each spec type maps to an interpret_* method that contains
the execution logic previously spread across node.process() methods.

Stateful interpreters maintain per-node state (counters, buffers, timers)
in a separate dict, keeping nodes themselves stateless/declarative.

Example:
    >>> from visualpath.backends.simple.interpreter import SimpleInterpreter
    >>> interpreter = SimpleInterpreter()
    >>> results = interpreter.interpret(node, data)

Debug hooks:
    >>> def on_interpret(event):
    ...     print(f"{event['phase']} {event['node']}: {event}")
    >>> interpreter = SimpleInterpreter(debug_hook=on_interpret)
"""

import logging
import operator
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from visualpath.flow.node import FlowData, FlowNode
from visualpath.flow.specs import (
    NodeSpec,
    SourceSpec,
    ModuleSpec,
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
    JoinSpec,
    CascadeFusionSpec,
    CollectorSpec,
    CustomSpec,
)

if TYPE_CHECKING:
    from visualpath.core.observation import Observation


# Comparison operators for signal filters
_COMPARISONS: Dict[str, Callable[[float, float], bool]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


@dataclass
class DebugEvent:
    """Debug event emitted by interpreter hooks.

    Attributes:
        phase: 'enter', 'exit', or 'state_change'
        node: Node name
        spec_type: Type name of the spec being interpreted
        input_data: Input FlowData (for 'enter')
        output_data: Output FlowData list (for 'exit')
        output_count: Number of outputs produced
        elapsed_ms: Processing time in milliseconds (for 'exit')
        state_key: State key that changed (for 'state_change')
        state_value: New state value (for 'state_change')
        dropped: True if node filtered/dropped the data
        extra: Additional context-specific info
    """

    phase: str
    node: str
    spec_type: str
    input_data: Optional["FlowData"] = None
    output_data: Optional[List["FlowData"]] = None
    output_count: int = 0
    elapsed_ms: float = 0.0
    state_key: Optional[str] = None
    state_value: Any = None
    dropped: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.phase == "enter":
            frame_id = getattr(self.input_data.frame, "frame_id", "?") if self.input_data else "?"
            return f"[ENTER] {self.node} ({self.spec_type}) frame={frame_id}"
        elif self.phase == "exit":
            status = "DROPPED" if self.dropped else f"OUT={self.output_count}"
            return f"[EXIT]  {self.node} ({self.spec_type}) {status} ({self.elapsed_ms:.3f}ms)"
        elif self.phase == "state_change":
            return f"[STATE] {self.node}.{self.state_key} = {self.state_value}"
        return f"[{self.phase.upper()}] {self.node}"


# Type alias for debug hook callback
DebugHook = Callable[[DebugEvent], None]


class NodeProcessingError(Exception):
    """Raised when a user-provided callable within a node fails.

    Wraps the original exception with context about which node and
    spec type caused the error, enabling better diagnostics.

    Attributes:
        node_name: Name of the node where the error occurred.
        spec_type: Type name of the spec being interpreted.
        original_error: The underlying exception.
    """

    def __init__(self, node_name: str, spec_type: str, original_error: Exception):
        self.node_name = node_name
        self.spec_type = spec_type
        self.original_error = original_error
        super().__init__(
            f"Error in node '{node_name}' ({spec_type}): {original_error}"
        )


class SimpleInterpreter:
    """Spec-based interpreter for synchronous execution.

    Reads NodeSpec from FlowNode instances and executes them.
    Maintains per-node state (counters, buffers, timers) internally.

    This interpreter is used by SimpleBackend and GraphExecutor for
    sequential/local execution of flow graphs.

    Error handling:
        When ``on_error="raise"`` (default), errors from user-provided
        callables (conditions, fusion_fn, processors) are wrapped in
        :class:`NodeProcessingError` and re-raised.
        When ``on_error="drop"``, errors are logged and the data is
        dropped (empty output).

    Debug hooks:
        Pass a debug_hook callback to observe internal operations:

        >>> def my_hook(event: DebugEvent):
        ...     print(event)
        >>> interpreter = SimpleInterpreter(debug_hook=my_hook)

        Or use the built-in print hook:

        >>> interpreter = SimpleInterpreter(debug=True)
    """

    def __init__(
        self,
        debug: bool = False,
        debug_hook: Optional[DebugHook] = None,
        on_error: str = "raise",
    ) -> None:
        # Per-node mutable state, keyed by node name
        self._state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._debug = debug
        self._debug_hook = debug_hook
        if on_error not in ("raise", "drop"):
            raise ValueError(f"on_error must be 'raise' or 'drop', got '{on_error}'")
        self._on_error = on_error
        self._current_node: str = ""  # Set during interpret()

    def _emit_debug(self, event: DebugEvent) -> None:
        """Emit a debug event to registered hooks."""
        if self._debug:
            print(event)
        if self._debug_hook is not None:
            self._debug_hook(event)

    def _emit_state_change(
        self, node_name: str, key: str, value: Any, spec_type: str = ""
    ) -> None:
        """Emit a state change debug event."""
        if self._debug or self._debug_hook is not None:
            self._emit_debug(DebugEvent(
                phase="state_change",
                node=node_name,
                spec_type=spec_type,
                state_key=key,
                state_value=value,
            ))

    def _handle_error(
        self, spec_type: str, error: Exception
    ) -> List[FlowData]:
        """Handle an error from a user-provided callable.

        Uses self._current_node for the actual graph node name.
        If on_error="drop", logs the error and returns empty list.
        If on_error="raise", wraps and re-raises as NodeProcessingError.
        """
        node_name = self._current_node
        if self._on_error == "drop":
            logger.error(
                "Error in node '%s' (%s): %s — data dropped",
                node_name, spec_type, error,
            )
            if self._debug or self._debug_hook is not None:
                self._emit_debug(DebugEvent(
                    phase="error",
                    node=node_name,
                    spec_type=spec_type,
                    extra={"error": str(error), "error_type": type(error).__name__},
                ))
            return []
        raise NodeProcessingError(node_name, spec_type, error) from error

    def reset(self) -> None:
        """Clear all interpreter state."""
        self._state.clear()

    def reset_node(self, node_name: str) -> None:
        """Clear state for a specific node."""
        self._state.pop(node_name, None)

    def interpret(self, node: FlowNode, data: FlowData) -> List[FlowData]:
        """Interpret a node's spec and execute it on the given data.

        Args:
            node: The FlowNode whose spec to interpret.
            data: Input FlowData to process.

        Returns:
            List of output FlowData (may be empty if filtered/buffered).

        Raises:
            TypeError: If the spec type is not recognized.
        """
        spec = node.spec
        spec_type = type(spec).__name__
        self._current_node = node.name
        has_debug = self._debug or self._debug_hook is not None

        # Emit enter event
        if has_debug:
            self._emit_debug(DebugEvent(
                phase="enter",
                node=node.name,
                spec_type=spec_type,
                input_data=data,
            ))

        start_time = time.perf_counter() if has_debug else 0

        match spec:
            case SourceSpec():
                outputs = self._interpret_source(spec, data)
            case ModuleSpec():
                outputs = self._interpret_modules(node, spec, data)
            case FilterSpec():
                outputs = self._interpret_filter(spec, data)
            case ObservationFilterSpec():
                outputs = self._interpret_observation_filter(spec, data)
            case SignalFilterSpec():
                outputs = self._interpret_signal_filter(spec, data)
            case SampleSpec():
                outputs = self._interpret_sample(node.name, spec, data)
            case RateLimitSpec():
                outputs = self._interpret_rate_limit(node.name, spec, data)
            case TimestampSampleSpec():
                outputs = self._interpret_timestamp_sample(node.name, spec, data)
            case BranchSpec():
                outputs = self._interpret_branch(spec, data)
            case FanOutSpec():
                outputs = self._interpret_fanout(spec, data)
            case MultiBranchSpec():
                outputs = self._interpret_multi_branch(spec, data)
            case ConditionalFanOutSpec():
                outputs = self._interpret_conditional_fanout(spec, data)
            case JoinSpec():
                outputs = self._interpret_join(node.name, spec, data)
            case CascadeFusionSpec():
                outputs = self._interpret_cascade_fusion(spec, data)
            case CollectorSpec():
                outputs = self._interpret_collector(node.name, spec, data)
            case CustomSpec():
                outputs = self._interpret_custom(spec, data)
            case _:
                raise TypeError(
                    f"SimpleInterpreter does not know how to interpret "
                    f"{type(spec).__name__} from node '{node.name}'"
                )

        # Emit exit event
        if has_debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._emit_debug(DebugEvent(
                phase="exit",
                node=node.name,
                spec_type=spec_type,
                output_data=outputs,
                output_count=len(outputs),
                elapsed_ms=elapsed_ms,
                dropped=len(outputs) == 0,
            ))

        return outputs

    def flush_node(self, node: FlowNode) -> List[FlowData]:
        """Flush any buffered data for a stateful node.

        Used for JoinNode (pending buffers) and CollectorNode (partial batches).
        """
        spec = node.spec
        match spec:
            case JoinSpec():
                return self._flush_join(node.name, spec)
            case CollectorSpec():
                return self._flush_collector(node.name, spec)
            case _:
                return []

    # -----------------------------------------------------------------
    # Source
    # -----------------------------------------------------------------

    def _interpret_source(
        self, spec: SourceSpec, data: FlowData
    ) -> List[FlowData]:
        """Source just passes data through with the default path_id."""
        return [data]

    # -----------------------------------------------------------------
    # Modules (unified analyzer/fusion processing)
    # -----------------------------------------------------------------

    def _toposort_modules(self, modules: tuple) -> list:
        """Topologically sort modules based on their depends/optional_depends."""
        from visualpath.core.graph import toposort_modules
        return toposort_modules(modules)

    def _level_sort_modules(self, modules: tuple) -> List[list]:
        """Group modules into dependency levels for parallel execution.

        Modules in the same level have no inter-dependencies and can
        run concurrently. Each level's modules only depend on modules
        from previous levels.

        Returns:
            List of levels, each containing modules that can run concurrently.
        """
        if not modules:
            return []

        name_to_mod = {m.name: m for m in modules}
        available = set(name_to_mod.keys())

        in_degree: Dict[str, int] = {m.name: 0 for m in modules}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for m in modules:
            for dep in list(getattr(m, 'depends', [])) + list(
                getattr(m, 'optional_depends', [])
            ):
                if dep in available:
                    in_degree[m.name] += 1
                    dependents[dep].append(m.name)

        # BFS by levels (Kahn's algorithm, level by level)
        levels: List[list] = []
        current_names = [name for name, d in in_degree.items() if d == 0]

        while current_names:
            levels.append([name_to_mod[name] for name in current_names])
            next_names: List[str] = []
            for name in current_names:
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        next_names.append(dep_name)
            current_names = next_names

        # Handle remaining modules (cycles) — append as final level
        processed = {m.name for level in levels for m in level}
        remaining = [m for m in modules if m.name not in processed]
        if remaining:
            levels.append(remaining)

        return levels

    def _interpret_modules(
        self, node: FlowNode, spec: ModuleSpec, data: FlowData
    ) -> List[FlowData]:
        """Run modules on the frame.

        Modules are topologically sorted by dependencies, then processed
        in order. Each module can depend on previous modules' outputs.
        Observations with should_trigger=True are added to results.

        When ``spec.parallel`` is True, independent modules within the
        same dependency level are dispatched concurrently via threads.
        """
        from visualpath.core.observation import Observation

        frame = data.frame
        if frame is None:
            return [data]

        # Build dependency map from existing data
        deps: Dict[str, Any] = {
            obs.source: obs for obs in data.observations
        }
        # Also include existing results by source if available
        for result in data.results:
            if hasattr(result, 'source') and result.source:
                deps[result.source] = result

        observations: List["Observation"] = []
        results: List["Observation"] = []

        if spec.parallel:
            self._run_modules_parallel(spec, frame, deps, observations, results)
        else:
            self._run_modules_sequential(spec, frame, deps, observations, results)

        # Update FlowData
        result = data.clone(
            observations=list(data.observations) + observations,
            results=list(data.results) + results,
            path_id=node.name,
        )

        return [result]

    def _run_modules_sequential(
        self,
        spec: ModuleSpec,
        frame: Any,
        deps: Dict[str, Any],
        observations: list,
        results: list,
    ) -> None:
        """Run modules sequentially in topological order."""
        for module in self._toposort_modules(spec.modules):
            output = self._dispatch_module(module, frame, deps)
            if output is not None:
                observations.append(output)
                deps[module.name] = output
                if output.should_trigger:
                    results.append(output)

    def _run_modules_parallel(
        self,
        spec: ModuleSpec,
        frame: Any,
        deps: Dict[str, Any],
        observations: list,
        results: list,
    ) -> None:
        """Run modules in parallel by dependency level.

        Groups modules into levels where all modules in a level have
        their dependencies satisfied by previous levels. Modules
        within the same level are dispatched concurrently via threads.
        """
        from concurrent.futures import ThreadPoolExecutor

        levels = self._level_sort_modules(spec.modules)

        for level in levels:
            if len(level) == 1:
                # Single module — no thread overhead
                module = level[0]
                output = self._dispatch_module(module, frame, deps)
                if output is not None:
                    observations.append(output)
                    deps[module.name] = output
                    if output.should_trigger:
                        results.append(output)
            else:
                # Multiple independent modules — dispatch concurrently
                with ThreadPoolExecutor(max_workers=len(level)) as pool:
                    futures = {
                        pool.submit(self._dispatch_module, module, frame, deps): module
                        for module in level
                    }
                # Collect results in original level order for deterministic output
                level_outputs: List = []
                for module in level:
                    future = next(f for f, m in futures.items() if m is module)
                    output = future.result()
                    level_outputs.append((module, output))

                for module, output in level_outputs:
                    if output is not None:
                        observations.append(output)
                        deps[module.name] = output
                        if output.should_trigger:
                            results.append(output)

    def _dispatch_module(
        self,
        module: Any,
        frame: Any,
        deps: Dict[str, Any],
    ) -> Any:
        """Prepare deps and call a single module."""
        module_deps = self._build_module_deps(module, deps)
        return self._call_module(module, frame, module_deps)

    def _build_module_deps(
        self,
        module: Any,
        deps: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build the deps dict for a module from the global deps map."""
        all_dep_names = list(getattr(module, 'depends', [])) + list(
            getattr(module, 'optional_depends', [])
        )
        if not all_dep_names:
            return None
        return {
            name: deps[name]
            for name in all_dep_names
            if name in deps
        }

    def _dispatch_module_batch(
        self,
        module: Any,
        frames: List[Any],
        deps_per_frame: List[Dict[str, Any]],
    ) -> List[Any]:
        """Dispatch a batch of frames to a module.

        If the module declares Capability.BATCHING, calls process_batch().
        Otherwise falls back to sequential process() calls.

        Args:
            module: Module to call.
            frames: List of frames.
            deps_per_frame: Per-frame deps dicts (same length as frames).

        Returns:
            List of observations (same length as frames).
        """
        from visualpath.core.capabilities import Capability

        caps = getattr(module, 'capabilities', None)
        has_batching = (
            caps is not None
            and Capability.BATCHING in caps.flags
            and hasattr(module, 'process_batch')
        )

        if has_batching:
            deps_list = [
                self._build_module_deps(module, d) for d in deps_per_frame
            ]
            return module.process_batch(frames, deps_list)

        # Sequential fallback
        results = []
        for frame, deps in zip(frames, deps_per_frame):
            module_deps = self._build_module_deps(module, deps)
            results.append(self._call_module(module, frame, module_deps))
        return results

    def interpret_modules_batch(
        self,
        node: FlowNode,
        spec: "ModuleSpec",
        datas: List[FlowData],
    ) -> List[List[FlowData]]:
        """Process a batch of FlowData items through a ModuleSpec node.

        Uses batch dispatch for modules with Capability.BATCHING.
        Modules without BATCHING are called sequentially per frame.

        Args:
            node: The FlowNode containing the ModuleSpec.
            spec: The ModuleSpec with modules to run.
            datas: List of FlowData items (one per frame).

        Returns:
            List of output FlowData lists, one per input FlowData.
        """
        from visualpath.core.observation import Observation

        n = len(datas)
        frames = [d.frame for d in datas]

        # Per-frame tracking
        per_frame_deps: List[Dict[str, Any]] = []
        per_frame_observations: List[list] = []
        per_frame_results: List[list] = []

        for data in datas:
            deps: Dict[str, Any] = {
                obs.source: obs for obs in data.observations
            }
            for result in data.results:
                if hasattr(result, 'source') and result.source:
                    deps[result.source] = result
            per_frame_deps.append(deps)
            per_frame_observations.append([])
            per_frame_results.append([])

        # Skip frames with no frame data
        valid_mask = [d.frame is not None for d in datas]
        if not any(valid_mask):
            return [[data] for data in datas]

        # Process modules by dependency level
        levels = self._level_sort_modules(spec.modules)

        for level in levels:
            for module in level:
                batch_outputs = self._dispatch_module_batch(
                    module, frames, per_frame_deps
                )
                for i, output in enumerate(batch_outputs):
                    if output is not None:
                        per_frame_observations[i].append(output)
                        per_frame_deps[i][module.name] = output
                        if output.should_trigger:
                            per_frame_results[i].append(output)

        # Build output FlowData for each frame
        all_outputs: List[List[FlowData]] = []
        for i, data in enumerate(datas):
            if not valid_mask[i]:
                all_outputs.append([data])
            else:
                result = data.clone(
                    observations=list(data.observations) + per_frame_observations[i],
                    results=list(data.results) + per_frame_results[i],
                    path_id=node.name,
                )
                all_outputs.append([result])

        return all_outputs

    def _call_module(
        self,
        module: Any,
        frame: Any,
        deps: Optional[Dict[str, Any]],
    ) -> Any:
        """Call a module's process method, respecting ErrorPolicy if declared."""
        policy = getattr(module, 'error_policy', None)
        if policy is None:
            return module.process(frame, deps)

        # Apply retry + error policy
        last_error = None
        max_attempts = 1 + policy.max_retries
        for attempt in range(max_attempts):
            try:
                return module.process(frame, deps)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.debug(
                        "Module '%s' attempt %d/%d failed: %s",
                        getattr(module, 'name', '?'),
                        attempt + 1, max_attempts, e,
                    )
                    continue

        # All attempts failed — apply on_error policy
        if policy.on_error == "raise":
            raise last_error  # type: ignore[misc]
        elif policy.on_error == "fallback" and policy.fallback_signals:
            from visualpath.core.observation import Observation
            return Observation(
                source=getattr(module, 'name', 'unknown'),
                frame_id=getattr(frame, 'frame_id', 0),
                t_ns=getattr(frame, 't_src_ns', 0),
                signals=dict(policy.fallback_signals),
            )
        # "skip" or default
        logger.warning(
            "Module '%s' failed after %d attempts, skipping: %s",
            getattr(module, 'name', '?'), max_attempts, last_error,
        )
        return None

    # -----------------------------------------------------------------
    # Filters
    # -----------------------------------------------------------------

    def _interpret_filter(
        self, spec: FilterSpec, data: FlowData
    ) -> List[FlowData]:
        """Pass data if condition is true, drop otherwise."""
        try:
            if spec.condition(data):
                return [data]
        except Exception as e:
            return self._handle_error("FilterSpec", e)
        return []

    def _interpret_observation_filter(
        self, spec: ObservationFilterSpec, data: FlowData
    ) -> List[FlowData]:
        """Pass data if enough observations exist."""
        if len(data.observations) >= spec.min_count:
            return [data]
        return []

    def _interpret_signal_filter(
        self, spec: SignalFilterSpec, data: FlowData
    ) -> List[FlowData]:
        """Pass data if any observation's signal passes threshold."""
        cmp_fn = _COMPARISONS.get(spec.comparison, operator.gt)
        for obs in data.observations:
            value = obs.signals.get(spec.signal_name)
            if value is not None and cmp_fn(value, spec.threshold):
                return [data]
        return []

    # -----------------------------------------------------------------
    # Samplers
    # -----------------------------------------------------------------

    def _interpret_sample(
        self, node_name: str, spec: SampleSpec, data: FlowData
    ) -> List[FlowData]:
        """Every-Nth sampler using internal counter."""
        state = self._state[node_name]
        count = state.get("count", 0) + 1
        state["count"] = count
        self._emit_state_change(node_name, "count", count, "SampleSpec")

        if count % spec.every_nth == 0:
            return [data]
        return []

    def _interpret_rate_limit(
        self, node_name: str, spec: RateLimitSpec, data: FlowData
    ) -> List[FlowData]:
        """Rate limiter using wall-clock time."""
        state = self._state[node_name]
        now = time.monotonic()
        last_time = state.get("last_time")

        if last_time is None:
            state["last_time"] = now
            self._emit_state_change(node_name, "last_time", now, "RateLimitSpec")
            return [data]

        elapsed_ms = (now - last_time) * 1000
        if elapsed_ms >= spec.min_interval_ms:
            state["last_time"] = now
            self._emit_state_change(node_name, "last_time", now, "RateLimitSpec")
            return [data]
        return []

    def _interpret_timestamp_sample(
        self, node_name: str, spec: TimestampSampleSpec, data: FlowData
    ) -> List[FlowData]:
        """Timestamp-based sampler using data timestamps."""
        state = self._state[node_name]
        last_ts = state.get("last_timestamp_ns")

        if last_ts is None:
            state["last_timestamp_ns"] = data.timestamp_ns
            self._emit_state_change(
                node_name, "last_timestamp_ns", data.timestamp_ns, "TimestampSampleSpec"
            )
            return [data]

        if data.timestamp_ns - last_ts >= spec.interval_ns:
            state["last_timestamp_ns"] = data.timestamp_ns
            self._emit_state_change(
                node_name, "last_timestamp_ns", data.timestamp_ns, "TimestampSampleSpec"
            )
            return [data]
        return []

    # -----------------------------------------------------------------
    # Branching
    # -----------------------------------------------------------------

    def _interpret_branch(
        self, spec: BranchSpec, data: FlowData
    ) -> List[FlowData]:
        """Binary branch: route to if_true or if_false path."""
        try:
            if spec.condition(data):
                return [data.with_path(spec.if_true)]
            return [data.with_path(spec.if_false)]
        except Exception as e:
            return self._handle_error("BranchSpec", e)

    def _interpret_fanout(
        self, spec: FanOutSpec, data: FlowData
    ) -> List[FlowData]:
        """Fan-out: clone data for each path."""
        return [data.clone(path_id=path) for path in spec.paths]

    def _interpret_multi_branch(
        self, spec: MultiBranchSpec, data: FlowData
    ) -> List[FlowData]:
        """Multi-way branch: route to first matching condition."""
        try:
            for condition, path_id in spec.branches:
                if condition(data):
                    return [data.with_path(path_id)]
        except Exception as e:
            return self._handle_error("MultiBranchSpec", e)
        if spec.default is not None:
            return [data.with_path(spec.default)]
        return []

    def _interpret_conditional_fanout(
        self, spec: ConditionalFanOutSpec, data: FlowData
    ) -> List[FlowData]:
        """Conditional fan-out: clone for each passing condition."""
        results = []
        try:
            for path_id, condition in spec.paths:
                if condition(data):
                    results.append(data.clone(path_id=path_id))
        except Exception as e:
            return self._handle_error("ConditionalFanOutSpec", e)
        return results

    # -----------------------------------------------------------------
    # Join
    # -----------------------------------------------------------------

    def _interpret_join(
        self, node_name: str, spec: JoinSpec, data: FlowData
    ) -> List[FlowData]:
        """Join: buffer data from multiple paths, emit when complete.

        Temporal semantics (when window_ns > 0):
        - Data from different paths is only joined if their timestamps
          are within ``window_ns`` of each other.
        - When ``lateness_ns > 0``, buffered data older than
          ``lateness_ns`` relative to the newest arrival is evicted.
        """
        state = self._state[node_name]

        # Unknown path → pass through
        if data.path_id not in spec.input_paths:
            return [data]

        # Buffer
        buffers = state.setdefault("buffers", {})
        buffers[data.path_id] = data
        self._emit_state_change(
            node_name, f"buffers[{data.path_id}]", "buffered", "JoinSpec"
        )

        # Evict stale data beyond lateness_ns
        if spec.lateness_ns > 0 and len(buffers) > 1:
            newest_ts = max(d.timestamp_ns for d in buffers.values())
            stale_paths = [
                pid for pid, d in buffers.items()
                if (newest_ts - d.timestamp_ns) > spec.lateness_ns
            ]
            for pid in stale_paths:
                del buffers[pid]
                self._emit_state_change(
                    node_name, f"buffers[{pid}]", "evicted (late)", "JoinSpec"
                )

        # Check temporal alignment (window_ns)
        if spec.window_ns > 0 and len(buffers) > 1:
            timestamps = [d.timestamp_ns for d in buffers.values()]
            spread = max(timestamps) - min(timestamps)
            if spread > spec.window_ns:
                # Timestamps too far apart — don't join yet
                return []

        # Check emit condition
        if spec.mode == "any":
            return self._emit_join(node_name, spec)
        elif spec.mode == "all":
            if set(buffers.keys()) >= set(spec.input_paths):
                return self._emit_join(node_name, spec)
        return []

    def _emit_join(
        self, node_name: str, spec: JoinSpec
    ) -> List[FlowData]:
        """Emit merged data and clear buffers."""
        state = self._state[node_name]
        buffers: Dict[str, FlowData] = state.get("buffers", {})

        if not buffers:
            return []

        # Merge observations and results from all buffered data
        merged_observations: List = []
        merged_results: List = []
        merged_metadata: Dict = {}
        first_data = next(iter(buffers.values()))

        for path_data in buffers.values():
            if spec.merge_observations:
                merged_observations.extend(path_data.observations)
            if spec.merge_results:
                merged_results.extend(path_data.results)
            merged_metadata.update(path_data.metadata)

        merged = first_data.clone(
            path_id=spec.output_path_id,
            observations=merged_observations,
            results=merged_results,
            metadata=merged_metadata,
        )

        # Clear buffers
        state["buffers"] = {}

        return [merged]

    def _flush_join(
        self, node_name: str, spec: JoinSpec
    ) -> List[FlowData]:
        """Flush pending join buffers."""
        return self._emit_join(node_name, spec)

    # -----------------------------------------------------------------
    # Cascade Fusion
    # -----------------------------------------------------------------

    def _interpret_cascade_fusion(
        self, spec: CascadeFusionSpec, data: FlowData
    ) -> List[FlowData]:
        """Apply cascade fusion function to data."""
        try:
            result = spec.fusion_fn(data)
        except Exception as e:
            return self._handle_error("CascadeFusionSpec", e)
        return [result]

    # -----------------------------------------------------------------
    # Collector
    # -----------------------------------------------------------------

    def _interpret_collector(
        self, node_name: str, spec: CollectorSpec, data: FlowData
    ) -> List[FlowData]:
        """Accumulate data into batches."""
        state = self._state[node_name]
        buffer: List[FlowData] = state.setdefault("buffer", [])
        buffer.append(data)
        self._emit_state_change(
            node_name, "buffer_size", len(buffer), "CollectorSpec"
        )

        if len(buffer) >= spec.batch_size:
            return self._emit_collector(node_name, spec)
        return []

    def _emit_collector(
        self, node_name: str, spec: CollectorSpec
    ) -> List[FlowData]:
        """Emit collected batch."""
        state = self._state[node_name]
        buffer: List[FlowData] = state.get("buffer", [])

        if not buffer:
            return []

        # Merge all buffered data
        merged_observations: List = []
        merged_results: List = []
        for item in buffer:
            merged_observations.extend(item.observations)
            merged_results.extend(item.results)

        batch = buffer[0].clone(
            observations=merged_observations,
            results=merged_results,
            metadata={**buffer[0].metadata, "_batch_size": len(buffer)},
        )

        # Clear buffer
        state["buffer"] = []

        return [batch]

    def _flush_collector(
        self, node_name: str, spec: CollectorSpec
    ) -> List[FlowData]:
        """Flush partial batch if emit_partial is True."""
        if spec.emit_partial:
            return self._emit_collector(node_name, spec)
        return []

    # -----------------------------------------------------------------
    # Custom
    # -----------------------------------------------------------------

    def _interpret_custom(
        self, spec: CustomSpec, data: FlowData
    ) -> List[FlowData]:
        """Execute a user-provided custom processor."""
        try:
            result = spec.processor(data)
        except Exception as e:
            return self._handle_error("CustomSpec", e)
        if isinstance(result, list):
            return result
        return [result]


__all__ = ["SimpleInterpreter", "NodeProcessingError", "DebugEvent", "DebugHook"]
