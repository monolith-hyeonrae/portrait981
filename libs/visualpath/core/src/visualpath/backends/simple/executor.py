"""Graph executor for running flow graphs via the interpreter.

GraphExecutor drives frame processing through a FlowGraph, using
SimpleInterpreter to interpret each node's spec. The executor handles:
- Converting frames to FlowData via SourceSpec
- Routing data through nodes based on edges and path_id
- Firing triggers at terminal nodes

This is a convenience wrapper around FlowGraph + SimpleInterpreter.
"""

from collections import defaultdict, deque
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowData
from visualpath.flow.graph import FlowGraph
from visualpath.backends.simple.interpreter import SimpleInterpreter, DebugHook
from visualpath.flow.specs import SourceSpec, ModuleSpec

if TYPE_CHECKING:
    from visualbase import Frame


class GraphExecutor:
    """Executes a FlowGraph using SimpleInterpreter.

    Args:
        graph: FlowGraph to execute.
        on_trigger: Optional trigger callback.
        batch_size: Number of frames to process together in batch mode.
            When > 1, modules with Capability.BATCHING receive frames via
            process_batch() for GPU batch inference optimization.
        debug: Print debug events to stdout.
        debug_hook: Custom debug event callback.

    Example:
        >>> executor = GraphExecutor(graph)
        >>> with executor:
        ...     for frame in video:
        ...         results = executor.process(frame)

    Debug mode:
        >>> executor = GraphExecutor(graph, debug=True)
        >>> # or with custom hook
        >>> executor = GraphExecutor(graph, debug_hook=my_hook)
    """

    def __init__(
        self,
        graph: FlowGraph,
        on_trigger: Optional[Callable[[FlowData], None]] = None,
        batch_size: int = 1,
        debug: bool = False,
        debug_hook: Optional[DebugHook] = None,
    ):
        self._graph = graph
        self._interpreter = SimpleInterpreter(debug=debug, debug_hook=debug_hook)
        self._initialized = False
        self._debug = debug
        self._batch_size = max(1, batch_size)

        if on_trigger is not None:
            self._graph.on_trigger(on_trigger)

    @property
    def graph(self) -> FlowGraph:
        return self._graph

    @property
    def interpreter(self) -> SimpleInterpreter:
        return self._interpreter

    def initialize(self) -> None:
        if self._initialized:
            return
        self._graph.validate()
        self._graph.initialize()
        self._interpreter.reset()
        self._initialized = True

    def cleanup(self) -> None:
        if not self._initialized:
            return
        self._graph.cleanup()
        self._interpreter.reset()
        self._initialized = False

    def __enter__(self) -> "GraphExecutor":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def process(self, frame: "Frame") -> List[FlowData]:
        """Process a frame through the flow graph.

        Args:
            frame: Input frame to process.

        Returns:
            List of FlowData that reached terminal nodes.
        """
        if not self._initialized:
            raise RuntimeError(
                "Executor not initialized. Use context manager or call initialize()."
            )

        entry_name = self._graph.entry_node
        if entry_name is None:
            return []

        entry_node = self._graph.nodes[entry_name]
        spec = entry_node.spec

        # Create FlowData from frame using SourceSpec's default_path_id
        if isinstance(spec, SourceSpec):
            initial_data = FlowData(
                frame=frame,
                path_id=spec.default_path_id,
                timestamp_ns=getattr(frame, "t_src_ns", 0),
            )
        else:
            initial_data = FlowData(
                frame=frame,
                timestamp_ns=getattr(frame, "t_src_ns", 0),
            )

        return self.process_data(initial_data)

    def process_data(self, data: FlowData) -> List[FlowData]:
        """Process FlowData through the graph starting from entry.

        Args:
            data: Initial FlowData to process.

        Returns:
            List of FlowData that reached terminal nodes.
        """
        if not self._initialized:
            raise RuntimeError("Executor not initialized.")

        entry_name = self._graph.entry_node
        if entry_name is None:
            return []

        terminal_nodes = set(self._graph.get_terminal_nodes())
        terminal_results: List[FlowData] = []

        # BFS through the graph
        queue: deque[tuple[str, FlowData]] = deque()
        queue.append((entry_name, data))

        while queue:
            node_name, current_data = queue.popleft()
            node = self._graph.nodes[node_name]

            # Interpret the node's spec
            outputs = self._interpreter.interpret(node, current_data)

            # Route outputs to successors
            for output_data in outputs:
                successors = self._graph.get_successors(
                    node_name, output_data.path_id
                )

                if self._debug:
                    if successors:
                        print(f"[ROUTE] {node_name} -> {successors} (path={output_data.path_id})")
                    else:
                        print(f"[TERMINAL] {node_name} (path={output_data.path_id})")

                if not successors:
                    terminal_results.append(output_data)
                    if node_name in terminal_nodes:
                        self._graph.fire_triggers(output_data)
                else:
                    for successor in successors:
                        queue.append((successor, output_data))

        return terminal_results

    def process_batch(self, frames: List["Frame"]) -> List[List[FlowData]]:
        """Process multiple frames through the graph with batch optimization.

        When batch_size > 1, groups frames at ModuleSpec nodes and uses
        batch dispatch for modules with Capability.BATCHING. Non-module
        nodes (source, filter, sampler) are processed individually.

        Args:
            frames: List of frames to process.

        Returns:
            List of terminal FlowData lists, one per input frame.
        """
        if self._batch_size <= 1 or len(frames) <= 1:
            return [self.process(frame) for frame in frames]

        if not self._initialized:
            raise RuntimeError(
                "Executor not initialized. Use context manager or call initialize()."
            )

        entry_name = self._graph.entry_node
        if entry_name is None:
            return [[] for _ in frames]

        entry_node = self._graph.nodes[entry_name]
        spec = entry_node.spec

        # Create FlowData for each frame
        initial_datas: List[FlowData] = []
        for frame in frames:
            if isinstance(spec, SourceSpec):
                data = FlowData(
                    frame=frame,
                    path_id=spec.default_path_id,
                    timestamp_ns=getattr(frame, "t_src_ns", 0),
                )
            else:
                data = FlowData(
                    frame=frame,
                    timestamp_ns=getattr(frame, "t_src_ns", 0),
                )
            initial_datas.append(data)

        return self._process_batch_bfs(initial_datas)

    def _process_batch_bfs(
        self, initial_datas: List[FlowData]
    ) -> List[List[FlowData]]:
        """BFS through the graph processing batches of FlowData.

        At ModuleSpec nodes, all frames are processed together using
        batch dispatch. At other nodes, frames are processed individually.

        Args:
            initial_datas: List of initial FlowData items.

        Returns:
            List of terminal FlowData lists, one per input data.
        """
        n = len(initial_datas)
        terminal_nodes = set(self._graph.get_terminal_nodes())

        # Track results per original frame index
        terminal_results: List[List[FlowData]] = [[] for _ in range(n)]

        # Map each FlowData back to its original frame index
        # Queue items: (node_name, list of (frame_idx, FlowData))
        entry_name = self._graph.entry_node
        assert entry_name is not None

        queue: deque[tuple[str, List[tuple[int, FlowData]]]] = deque()
        queue.append((
            entry_name,
            [(i, d) for i, d in enumerate(initial_datas)],
        ))

        while queue:
            node_name, indexed_datas = queue.popleft()
            node = self._graph.nodes[node_name]

            # Batch dispatch for ModuleSpec nodes
            if isinstance(node.spec, ModuleSpec) and len(indexed_datas) > 1:
                outputs_per_frame = (
                    self._interpreter.interpret_modules_batch(
                        node, node.spec, [d for _, d in indexed_datas]
                    )
                )
            else:
                # Individual dispatch for other node types
                outputs_per_frame = [
                    self._interpreter.interpret(node, d)
                    for _, d in indexed_datas
                ]

            # Route outputs to successors, grouping by successor node
            successor_groups: dict[str, List[tuple[int, FlowData]]] = defaultdict(list)

            for (frame_idx, _), outputs in zip(indexed_datas, outputs_per_frame):
                for output_data in outputs:
                    successors = self._graph.get_successors(
                        node_name, output_data.path_id
                    )

                    if not successors:
                        terminal_results[frame_idx].append(output_data)
                        if node_name in terminal_nodes:
                            self._graph.fire_triggers(output_data)
                    else:
                        for successor in successors:
                            successor_groups[successor].append(
                                (frame_idx, output_data)
                            )

            # Enqueue successor groups
            for successor, group in successor_groups.items():
                queue.append((successor, group))

        return terminal_results
