"""SimpleBackend implementation using GraphExecutor.

SimpleBackend executes FlowGraph pipelines using the GraphExecutor,
processing frames sequentially through the DAG.
"""

from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

from visualpath.backends.base import ExecutionBackend, PipelineResult

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData


class SimpleBackend(ExecutionBackend):
    """GraphExecutor-based sequential execution backend.

    SimpleBackend processes frames sequentially through a FlowGraph using
    GraphExecutor. It is the default backend for local video processing
    and development/debugging.

    For complex scheduling, sampling, or branching, construct a FlowGraph
    with the appropriate nodes (SamplerNode, RateLimiterNode, JoinNode, etc.).

    Examples:
        >>> from visualpath.flow.graph import FlowGraph
        >>> graph = FlowGraph.from_modules([face_ext, smile_fusion])
        >>> backend = SimpleBackend()
        >>> result = backend.execute(frames, graph)
        >>> print(result.triggers)

        >>> # With FlowGraphBuilder for complex pipelines
        >>> from visualpath.flow import FlowGraphBuilder
        >>> graph = (FlowGraphBuilder()
        ...     .source("frames")
        ...     .sample(every_nth=3)
        ...     .path("main", analyzers=[face_ext], fusion=smile_fusion)
        ...     .build())
        >>> result = backend.execute(frames, graph)
    """

    def execute(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        *,
        on_frame: Optional[Callable[["Frame", List["FlowData"]], bool]] = None,
    ) -> PipelineResult:
        """Execute a FlowGraph-based pipeline.

        Processing flow:
        1. Initialize all graph nodes
        2. Process each frame through GraphExecutor
        3. Call on_frame callback if provided (stop on False)
        4. Collect triggers from FlowData results
        5. Clean up all graph nodes

        Args:
            frames: Iterator of Frame objects (not materialized to list).
            graph: FlowGraph defining the pipeline.
            on_frame: Optional per-frame callback. See ExecutionBackend.execute().

        Returns:
            PipelineResult with triggers and frame count.
        """
        from visualpath.backends.simple.executor import GraphExecutor

        triggers = []
        frame_count = 0

        executor = GraphExecutor(graph)

        with executor:
            for frame in frames:
                frame_count += 1
                terminal_results = executor.process(frame)

                # Collect triggers directly from terminal results
                for data in terminal_results:
                    for result in data.results:
                        if result.should_trigger and result.trigger:
                            triggers.append(result.trigger)

                if on_frame is not None:
                    if not on_frame(frame, terminal_results):
                        break

        return PipelineResult(
            triggers=triggers,
            frame_count=frame_count,
        )


__all__ = ["SimpleBackend"]
