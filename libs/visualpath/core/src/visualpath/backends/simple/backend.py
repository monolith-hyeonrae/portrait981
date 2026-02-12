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

    Args:
        batch_size: Number of frames to collect before batch processing.
            When > 1, modules with Capability.BATCHING receive frames via
            process_batch() for GPU batch inference optimization.
            Default 1 (frame-by-frame, backward compatible).

    Examples:
        >>> from visualpath.flow.graph import FlowGraph
        >>> graph = FlowGraph.from_modules([face_ext, smile_fusion])
        >>> backend = SimpleBackend()
        >>> result = backend.execute(frames, graph)
        >>> print(result.triggers)

        >>> # With batch processing for GPU modules
        >>> backend = SimpleBackend(batch_size=8)
        >>> result = backend.execute(frames, graph)
    """

    def __init__(self, batch_size: int = 1):
        self._batch_size = max(1, batch_size)

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

        triggers: list = []
        frame_count = 0

        executor = GraphExecutor(graph, batch_size=self._batch_size)

        with executor:
            if self._batch_size <= 1:
                # Frame-by-frame (original path)
                for frame in frames:
                    frame_count += 1
                    terminal_results = executor.process(frame)
                    self._collect_triggers(terminal_results, triggers)
                    if on_frame is not None:
                        if not on_frame(frame, terminal_results):
                            break
            else:
                # Batch collection
                batch: list = []
                stopped = False
                for frame in frames:
                    frame_count += 1
                    batch.append(frame)

                    if len(batch) >= self._batch_size:
                        stopped = self._flush_batch(
                            executor, batch, triggers, on_frame
                        )
                        batch = []
                        if stopped:
                            break

                # Flush remaining frames
                if batch and not stopped:
                    self._flush_batch(executor, batch, triggers, on_frame)

        return PipelineResult(
            triggers=triggers,
            frame_count=frame_count,
        )

    @staticmethod
    def _collect_triggers(
        terminal_results: list,
        triggers: list,
    ) -> None:
        """Collect triggers from terminal FlowData results."""
        for data in terminal_results:
            for result in data.results:
                if result.should_trigger and result.trigger:
                    triggers.append(result.trigger)

    @staticmethod
    def _flush_batch(
        executor: "GraphExecutor",
        batch: list,
        triggers: list,
        on_frame,
    ) -> bool:
        """Process a batch of frames and collect triggers.

        Returns True if on_frame callback requested stop.
        """
        batch_results = executor.process_batch(batch)
        for frame, terminal_results in zip(batch, batch_results):
            SimpleBackend._collect_triggers(terminal_results, triggers)
            if on_frame is not None:
                if not on_frame(frame, terminal_results):
                    return True
        return False


__all__ = ["SimpleBackend"]
