"""Base execution backend interface.

ExecutionBackend defines the single abstract interface for pipeline execution.
Backends are spec interpreters — they read the FlowGraph (AST) and execute it.

Each backend provides its own interpretation of NodeSpecs:
- SimpleBackend: uses SimpleInterpreter for synchronous execution
- PathwayBackend: converts specs to Pathway operators for streaming
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame, Trigger
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData


@dataclass
class PipelineResult:
    """Result from executing a pipeline.

    Attributes:
        triggers: List of triggers that fired during processing.
        frame_count: Total frames processed.
        stats: Optional backend-specific statistics.
    """

    triggers: List["Trigger"] = field(default_factory=list)
    frame_count: int = 0
    stats: dict = field(default_factory=dict)


class ExecutionBackend(ABC):
    """Abstract base class for pipeline execution backends.

    One method: ``execute(frames, graph) -> PipelineResult``.

    Example:
        >>> graph = FlowGraph.from_modules([face_ext, smile_fusion])
        >>> result = SimpleBackend().execute(frames, graph)
    """

    @property
    def name(self) -> str:
        """Backend identifier name."""
        return self.__class__.__name__

    @abstractmethod
    def execute(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        *,
        on_frame: Optional[Callable[["Frame", List["FlowData"]], bool]] = None,
    ) -> PipelineResult:
        """Execute a FlowGraph-based pipeline.

        Args:
            frames: Iterator of Frame objects from video source.
            graph: FlowGraph defining the processing pipeline.
            on_frame: Optional per-frame callback ``(frame, terminal_results) -> bool``.
                Called after each frame is processed. Return True to continue,
                False to stop early. ``terminal_results`` is a list of FlowData
                from terminal nodes.

        Returns:
            PipelineResult with triggers, frame_count, and optional stats.
        """
        ...


__all__ = ["ExecutionBackend", "PipelineResult"]
