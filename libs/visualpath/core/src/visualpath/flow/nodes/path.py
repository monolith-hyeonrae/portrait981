"""Path node for module execution.

PathNode declares modules to run on frames.
Execution is handled by the interpreter.
"""

from typing import List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowNode
from visualpath.flow.specs import ModuleSpec, NodeSpec

if TYPE_CHECKING:
    from visualpath.core.isolation import IsolationConfig
    from visualpath.core.module import Module


class PathNode(FlowNode):
    """Node that declares module execution.

    Usage:
        PathNode(name="analysis", modules=[face_detector, smile_trigger])

    Backend interprets the spec and runs modules on frames.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        modules: Optional[List["Module"]] = None,
        parallel: bool = False,
        join_window_ns: int = 100_000_000,
        isolation: Optional["IsolationConfig"] = None,
    ):
        """Initialize a PathNode.

        Args:
            name: Name for this node.
            modules: List of unified modules (analyzers and triggers).
            parallel: Whether independent modules can run in parallel.
            join_window_ns: Window for auto-joining parallel branches.
            isolation: Optional isolation configuration for module execution.

        Raises:
            ValueError: If neither 'name' nor 'modules' is provided.
        """
        if modules is None and name is None:
            raise ValueError("Either 'modules' or 'name' must be provided")

        self._modules: tuple = tuple(modules) if modules is not None else ()

        if name is None:
            if self._modules:
                name = f"path_{self._modules[0].name}"
            else:
                name = "path_empty"
        self._name = name

        self._parallel = parallel
        self._join_window_ns = join_window_ns
        self._isolation = isolation

    @property
    def name(self) -> str:
        return self._name

    @property
    def modules(self) -> tuple:
        """Get modules list."""
        return self._modules

    @property
    def spec(self) -> NodeSpec:
        """Get spec for this node."""
        return ModuleSpec(
            modules=self._modules,
            parallel=self._parallel,
            join_window_ns=self._join_window_ns,
            isolation=self._isolation,
        )

    def initialize(self) -> None:
        """Initialize all modules."""
        for module in self._modules:
            if hasattr(module, 'initialize'):
                module.initialize()

    def cleanup(self) -> None:
        """Cleanup all modules."""
        for module in self._modules:
            if hasattr(module, 'cleanup'):
                module.cleanup()
