"""Path abstraction for grouping modules.

A Path represents a logical grouping of related modules that share
common fusion logic. Examples:
- Human/When Path: face, pose, gesture modules with highlight fusion
- Scene/What Path: object, OCR, depth modules with scene fusion

Paths allow:
- Grouped configuration of modules
- Independent isolation levels per group
- Parallel execution of paths with different fusion strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from visualpath.core.observation import Observation
from visualpath.core.module import Module
from visualpath.core.isolation import IsolationLevel, IsolationConfig

if TYPE_CHECKING:
    from visualbase import Frame


@dataclass
class PathConfig:
    """Configuration for a single Path.

    Attributes:
        name: Unique name for this path.
        modules: List of module names to include.
        default_isolation: Default isolation level for modules in this path.
        module_config: Per-module configuration overrides.
    """

    name: str
    modules: List[str] = field(default_factory=list)
    default_isolation: IsolationLevel = IsolationLevel.INLINE
    module_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class Path:
    """A group of modules with shared fusion logic.

    A Path manages a set of modules and their fusion module,
    handling initialization, execution, and cleanup.

    Example:
        >>> path = Path(
        ...     name="human",
        ...     modules=[face_module, pose_module],
        ...     fusion=highlight_fusion,
        ... )
        >>> with path:
        ...     for frame in video:
        ...         results = path.process(frame)
        ...         for result in results:
        ...             if result.should_trigger:
        ...                 handle_trigger(result)
    """

    def __init__(
        self,
        name: str,
        modules: Optional[List[Module]] = None,
        fusion: Optional[Module] = None,
        isolation_config: Optional[IsolationConfig] = None,
        max_workers: Optional[int] = None,
        # Legacy alias
        analyzers: Optional[List[Module]] = None,
    ):
        """Initialize a Path.

        Args:
            name: Unique name for this path.
            modules: List of module instances.
            fusion: Optional fusion module for combining observations.
            isolation_config: Optional isolation configuration.
            max_workers: Max thread workers for parallel execution.
            analyzers: Deprecated alias for modules.
        """
        self._name = name
        self._modules = modules or analyzers or []
        self._fusion = fusion
        self._isolation_config = isolation_config or IsolationConfig()
        self._max_workers = max_workers or len(self._modules)

        self._initialized = False

    @property
    def name(self) -> str:
        """Get the path name."""
        return self._name

    @property
    def modules(self) -> List[Module]:
        """Get the list of modules."""
        return self._modules

    @property
    def analyzers(self) -> List[Module]:
        """Get the list of modules (legacy alias)."""
        return self._modules

    @property
    def fusion(self) -> Optional[Module]:
        """Get the fusion module."""
        return self._fusion

    def add_module(self, module: Module) -> None:
        """Add a module to this path.

        Args:
            module: Module instance to add.
        """
        self._modules.append(module)

    def initialize(self) -> None:
        """Initialize all modules and fusion module."""
        if self._initialized:
            return

        # Validate dependencies before initialization
        self._validate_dependencies()

        for module in self._modules:
            module.initialize()

        self._initialized = True

    def _validate_dependencies(self) -> None:
        """Validate that all module dependencies are satisfied.

        Checks:
        - All depends are provided by earlier modules or external deps
        - No circular dependencies within this path

        Raises:
            ValueError: If dependencies are not satisfied.
        """
        available: set[str] = set()

        for module in self._modules:
            # Check if all depends are available
            depends = set(module.depends) if module.depends else set()
            missing = depends - available

            if missing:
                raise ValueError(
                    f"Module '{module.name}' depends on {missing}, "
                    f"but only {available or 'nothing'} is available. "
                    f"Reorder modules or provide external_deps."
                )

            # This module's output is now available for subsequent modules
            available.add(module.name)

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph for modules in this path.

        Returns:
            Dict mapping module names to their dependencies.
        """
        return {
            mod.name: list(mod.depends) if mod.depends else []
            for mod in self._modules
        }

    def cleanup(self) -> None:
        """Clean up all modules and fusion module."""
        if not self._initialized:
            return

        for module in self._modules:
            module.cleanup()

        self._initialized = False

    def __enter__(self) -> "Path":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def analyze_all(
        self,
        frame: "Frame",
        external_deps: Optional[Dict[str, Observation]] = None,
    ) -> List[Observation]:
        """Run all modules on a frame.

        Args:
            frame: The input frame to process.
            external_deps: Optional observations from previous modules
                          (e.g., from upstream PathNodes in FlowGraph).

        Returns:
            List of observations from all modules.
        """
        if not self._initialized:
            raise RuntimeError("Path not initialized. Use context manager or call initialize().")

        return self._run_modules(frame, external_deps)

    def _run_modules(
        self,
        frame: "Frame",
        external_deps: Optional[Dict[str, Observation]] = None,
    ) -> List[Observation]:
        """Run modules with dependency resolution.

        Modules are run in order, with each module receiving
        observations from its dependencies.
        """
        # Build deps dict from external deps
        deps: Dict[str, Observation] = dict(external_deps) if external_deps else {}
        observations: List[Observation] = []

        for module in self._modules:
            # Build deps for this module
            module_deps = None
            if module.depends:
                module_deps = {
                    name: deps[name]
                    for name in module.depends
                    if name in deps
                }

            obs = module.process(frame, module_deps)
            if obs is not None:
                observations.append(obs)
                # Add to deps for subsequent modules
                deps[module.name] = obs

        return observations

    def process(self, frame: "Frame") -> List[Observation]:
        """Process a frame through modules and fusion.

        Args:
            frame: The input frame to process.

        Returns:
            List of Observations from fusion processing.
        """
        observations = self.analyze_all(frame)

        if self._fusion is None:
            # No fusion - return empty results
            return []

        # Aggregate all observations and call fusion once
        deps = {obs.source: obs for obs in observations}
        return [self._fusion.process(frame, deps)]


class PathOrchestrator:
    """Orchestrates multiple Paths.

    Runs multiple paths in parallel or sequentially, collecting
    results from all paths.

    Example:
        >>> orchestrator = PathOrchestrator([human_path, scene_path])
        >>> with orchestrator:
        ...     for frame in video:
        ...         all_results = orchestrator.process_all(frame)
        ...         for path_name, results in all_results.items():
        ...             for result in results:
        ...                 if result.should_trigger:
        ...                     handle_trigger(path_name, result)
    """

    def __init__(
        self,
        paths: List[Path],
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """Initialize the orchestrator.

        Args:
            paths: List of Path instances to orchestrate.
            parallel: Whether to run paths in parallel.
            max_workers: Max thread workers for parallel path execution.
        """
        self._paths = paths
        self._parallel = parallel
        self._max_workers = max_workers or len(paths)

        self._initialized = False
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def paths(self) -> List[Path]:
        """Get the list of paths."""
        return self._paths

    def initialize(self) -> None:
        """Initialize all paths."""
        if self._initialized:
            return

        for path in self._paths:
            path.initialize()

        if self._parallel and len(self._paths) > 1:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        self._initialized = True

    def cleanup(self) -> None:
        """Clean up all paths."""
        if not self._initialized:
            return

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        for path in self._paths:
            path.cleanup()

        self._initialized = False

    def __enter__(self) -> "PathOrchestrator":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def process_all(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Process a frame through all paths.

        Args:
            frame: The input frame to process.

        Returns:
            Dict mapping path names to their Observations.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Use context manager or call initialize().")

        if self._parallel and self._executor and len(self._paths) > 1:
            return self._process_parallel(frame)
        else:
            return self._process_sequential(frame)

    def _process_sequential(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Process paths sequentially."""
        results = {}
        for path in self._paths:
            results[path.name] = path.process(frame)
        return results

    def _process_parallel(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Process paths in parallel."""
        if self._executor is None:
            return self._process_sequential(frame)

        futures = {
            self._executor.submit(path.process, frame): path
            for path in self._paths
        }

        results = {}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path.name] = future.result()
            except Exception:
                results[path.name] = []

        return results

    def analyze_all(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Extract observations from all paths without fusion.

        Args:
            frame: The input frame to process.

        Returns:
            Dict mapping path names to their observations.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized.")

        results = {}
        for path in self._paths:
            results[path.name] = path.analyze_all(frame)
        return results
