"""Pathway pipeline builder for facemoment.

.. deprecated::
    FacemomentPipeline is deprecated. Use ``facemoment.main.run()`` instead,
    which routes all execution through FlowGraph → Backend.execute().

    The CUDA conflict detection logic has been moved to
    ``facemoment.main._build_isolation_config()`` and is handled
    transparently by WorkerBackend.

This module is retained for backwards compatibility. New code should use
``fm.run()`` or ``facemoment.main.build_graph()`` directly.
"""

import warnings
from typing import Callable, Dict, Iterator, List, Optional, Set, Union
import logging

from visualbase import Frame, Trigger

from vpx.sdk import Observation

logger = logging.getLogger(__name__)

# CUDA conflict groups: analyzers sharing the same CUDA runtime binding.
_CUDA_GROUPS: Dict[str, Set[str]] = {
    "onnxruntime": {"face.detect", "face.expression"},
    "torch": {"body.pose"},
}

# Known inter-worker dependencies (mirrors Module.depends declarations).
# Used by _run_workers_parallel() to phase workers correctly without
# importing analyzer classes (which would cause CUDA conflicts).
_KNOWN_WORKER_DEPS: Dict[str, List[str]] = {
    "face.expression": ["face.detect"],
    "face.classify": ["face.detect"],
}

# Check for Pathway availability
try:
    from visualpath.backends.pathway import PathwayBackend
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


class FacemomentPipeline:
    """Pathway-based pipeline with facemoment-specific logic.

    .. deprecated::
        Use ``facemoment.main.run()`` instead for batch execution.
        All batch execution now goes through FlowGraph → Backend.execute()
        with isolation handled by WorkerBackend.

    This class is retained for backwards compatibility.
    ``run()`` delegates to the unified FlowGraph execution path.
    ``initialize()``/``cleanup()`` manage analyzer lifecycle for
    consumers that iterate analyzers directly (e.g. debug command).
    """

    def __init__(
        self,
        analyzers: Optional[List[str]] = None,
        fusion_config: Optional[Dict] = None,
        window_ns: int = 100_000_000,
        auto_inject_classifier: bool = True,
        venv_paths: Optional[Dict[str, str]] = None,
        distributed: bool = False,
    ):
        warnings.warn(
            "FacemomentPipeline is deprecated. Use facemoment.main.run() instead. "
            "All execution now goes through FlowGraph → Backend.execute().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._analyzer_names = analyzers or ["face.detect", "face.expression", "body.pose", "hand.gesture"]
        self._fusion_config = fusion_config or {}
        self._window_ns = window_ns
        self._auto_inject_classifier = auto_inject_classifier
        self._venv_paths = venv_paths or {}
        self._distributed = distributed
        self._analyzers = []
        self._workers: Dict[str, object] = {}
        self._fusion = None
        self._classifier = None
        self._initialized = False
        self.actual_backend: Optional[str] = None

    _TORCH_ANALYZERS = frozenset({"body.pose"})

    @staticmethod
    def _detect_cuda_conflicts(names: List[str]) -> Set[str]:
        """Detect CUDA conflicts among active analyzers.

        .. deprecated::
            Use ``facemoment.main._detect_cuda_conflicts()`` instead.
        """
        from facemoment.main import _detect_cuda_conflicts
        return _detect_cuda_conflicts(names)

    # ------------------------------------------------------------------
    # Lifecycle: initialize / cleanup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load analyzers and fusion for direct iteration.

        Used by consumers that iterate ``self.analyzers`` directly
        (e.g. the debug command's inline frame-by-frame loop).
        For batch execution, use ``run()`` instead.
        """
        self._analyzers = self._build_analyzers()
        self._fusion = self._build_fusion()

        # Start workers
        for name in list(self._workers):
            try:
                self._workers[name].start()
            except Exception as exc:
                logger.warning("Worker '%s' failed to start: %s — removing", name, exc)
                del self._workers[name]

        # Initialize analyzers
        for ext in self._analyzers:
            try:
                ext.initialize()
            except Exception:
                logger.debug("Analyzer '%s' initialize() failed (non-fatal)", ext.name)

        self._initialized = True

    def cleanup(self) -> None:
        """Stop workers and clean up analyzers."""
        for name, worker in list(self._workers.items()):
            try:
                worker.stop()
            except Exception:
                pass
        self._workers.clear()

        for ext in self._analyzers:
            try:
                ext.cleanup()
            except Exception:
                pass

        self._initialized = False

    # ------------------------------------------------------------------
    # Internal: build analyzers and fusion
    # ------------------------------------------------------------------

    def _build_analyzers(self) -> list:
        """Create analyzer instances from names.

        Handles:
        - CUDA conflict detection → minority group to subprocess workers
        - Auto-inject face_classifier when face/face_detect is present
        """
        from visualpath.plugin import create_analyzer
        from facemoment.main import _detect_cuda_conflicts

        names = list(self._analyzer_names)

        # Auto-inject classifier
        if self._auto_inject_classifier:
            if "face.detect" in names:
                if "face.classify" not in names:
                    names.append("face.classify")

        # Detect CUDA conflicts
        isolated = _detect_cuda_conflicts(names)

        # --distributed: force all vpx analyzers into subprocess
        if self._distributed:
            from visualpath.plugin import discover_analyzers
            ep_map = discover_analyzers()
            for name in names:
                if name in ep_map:
                    module_path = ep_map[name].value.split(":")[0]
                    if module_path.startswith("vpx."):
                        isolated.add(name)

        analyzers = []
        for name in names:
            # 1) venv path specified → VenvWorker (regardless of CUDA conflict)
            if name in self._venv_paths:
                try:
                    from visualpath.process.launcher import VenvWorker
                    worker = VenvWorker(
                        analyzer_name=name,
                        venv_path=self._venv_paths[name],
                    )
                    self._workers[name] = worker
                    logger.info("Analyzer '%s' will run in venv '%s'", name, self._venv_paths[name])
                except Exception as exc:
                    logger.warning("Failed to create VenvWorker for '%s': %s", name, exc)
                continue

            # 2) CUDA conflict → ProcessWorker
            if name in isolated:
                # Create subprocess worker for isolated analyzers
                try:
                    from visualpath.process.launcher import ProcessWorker
                    worker = ProcessWorker(analyzer_name=name)
                    self._workers[name] = worker
                    logger.info("Analyzer '%s' will run in subprocess (CUDA conflict)", name)
                except Exception as exc:
                    logger.warning("Failed to create worker for '%s': %s", name, exc)
                continue

            try:
                ext = create_analyzer(name)
                analyzers.append(ext)
                if name == "face.classify":
                    self._classifier = ext
            except Exception as exc:
                logger.warning("Failed to load analyzer '%s': %s", name, exc)

        # Build worker dependency map (only deps among active workers)
        self._worker_depends = {
            name: [d for d in _KNOWN_WORKER_DEPS.get(name, []) if d in self._workers]
            for name in self._workers
        }

        return analyzers

    def _build_fusion(self):
        """Create fusion instance from config."""
        from facemoment.moment_detector.fusion import HighlightFusion

        cooldown = self._fusion_config.get("cooldown_sec", 2.0)
        main_only = self._fusion_config.get("main_only", True)
        return HighlightFusion(cooldown_sec=cooldown, main_only=main_only)

    def _merge_observations(self, obs_list: list, frame) -> Observation:
        """Merge multiple observations into a single merged observation."""
        from facemoment.pipeline.utils import merge_observations
        return merge_observations(obs_list, frame)

    # ------------------------------------------------------------------
    # Batch execution: delegates to FlowGraph
    # ------------------------------------------------------------------

    def run(
        self,
        frames: Union[Iterator[Frame], List[Frame]],
        on_trigger: Optional[Callable[[Trigger], None]] = None,
    ) -> List[Trigger]:
        """Run the facemoment pipeline.

        Delegates to the unified FlowGraph → Backend execution path.
        """
        from facemoment.main import build_modules, build_graph, _build_isolation_config

        # Build modules
        cooldown = self._fusion_config.get("cooldown_sec", 2.0)
        main_only = self._fusion_config.get("main_only", True)
        modules = build_modules(
            self._analyzer_names,
            cooldown=cooldown,
            main_only=main_only,
        )

        # Build isolation config
        isolation_config = _build_isolation_config(self._analyzer_names)

        # Build graph
        graph = build_graph(modules, isolation=isolation_config, on_trigger=on_trigger)

        # Select backend
        from facemoment.main import _get_backend
        engine = _get_backend("pathway" if PATHWAY_AVAILABLE else "simple",
                              has_isolation=isolation_config is not None)
        self.actual_backend = engine.name

        # Execute
        frame_iter = iter(frames) if isinstance(frames, list) else frames
        pipeline_result = engine.execute(frame_iter, graph)

        return pipeline_result.triggers

    def analyzer_runtime_info(self) -> Dict[str, "RuntimeInfo"]:
        """Return runtime info for all analyzers (inline + workers)."""
        from visualpath.core.module import RuntimeInfo

        result = {}
        for ext in self._analyzers:
            result[ext.name] = ext.runtime_info
        for name, worker in self._workers.items():
            info = worker.worker_info
            result[name] = RuntimeInfo(isolation=info.isolation_level, pid=info.pid)
        return result

    @property
    def analyzers(self) -> List:
        return self._analyzers

    @property
    def fusion(self):
        return self._fusion

    @property
    def workers(self) -> Dict:
        return self._workers

    @property
    def worker_depends(self) -> Dict[str, List[str]]:
        return self._worker_depends


__all__ = ["FacemomentPipeline", "PATHWAY_AVAILABLE"]
