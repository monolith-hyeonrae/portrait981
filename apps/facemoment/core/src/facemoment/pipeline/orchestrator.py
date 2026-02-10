"""PipelineOrchestrator - A-B*-C-A distributed pipeline orchestration.

.. deprecated::
    PipelineOrchestrator is deprecated. Use ``facemoment.main.run()`` instead,
    which routes all execution through FlowGraph → Backend.execute().

    Worker isolation is now handled transparently by WorkerBackend via
    ModuleSpec.isolation configuration.

This module is retained for backwards compatibility.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from visualbase import ClipResult, Frame, Trigger, VisualBase, FileSource

from visualpath.core import IsolationLevel
from visualpath.process import (
    WorkerLauncher,
    BaseWorker,
    WorkerResult,
)
from visualpath.plugin import create_analyzer, load_fusion

from facemoment.pipeline.config import AnalyzerConfig, FusionConfig, PipelineConfig
from facemoment.moment_detector.fusion.base import BaseFusion
from vpx.sdk import Observation

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    frames_processed: int = 0
    triggers_fired: int = 0
    clips_extracted: int = 0
    processing_time_sec: float = 0.0
    avg_frame_time_ms: float = 0.0
    worker_stats: Dict[str, Dict[str, Any]] = None
    backend_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.worker_stats is None:
            self.worker_stats = {}
        if self.backend_stats is None:
            self.backend_stats = {}


class PipelineOrchestrator:
    """Orchestrates the A-B*-C-A distributed pipeline.

    .. deprecated::
        Use ``facemoment.main.run()`` instead. All execution now goes through
        FlowGraph → Backend.execute() with isolation handled by WorkerBackend.

    This class is retained for backwards compatibility. Internally it
    delegates to the unified FlowGraph execution path where possible.
    """

    def __init__(
        self,
        analyzer_configs: List[AnalyzerConfig],
        fusion: Optional[BaseFusion] = None,
        fusion_config: Optional[FusionConfig] = None,
        clip_output_dir: Optional[Path] = None,
        backend: str = "pathway",
    ):
        warnings.warn(
            "PipelineOrchestrator is deprecated. Use facemoment.main.run() instead. "
            "All execution now goes through FlowGraph → Backend.execute().",
            DeprecationWarning,
            stacklevel=2,
        )

        if not analyzer_configs:
            raise ValueError("At least one analyzer config is required")

        self._analyzer_configs = analyzer_configs
        self._fusion = fusion
        self._fusion_config = fusion_config or FusionConfig()
        self._clip_output_dir = Path(clip_output_dir) if clip_output_dir else Path("./clips")
        self._backend = backend

        self._workers: Dict[str, BaseWorker] = {}
        self._vb: Optional[VisualBase] = None
        self._initialized = False

        self._on_frame: Optional[Callable[[Frame], None]] = None
        self._on_observations: Optional[Callable[[List[Observation]], None]] = None
        self._on_trigger: Optional[Callable[[Trigger, Observation], None]] = None

        self._stats = PipelineStats()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelineOrchestrator":
        """Create orchestrator from a PipelineConfig."""
        return cls(
            analyzer_configs=config.analyzers,
            fusion_config=config.fusion,
            clip_output_dir=Path(config.clip_output_dir),
            backend=config.backend,
        )

    def _create_fusion(self) -> BaseFusion:
        """Create fusion instance from config."""
        if self._fusion is not None:
            return self._fusion

        try:
            FusionClass = load_fusion(self._fusion_config.name)
            kwargs = {
                "cooldown_sec": self._fusion_config.cooldown_sec,
                **self._fusion_config.kwargs,
            }
            return FusionClass(**kwargs)
        except (KeyError, ImportError):
            from facemoment.moment_detector.fusion import HighlightFusion
            return HighlightFusion(
                cooldown_sec=self._fusion_config.cooldown_sec,
                **self._fusion_config.kwargs,
            )

    def run(
        self,
        video_path: str,
        fps: int = 10,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> List[ClipResult]:
        """Run the complete pipeline on a video file.

        Delegates to the unified FlowGraph → Backend execution path.
        """
        from facemoment.main import (
            build_modules,
            build_graph,
            _build_isolation_config,
            _get_backend,
        )
        from visualpath.core.isolation import IsolationConfig, IsolationLevel as IL

        clips: List[ClipResult] = []
        start_time = time.time()
        self._stats = PipelineStats()

        # Build modules from analyzer configs
        analyzer_names = [c.name for c in self._analyzer_configs]
        modules = build_modules(
            analyzer_names,
            cooldown=self._fusion_config.cooldown_sec,
        )

        # Build isolation config from analyzer configs
        overrides = {}
        venv_paths = {}
        for config in self._analyzer_configs:
            if config.effective_isolation > IL.INLINE:
                overrides[config.name] = config.effective_isolation
            if config.venv_path:
                venv_paths[config.name] = config.venv_path

        isolation_config = None
        if overrides:
            isolation_config = IsolationConfig(
                default_level=IL.INLINE,
                overrides=overrides,
                venv_paths=venv_paths,
            )

        # If no explicit isolation, fall back to CUDA conflict detection
        if isolation_config is None:
            isolation_config = _build_isolation_config(analyzer_names)

        # Build graph
        triggers = []

        def on_trigger_internal(data):
            for result in data.results:
                if result.should_trigger and result.trigger:
                    trigger = result.trigger
                    triggers.append(trigger)
                    self._stats.triggers_fired += 1

                    if self._on_trigger:
                        self._on_trigger(trigger, result)

        graph = build_graph(modules, isolation=isolation_config, on_trigger=on_trigger_internal)

        # Select backend
        engine = _get_backend(self._backend)

        # Open video and execute
        self._clip_output_dir.mkdir(parents=True, exist_ok=True)
        self._vb = VisualBase(clip_output_dir=self._clip_output_dir)
        self._vb.connect(FileSource(video_path))

        try:
            frames = self._vb.get_stream(fps=fps, resolution=resolution)
            pipeline_result = engine.execute(frames, graph)
            self._stats.frames_processed = pipeline_result.frame_count
            self._stats.backend_stats = pipeline_result.stats

            # Extract clips from triggers
            for trigger in triggers:
                clip_result = self._vb.trigger(trigger)
                clips.append(clip_result)
                if clip_result.success:
                    self._stats.clips_extracted += 1

            self._vb.disconnect()
        except Exception:
            if self._vb:
                self._vb.disconnect()
            raise

        elapsed = time.time() - start_time
        self._stats.processing_time_sec = elapsed
        if self._stats.frames_processed > 0:
            self._stats.avg_frame_time_ms = (elapsed * 1000) / self._stats.frames_processed

        return clips

    def run_stream(
        self,
        video_path: str,
        fps: int = 10,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Iterator[Tuple[Frame, List[Observation], Optional[Observation]]]:
        """Run pipeline as a stream.

        .. deprecated::
            Streaming mode is not supported in the unified path.
            Use ``facemoment.main.run()`` for batch processing.
        """
        warnings.warn(
            "PipelineOrchestrator.run_stream() is deprecated and may not "
            "work correctly. Use facemoment.main.run() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Yield nothing — streaming not supported via FlowGraph path
        return iter([])

    def set_on_frame(self, callback: Callable[[Frame], None]) -> None:
        self._on_frame = callback

    def set_on_observations(
        self, callback: Callable[[List[Observation]], None]
    ) -> None:
        self._on_observations = callback

    def set_on_trigger(
        self, callback: Callable[[Trigger, Observation], None]
    ) -> None:
        self._on_trigger = callback

    def get_stats(self) -> PipelineStats:
        return self._stats

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def worker_names(self) -> List[str]:
        return list(self._workers.keys())

    @property
    def clip_output_dir(self) -> Path:
        return self._clip_output_dir
