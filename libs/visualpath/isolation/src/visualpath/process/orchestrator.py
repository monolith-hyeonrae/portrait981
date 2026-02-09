"""AnalyzerOrchestrator - Thread-parallel analyzer execution.

Provides thread-based parallelism for running multiple analyzers
simultaneously on the same frame. Useful for Library mode where
analyzers run in the same process.

Architecture:
    Frame ──→ [Thread Pool] ──→ Observations
                  │
                  ├── Analyzer1 (thread 1)
                  ├── Analyzer2 (thread 2)
                  └── Analyzer3 (thread 3)

Example:
    >>> from visualpath.process import AnalyzerOrchestrator
    >>>
    >>> orchestrator = AnalyzerOrchestrator(
    ...     analyzers=[ext1, ext2, ext3],
    ...     max_workers=3,
    ... )
    >>> orchestrator.initialize()
    >>> for frame in frames:
    ...     observations = orchestrator.analyze_all(frame)
    ...     for obs in observations:
    ...         process(obs)
    >>> orchestrator.cleanup()
"""

import time
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, TimeoutError

from visualbase import Frame

from visualpath.core.observation import Observation
from visualpath.core.module import Module
from visualpath.observability import ObservabilityHub

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


class AnalyzerOrchestrator:
    """Orchestrates parallel execution of multiple analyzers.

    Manages a pool of analyzers and runs them in parallel using threads.
    Collects observations and provides unified access to results.

    Args:
        analyzers: List of analyzers to run in parallel.
        max_workers: Maximum number of worker threads. Default: len(analyzers).
        timeout: Timeout in seconds for each analysis. Default: 5.0.
        observability_hub: Optional custom observability hub (uses global if None).

    Thread Safety:
        - Each analyzer runs in its own thread
        - Observations are collected thread-safely
        - Initialize/cleanup are called from the main thread
    """

    def __init__(
        self,
        analyzers: List[Module],
        max_workers: Optional[int] = None,
        timeout: float = 5.0,
        observability_hub: Optional[ObservabilityHub] = None,
    ):
        if not analyzers:
            raise ValueError("At least one analyzer is required")

        self._analyzers = analyzers
        self._max_workers = max_workers or len(analyzers)
        self._timeout = timeout
        self._hub = observability_hub or _hub

        self._executor: Optional[ThreadPoolExecutor] = None
        self._initialized = False

        # Stats
        self._frames_processed = 0
        self._total_observations = 0
        self._timeouts = 0
        self._errors = 0
        self._total_time_ns = 0

    def initialize(self) -> None:
        """Initialize all analyzers and create thread pool.

        Must be called before analyze_all().
        """
        if self._initialized:
            return

        # Initialize all analyzers
        for ext in self._analyzers:
            try:
                ext.initialize()
                logger.debug(f"Initialized analyzer: {ext.name}")
            except Exception as e:
                logger.error(f"Failed to initialize {ext.name}: {e}")
                raise

        # Create thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="analyzer_",
        )

        self._initialized = True
        logger.info(
            f"AnalyzerOrchestrator initialized with {len(self._analyzers)} analyzers, "
            f"{self._max_workers} workers"
        )

    def analyze_all(self, frame: Frame) -> List[Observation]:
        """Run all analyzers on a frame in parallel.

        Args:
            frame: Frame to process.

        Returns:
            List of observations from all analyzers.
            May be fewer than number of analyzers if some return None.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized or self._executor is None:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        start_ns = time.perf_counter_ns()
        observations: List[Observation] = []
        timed_out_analyzers: List[str] = []

        # Submit all analyzers
        futures: Dict[Future, Module] = {}
        for ext in self._analyzers:
            future = self._executor.submit(self._safe_analyze, ext, frame)
            futures[future] = ext

        # Collect results with timeout
        try:
            for future in as_completed(futures, timeout=self._timeout):
                ext = futures[future]
                try:
                    obs = future.result()
                    if obs is not None:
                        observations.append(obs)
                        self._total_observations += 1
                except TimeoutError:
                    logger.warning(f"Analyzer {ext.name} timed out")
                    self._timeouts += 1
                    timed_out_analyzers.append(ext.name)
                except Exception as e:
                    logger.error(f"Analyzer {ext.name} error: {e}")
                    self._errors += 1
        except TimeoutError:
            # Some futures didn't complete in time
            for future, ext in futures.items():
                if not future.done():
                    timed_out_analyzers.append(ext.name)
                    self._timeouts += 1
                    logger.warning(f"Analyzer {ext.name} timed out")

        self._frames_processed += 1
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._total_time_ns += elapsed_ns

        # Emit observability records
        if self._hub.enabled:
            processing_ms = elapsed_ns / 1_000_000
            self._emit_timing(frame.frame_id, processing_ms)
            if timed_out_analyzers:
                self._emit_timeout(frame.frame_id, timed_out_analyzers)

        return observations

    def _emit_timing(self, frame_id: int, processing_ms: float) -> None:
        """Emit timing record. Override for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="timing",
            frame_id=frame_id,
            data={
                "component": "orchestrator",
                "processing_ms": processing_ms,
                "threshold_ms": self._timeout * 1000,
                "is_slow": processing_ms > self._timeout * 1000,
            },
        ))

    def _emit_timeout(self, frame_id: int, timed_out: List[str]) -> None:
        """Emit timeout/frame drop record. Override for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="frame_drop",
            frame_id=frame_id,
            data={
                "dropped_frame_ids": [frame_id],
                "reason": f"timeout:{','.join(timed_out)}",
            },
        ))

    def _safe_analyze(
        self, analyzer: Module, frame: Frame
    ) -> Optional[Observation]:
        """Safely run analysis with error handling.

        Args:
            analyzer: Analyzer to run.
            frame: Frame to process.

        Returns:
            Observation or None on error.
        """
        try:
            return analyzer.process(frame)
        except Exception as e:
            logger.error(f"Analyze error in {analyzer.name}: {e}")
            self._errors += 1
            return None

    def analyze_sequential(self, frame: Frame) -> List[Observation]:
        """Run all analyzers sequentially (no parallelism).

        Useful for debugging or when parallelism is not needed.

        Args:
            frame: Frame to process.

        Returns:
            List of observations from all analyzers.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        start_ns = time.perf_counter_ns()
        observations: List[Observation] = []

        for ext in self._analyzers:
            try:
                obs = ext.process(frame)
                if obs is not None:
                    observations.append(obs)
                    self._total_observations += 1
            except Exception as e:
                logger.error(f"Analyze error in {ext.name}: {e}")
                self._errors += 1

        self._frames_processed += 1
        self._total_time_ns += time.perf_counter_ns() - start_ns

        return observations

    def cleanup(self) -> None:
        """Clean up all analyzers and shutdown thread pool."""
        if not self._initialized:
            return

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Cleanup analyzers
        for ext in self._analyzers:
            try:
                ext.cleanup()
                logger.debug(f"Cleaned up analyzer: {ext.name}")
            except Exception as e:
                logger.error(f"Failed to cleanup {ext.name}: {e}")

        self._initialized = False
        logger.info("AnalyzerOrchestrator shut down")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Dict with processing statistics.
        """
        avg_time_ms = (
            (self._total_time_ns / self._frames_processed / 1_000_000)
            if self._frames_processed > 0
            else 0
        )
        return {
            "frames_processed": self._frames_processed,
            "total_observations": self._total_observations,
            "timeouts": self._timeouts,
            "errors": self._errors,
            "avg_time_ms": avg_time_ms,
            "analyzers": [ext.name for ext in self._analyzers],
            "max_workers": self._max_workers,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self._initialized

    @property
    def analyzer_names(self) -> List[str]:
        """Get names of all analyzers."""
        return [ext.name for ext in self._analyzers]

    def __enter__(self) -> "AnalyzerOrchestrator":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
