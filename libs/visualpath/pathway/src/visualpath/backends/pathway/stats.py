"""Statistics collection for PathwayBackend.

PathwayStats provides thread-safe counters and timing metrics for
monitoring Pathway pipeline execution. No Pathway dependency required.

Example:
    >>> stats = PathwayStats()
    >>> stats.record_ingestion()
    >>> stats.record_analysis("face", 12.5, success=True)
    >>> print(stats.throughput_fps)
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathwayStats:
    """Thread-safe statistics for PathwayBackend pipeline execution.

    All public mutation methods acquire an internal lock, making this
    class safe for concurrent use from Pathway's connector thread,
    UDF threads, and subscribe callbacks.
    """

    # Frame counters
    frames_ingested: int = 0
    frames_analyzed: int = 0

    # Analysis counters
    analyses_completed: int = 0
    analyses_failed: int = 0

    # Trigger / observation output counters
    triggers_fired: int = 0
    observations_output: int = 0

    # Timing (milliseconds)
    total_analysis_ms: float = 0.0

    # Per-analyzer EMA times (name -> ema_ms)
    per_analyzer_time_ms: Dict[str, float] = field(default_factory=dict)

    # Raw analysis times for percentile calculation
    _analysis_times: List[float] = field(default_factory=list, repr=False)

    # Per-analyzer raw times for EMA
    _per_analyzer_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int), repr=False,
    )

    # Pipeline wall-clock (nanoseconds, monotonic)
    pipeline_start_ns: int = 0
    pipeline_end_ns: int = 0

    # Internal lock
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # EMA smoothing factor
    _ema_alpha: float = field(default=0.3, repr=False)

    # --- Mutation methods ------------------------------------------------

    def record_ingestion(self) -> None:
        """Record a frame ingested via ConnectorSubject."""
        with self._lock:
            self.frames_ingested += 1

    def record_analysis(
        self,
        analyzer_name: str,
        elapsed_ms: float,
        success: bool = True,
    ) -> None:
        """Record a single analyzer invocation.

        Args:
            analyzer_name: Name of the analyzer.
            elapsed_ms: Wall-clock time in milliseconds.
            success: Whether the analysis succeeded.
        """
        with self._lock:
            if success:
                self.analyses_completed += 1
            else:
                self.analyses_failed += 1

            self.total_analysis_ms += elapsed_ms
            self._analysis_times.append(elapsed_ms)

            # EMA per analyzer
            self._per_analyzer_counts[analyzer_name] += 1
            prev = self.per_analyzer_time_ms.get(analyzer_name)
            if prev is None:
                self.per_analyzer_time_ms[analyzer_name] = elapsed_ms
            else:
                alpha = self._ema_alpha
                self.per_analyzer_time_ms[analyzer_name] = (
                    alpha * elapsed_ms + (1 - alpha) * prev
                )

    def record_frame_analyzed(self) -> None:
        """Record that all analyzers finished for one frame."""
        with self._lock:
            self.frames_analyzed += 1

    def record_trigger(self) -> None:
        """Record a trigger fired by fusion."""
        with self._lock:
            self.triggers_fired += 1

    def record_observation_output(self) -> None:
        """Record an observation received in subscribe callback."""
        with self._lock:
            self.observations_output += 1

    def mark_pipeline_start(self) -> None:
        """Record pipeline start time."""
        with self._lock:
            self.pipeline_start_ns = time.perf_counter_ns()

    def mark_pipeline_end(self) -> None:
        """Record pipeline end time."""
        with self._lock:
            self.pipeline_end_ns = time.perf_counter_ns()

    def reset(self) -> None:
        """Reset all counters and timings."""
        with self._lock:
            self.frames_ingested = 0
            self.frames_analyzed = 0
            self.analyses_completed = 0
            self.analyses_failed = 0
            self.triggers_fired = 0
            self.observations_output = 0
            self.total_analysis_ms = 0.0
            self.per_analyzer_time_ms.clear()
            self._analysis_times.clear()
            self._per_analyzer_counts.clear()
            self.pipeline_start_ns = 0
            self.pipeline_end_ns = 0

    # --- Computed properties ---------------------------------------------

    def _pipeline_duration_sec_unlocked(self) -> float:
        """Pipeline duration without acquiring lock. Caller must hold lock."""
        start = self.pipeline_start_ns
        end = self.pipeline_end_ns
        if start == 0 or end == 0:
            return 0.0
        return (end - start) / 1_000_000_000

    def _avg_analysis_ms_unlocked(self) -> float:
        """Average analysis time without lock. Caller must hold lock."""
        n = len(self._analysis_times)
        if n == 0:
            return 0.0
        return self.total_analysis_ms / n

    def _p95_analysis_ms_unlocked(self) -> float:
        """P95 analysis time without lock. Caller must hold lock."""
        times = sorted(self._analysis_times)
        if not times:
            return 0.0
        idx = int(len(times) * 0.95)
        idx = min(idx, len(times) - 1)
        return times[idx]

    def _throughput_fps_unlocked(self) -> float:
        """Throughput FPS without lock. Caller must hold lock."""
        duration = self._pipeline_duration_sec_unlocked()
        if duration <= 0:
            return 0.0
        return self.frames_analyzed / duration

    @property
    def throughput_fps(self) -> float:
        """Frames analyzed per second based on pipeline wall-clock."""
        with self._lock:
            return self._throughput_fps_unlocked()

    @property
    def avg_analysis_ms(self) -> float:
        """Average analysis time across all invocations."""
        with self._lock:
            return self._avg_analysis_ms_unlocked()

    @property
    def p95_analysis_ms(self) -> float:
        """95th-percentile analysis time."""
        with self._lock:
            return self._p95_analysis_ms_unlocked()

    @property
    def pipeline_duration_sec(self) -> float:
        """Pipeline wall-clock duration in seconds."""
        with self._lock:
            return self._pipeline_duration_sec_unlocked()

    # --- Serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        """Snapshot of all stats as a plain dict.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        with self._lock:
            return {
                "frames_ingested": self.frames_ingested,
                "frames_analyzed": self.frames_analyzed,
                "analyses_completed": self.analyses_completed,
                "analyses_failed": self.analyses_failed,
                "triggers_fired": self.triggers_fired,
                "observations_output": self.observations_output,
                "total_analysis_ms": self.total_analysis_ms,
                "per_analyzer_time_ms": dict(self.per_analyzer_time_ms),
                "pipeline_start_ns": self.pipeline_start_ns,
                "pipeline_end_ns": self.pipeline_end_ns,
                "throughput_fps": self._throughput_fps_unlocked(),
                "avg_analysis_ms": self._avg_analysis_ms_unlocked(),
                "p95_analysis_ms": self._p95_analysis_ms_unlocked(),
                "pipeline_duration_sec": self._pipeline_duration_sec_unlocked(),
            }


__all__ = ["PathwayStats"]
