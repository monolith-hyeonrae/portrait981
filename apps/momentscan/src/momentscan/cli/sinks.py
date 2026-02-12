"""Trace output sinks for observability.

This module re-exports sinks from visualpath and adds MomentScan-specific
formatting for console output.

Sinks receive trace records and handle their output to various destinations:
- FileSink: JSONL file output
- ConsoleSink: Formatted console output
- MemorySink: In-memory buffer for testing/analysis
"""

import sys
from typing import List, Optional, TextIO

# Re-export sinks from visualpath
from visualpath.observability.sinks import (
    FileSink,
    NullSink,
    ConsoleSink as BaseConsoleSink,
    MemorySink as BaseMemorySink,
)
from visualpath.observability import Sink, TraceLevel
from visualpath.observability.records import TraceRecord, TimingRecord, FrameDropRecord, SyncDelayRecord

# Import momentscan-specific records
from momentscan.algorithm.monitoring.records import (
    BackpressureRecord,
    PathwayFrameRecord,
    PipelineStatsRecord,
)


class MemorySink(BaseMemorySink):
    """Memory sink with MomentScan-specific helper methods."""
    pass


class ConsoleSink(BaseConsoleSink):
    """Console sink with MomentScan-specific formatting.

    Extends the base ConsoleSink to format MomentScan-specific
    records like BackpressureRecord and PipelineStatsRecord.
    """

    def _format_record(self, record: TraceRecord) -> Optional[str]:
        """Format a record for console output."""
        if isinstance(record, BackpressureRecord):
            return self._format_backpressure(record)
        elif isinstance(record, PathwayFrameRecord):
            return self._format_pathway_frame(record)
        elif isinstance(record, PipelineStatsRecord):
            return self._format_pipeline_stats(record)
        elif isinstance(record, TimingRecord):
            return self._format_timing(record)
        elif isinstance(record, FrameDropRecord):
            return self._format_frame_drop(record)
        elif isinstance(record, SyncDelayRecord):
            return self._format_sync_delay(record)
        else:
            return None

    def _format_backpressure(self, record: BackpressureRecord) -> str:
        """Format backpressure/rolling performance record."""
        tag = self._colorize("[PERF]", "magenta")

        if record.fps_ratio >= 0.9:
            fps_str = self._colorize(f"{record.effective_fps:.1f}fps ({record.fps_ratio:.0%})", "green")
        elif record.fps_ratio >= 0.7:
            fps_str = self._colorize(f"{record.effective_fps:.1f}fps ({record.fps_ratio:.0%})", "yellow")
        else:
            fps_str = self._colorize(f"{record.effective_fps:.1f}fps ({record.fps_ratio:.0%})", "red")

        bottleneck = ""
        if record.slowest_analyzer:
            bottleneck = f" bottleneck={record.slowest_analyzer} ({record.bottleneck_pct:.0f}%)"

        return (
            f"{tag} Frames {record.start_frame}-{record.end_frame}: "
            f"{fps_str}{bottleneck}"
        )

    def _format_pathway_frame(self, record: PathwayFrameRecord) -> Optional[str]:
        """Format pathway frame record (only slow frames >100ms)."""
        if record.total_frame_ms <= 100:
            return None

        tag = self._colorize("[FRAME]", "yellow")
        return (
            f"{tag} Frame {record.frame_id}: "
            f"{record.total_frame_ms:.0f}ms"
        )

    def _format_pipeline_stats(self, record: PipelineStatsRecord) -> str:
        """Format pipeline session summary."""
        lines = []
        sep = "=" * 50
        lines.append(sep)
        lines.append("Pipeline Summary")
        lines.append(sep)
        lines.append(f"  Duration:      {record.wall_time_sec:.1f}s ({record.total_frames} frames)")
        lines.append(f"  Effective FPS: {record.effective_fps:.1f}")

        if record.analyzer_stats:
            lines.append("")
            lines.append("  Analyzer Performance:")
            header = f"  {'Analyzer':<14} {'Avg ms':>8} {'P95 ms':>8} {'Max ms':>8} {'Errors':>8}"
            lines.append(header)
            lines.append("  " + "-" * 48)
            for name, stats in record.analyzer_stats.items():
                avg = stats.get("avg_ms", 0)
                p95 = stats.get("p95_ms", 0)
                mx = stats.get("max_ms", 0)
                errs = int(stats.get("errors", 0))
                lines.append(f"  {name:<14} {avg:>8.1f} {p95:>8.1f} {mx:>8.1f} {errs:>8}")

        lines.append(sep)

        return "\n".join(lines)


__all__ = [
    "FileSink",
    "ConsoleSink",
    "MemorySink",
    "NullSink",
]
