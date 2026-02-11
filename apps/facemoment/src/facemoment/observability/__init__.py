"""Observability system for FaceMoment.

This module re-exports the core observability components from visualpath
and adds FaceMoment-specific trace record types.

Provides tracing and logging infrastructure to track:
- Frame-by-frame analysis results
- Gate state transitions
- Trigger decision processes
- Component timing and performance
- Stream synchronization issues

Trace Levels:
- OFF: No tracing (production default)
- MINIMAL: Trigger events only (<1% overhead)
- NORMAL: Frame summaries + gate changes (~5% overhead)
- VERBOSE: Full signal details + timing (~15% overhead)

Example:
    >>> from facemoment.observability import ObservabilityHub, TraceLevel
    >>> hub = ObservabilityHub.get_instance()
    >>> hub.configure(level=TraceLevel.NORMAL)
    >>> hub.add_sink(FileSink("/tmp/trace.jsonl"))
    >>>
    >>> # In analyzer code:
    >>> if hub.enabled:
    ...     hub.emit(FrameAnalyzeRecord(...))
"""

# Re-export core observability from visualpath
from visualpath.observability import (
    TraceLevel,
    Sink,
    ObservabilityHub,
    TraceRecord,
)

# Re-export sinks from facemoment's extended versions
from facemoment.observability.sinks import (
    FileSink,
    ConsoleSink,
    MemorySink,
    NullSink,
)

# Re-export PipelineMonitor
from facemoment.observability.pipeline_monitor import PipelineMonitor

__all__ = [
    # Core (from visualpath)
    "TraceLevel",
    "Sink",
    "ObservabilityHub",
    # Records (TraceRecord from visualpath)
    "TraceRecord",
    # Sinks (from facemoment.observability.sinks)
    "FileSink",
    "ConsoleSink",
    "MemorySink",
    "NullSink",
    # Monitoring
    "PipelineMonitor",
]
