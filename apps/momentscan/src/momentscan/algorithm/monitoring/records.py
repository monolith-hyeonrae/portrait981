"""Pipeline monitoring and analysis trace records.

These records are emitted by PipelineMonitor during real-time
pipeline execution for performance tracking and diagnostics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from visualpath.observability import TraceLevel
from visualpath.observability.records import TraceRecord


# =============================================================================
# Analysis Records
# =============================================================================


@dataclass
class FrameAnalyzeRecord(TraceRecord):
    """Record of frame analysis results.

    Emitted by analyzers after processing each frame.
    Contains summary of what was analyzed.
    """
    record_type: str = field(default="frame_analyze", init=False)

    frame_id: int = 0
    t_ns: int = 0
    source: str = ""  # "face", "pose", "gesture"

    # Summary (NORMAL level)
    face_count: int = 0
    pose_count: int = 0
    gesture_detected: bool = False

    # Detailed signals (VERBOSE level)
    signals: Dict[str, float] = field(default_factory=dict)

    # Processing time
    processing_ms: float = 0.0


@dataclass
class FaceAnalyzeDetail(TraceRecord):
    """Detailed face analysis record for VERBOSE level.

    Contains per-face information including pose angles,
    expression values, and tracking data.
    """
    record_type: str = field(default="face_analyze_detail", init=False)
    min_level: TraceLevel = field(default=TraceLevel.VERBOSE, repr=False)

    frame_id: int = 0
    face_id: int = 0

    # Detection
    confidence: float = 0.0
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h normalized

    # Pose
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Expression
    expression: float = 0.0
    dominant_emotion: str = ""

    # Tracking
    inside_frame: bool = True
    area_ratio: float = 0.0
    center_distance: float = 0.0


# =============================================================================
# Pathway Pipeline Monitoring Records
# =============================================================================


@dataclass
class PathwayFrameRecord(TraceRecord):
    """Per-frame pipeline summary for Pathway backend.

    Emitted at NORMAL level for each frame processed through the
    Pathway pipeline, summarizing analyzer timings and fusion result.
    """
    record_type: str = field(default="pathway_frame", init=False)

    frame_id: int = 0
    t_ns: int = 0
    analyzer_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_frame_ms: float = 0.0
    observations_produced: List[str] = field(default_factory=list)
    observations_failed: List[str] = field(default_factory=list)
    fusion_ms: float = 0.0
    fusion_decision: str = ""  # "triggered"/"no_trigger"/"cooldown"/"gate_closed"


@dataclass
class AnalyzerTimingRecord(TraceRecord):
    """Individual analyzer timing with sub-component breakdown.

    Emitted at NORMAL level for each analyzer on each frame.
    """
    record_type: str = field(default="analyzer_timing", init=False)

    frame_id: int = 0
    analyzer_name: str = ""
    processing_ms: float = 0.0
    produced_observation: bool = False
    sub_timings_ms: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationMergeRecord(TraceRecord):
    """Detailed observation merge information.

    Emitted at VERBOSE level when observations from multiple analyzers
    are merged for fusion input.
    """
    record_type: str = field(default="observation_merge", init=False)
    min_level: TraceLevel = field(default=TraceLevel.VERBOSE, repr=False)

    frame_id: int = 0
    input_sources: List[str] = field(default_factory=list)
    input_signal_counts: Dict[str, int] = field(default_factory=dict)
    merged_signal_keys: List[str] = field(default_factory=list)
    main_face_id: Optional[int] = None
    main_face_source: str = ""  # "classifier_obs"/"merged_signals"/"none"


@dataclass
class BackpressureRecord(TraceRecord):
    """Rolling performance statistics emitted periodically.

    Emitted at NORMAL level every N frames (default 50) with
    aggregated timing statistics for bottleneck analysis.
    """
    record_type: str = field(default="backpressure", init=False)

    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0
    wall_time_sec: float = 0.0
    effective_fps: float = 0.0
    target_fps: float = 0.0
    fps_ratio: float = 0.0
    analyzer_avg_ms: Dict[str, float] = field(default_factory=dict)
    analyzer_max_ms: Dict[str, float] = field(default_factory=dict)
    analyzer_p95_ms: Dict[str, float] = field(default_factory=dict)
    slowest_analyzer: str = ""
    bottleneck_pct: float = 0.0
    fusion_avg_ms: float = 0.0


@dataclass
class PipelineStatsRecord(TraceRecord):
    """Session-level pipeline summary emitted at session end.

    Emitted at MINIMAL level with overall statistics.
    """
    record_type: str = field(default="pipeline_stats", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    total_frames: int = 0
    total_triggers: int = 0
    wall_time_sec: float = 0.0
    effective_fps: float = 0.0
    analyzer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fusion_avg_ms: float = 0.0
    gate_open_pct: float = 0.0


__all__ = [
    "FrameAnalyzeRecord",
    "FaceAnalyzeDetail",
    "PathwayFrameRecord",
    "AnalyzerTimingRecord",
    "ObservationMergeRecord",
    "BackpressureRecord",
    "PipelineStatsRecord",
]
