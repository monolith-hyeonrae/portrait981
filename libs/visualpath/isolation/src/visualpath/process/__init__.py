"""Process wrappers for distributed execution.

This module provides wrappers for running analyzers and fusion modules
as independent processes, with support for different isolation levels.

Components:
- Workers: Different isolation levels (inline, thread, process, venv)
- Mappers: Observation ↔ Message serialization
- IPC Processes: A-B*-C architecture (AnalyzerProcess, FusionProcess)
- Orchestrator: Thread-parallel analyzer execution
"""

from visualpath.process.launcher import (
    WorkerLauncher,
    BaseWorker,
    WorkerResult,
    InlineWorker,
    ThreadWorker,
    ProcessWorker,
    VenvWorker,
)
from visualpath.process.worker_module import WorkerModule
from visualpath.process.mapper import (
    ObservationMapper,
    DefaultObservationMapper,
    CompositeMapper,
)
from visualpath.process.ipc import (
    AnalyzerProcess,
    FusionProcess,
    ALIGNMENT_WINDOW_NS,
)
from visualpath.process.orchestrator import AnalyzerOrchestrator

__all__ = [
    # Workers
    "WorkerLauncher",
    "BaseWorker",
    "WorkerResult",
    "InlineWorker",
    "ThreadWorker",
    "ProcessWorker",
    "VenvWorker",
    # WorkerModule
    "WorkerModule",
    # Mappers
    "ObservationMapper",
    "DefaultObservationMapper",
    "CompositeMapper",
    # IPC Processes
    "AnalyzerProcess",
    "FusionProcess",
    "ALIGNMENT_WINDOW_NS",
    # Orchestrator
    "AnalyzerOrchestrator",
]
