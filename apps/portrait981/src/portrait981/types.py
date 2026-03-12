"""Data types for portrait981 pipeline."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class JobSpec:
    """Input specification for a pipeline job."""

    video_path: str = ""
    member_id: str = ""
    pose: Optional[str] = None
    category: Optional[str] = None
    prompt: str = ""
    workflow: str = "default"
    collection_path: Optional[str] = None
    output_dir: Optional[str] = None
    top_k: int = 3
    scan_only: bool = False
    generate_only: bool = False


class JobStatus(str, Enum):
    """Pipeline job status."""

    PENDING = "pending"
    SCANNING = "scanning"
    INGESTED = "ingested"
    GENERATING = "generating"
    DONE = "done"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class StepTiming:
    """Timing information for pipeline steps."""

    scan_sec: float = 0.0
    lookup_sec: float = 0.0
    generate_sec: float = 0.0
    total_sec: float = 0.0


@dataclass
class StepEvent:
    """Progress event emitted at each pipeline step transition."""

    job_id: str
    member_id: str
    step: str       # "scan", "lookup", "generate"
    status: str     # "started", "completed", "failed", "skipped", "progress"
    detail: str = ""
    elapsed_sec: float = 0.0
    job_index: int = 0      # 0-based index within batch
    job_total: int = 1      # total jobs in batch
    video_name: str = ""    # video file stem for display
    frame_id: int = 0       # current frame (for progress updates)


StepCallback = Callable[[StepEvent], None]


@dataclass
class JobResult:
    """Output of a pipeline job."""

    job: JobSpec
    status: JobStatus
    scan_result: Optional[Any] = None
    generation_result: Optional[Any] = None
    ref_count: int = 0
    error: Optional[str] = None
    timing: StepTiming = field(default_factory=StepTiming)


@dataclass
class PipelineConfig:
    """Global pipeline configuration."""

    max_scan_workers: int = 1
    max_generate_workers: int = 2
    comfy_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8188"])
    api_key: Optional[str] = None
    default_collection_path: Optional[str] = None
    default_workflow: str = "default"
    scan_fps: int = 10
    scan_backend: str = "simple"


class JobHandle:
    """Async handle for a submitted pipeline job."""

    def __init__(self, job: JobSpec) -> None:
        self.job_id: str = uuid.uuid4().hex[:12]
        self.job: JobSpec = job
        self._status: JobStatus = JobStatus.PENDING
        self._result: Optional[JobResult] = None
        self._event = threading.Event()

    @property
    def status(self) -> JobStatus:
        return self._status

    @status.setter
    def status(self, value: JobStatus) -> None:
        self._status = value

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: Optional[float] = None) -> JobResult:
        """Block until the job completes and return the result."""
        self._event.wait(timeout=timeout)
        if self._result is None:
            raise TimeoutError(f"Job {self.job_id} did not complete in time")
        return self._result

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for job completion. Returns True if completed."""
        return self._event.wait(timeout=timeout)

    def _set_result(self, result: JobResult) -> None:
        self._result = result
        self._status = result.status
        self._event.set()
