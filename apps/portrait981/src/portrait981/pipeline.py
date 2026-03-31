"""Portrait981 pipeline — core orchestrator."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import momentscan as ms
from personmemory.ingest import lookup_frames
from reportrait.generator import PortraitGenerator
from reportrait.types import GenerationConfig, GenerationRequest

from portrait981.node_pool import NodePool
from portrait981.types import (
    JobHandle,
    JobResult,
    JobSpec,
    JobStatus,
    PipelineConfig,
    StepCallback,
    StepEvent,
    StepTiming,
)

logger = logging.getLogger(__name__)


class Portrait981Pipeline:
    """Orchestrates momentscan -> personmemory -> reportrait pipeline.

    Execution strategy:
    - scan: always runs in the calling thread (GPU-bound, needs signal handlers)
    - generate: can run in a thread pool (I/O-bound, ComfyUI has its own queue)

    ``run_one()`` runs everything in the calling thread.
    ``run_batch()`` runs scans sequentially in the calling thread,
    then dispatches generate steps to a thread pool for parallelism.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_step: Optional[StepCallback] = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._on_step = on_step
        self._member_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._node_pool = NodePool(self._config.comfy_urls)
        self._gen_executor: Optional[ThreadPoolExecutor] = None
        self._shutdown = False
        self._interrupted = False

    def _get_gen_executor(self) -> ThreadPoolExecutor:
        """Lazy-init thread pool for generate steps."""
        if self._gen_executor is None:
            self._gen_executor = ThreadPoolExecutor(
                max_workers=self._config.max_generate_workers,
                thread_name_prefix="p981-gen",
            )
        return self._gen_executor

    def interrupt(self) -> None:
        """Signal the pipeline to stop processing remaining jobs."""
        self._interrupted = True

    def run_one(self, job: JobSpec) -> JobResult:
        """Execute a single job synchronously in the calling thread."""
        if self._shutdown:
            raise RuntimeError("Pipeline has been shut down")
        handle = JobHandle(job)
        self._set_batch_context(handle, 0, 1)
        self._execute_job(handle)
        return handle.result()

    def run_batch(self, jobs: List[JobSpec]) -> List[JobResult]:
        """Execute multiple jobs.

        Scans run sequentially in the calling thread (GPU-bound).
        Generate steps are dispatched to a thread pool for I/O parallelism.
        Ctrl+C during scan interrupts immediately; remaining jobs are skipped.
        """
        if self._shutdown:
            raise RuntimeError("Pipeline has been shut down")

        total = len(jobs)
        handles = [JobHandle(job) for job in jobs]
        for i, handle in enumerate(handles):
            self._set_batch_context(handle, i, total)

        gen_futures = []

        # Phase 1: scan + lookup sequentially in main thread
        for handle in handles:
            if self._interrupted:
                handle._set_result(JobResult(
                    job=handle.job,
                    status=JobStatus.FAILED,
                    error="interrupted",
                    timing=StepTiming(),
                ))
                continue
            self._execute_scan_and_lookup(handle)

        # Phase 2: dispatch generate steps to thread pool
        for handle in handles:
            if handle.status == JobStatus.FAILED:
                continue
            if handle.job.scan_only:
                continue

            state = handle._pending_generate
            if state is None:
                continue

            future = self._get_gen_executor().submit(
                self._execute_generate, handle, state,
            )
            gen_futures.append((handle, future))

        # Wait for all generate steps
        for handle, future in gen_futures:
            future.result()

        # Finalize handles that didn't need generate
        for handle in handles:
            if not handle.done:
                state = getattr(handle, "_pending_generate", None)
                timing = handle._timing
                timing.total_sec = time.monotonic() - handle._total_start
                handle._set_result(JobResult(
                    job=handle.job,
                    status=JobStatus.DONE,
                    scan_result=handle._scan_result,
                    ref_count=getattr(state, "ref_count", 0) if state else 0,
                    timing=timing,
                ))

        return [h.result() for h in handles]

    def submit(self, job: JobSpec) -> JobHandle:
        """Submit a job for async execution. Returns a handle.

        Note: scan runs in a background thread, which means momentscan's
        SIGINT handler will be silently skipped. Use run_one() or run_batch()
        from the main thread when SIGINT handling is needed.
        """
        if self._shutdown:
            raise RuntimeError("Pipeline has been shut down")
        handle = JobHandle(job)
        self._set_batch_context(handle, 0, 1)
        self._get_gen_executor().submit(self._execute_job, handle)
        return handle

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the pipeline executor."""
        self._shutdown = True
        if self._gen_executor is not None:
            self._gen_executor.shutdown(wait=wait)

    # -- Batch context --

    def _set_batch_context(self, handle: JobHandle, index: int, total: int) -> None:
        handle._batch_index = index
        handle._batch_total = total
        video_path = handle.job.video_path
        handle._video_name = Path(video_path).stem if video_path else handle.job.member_id

    # -- Event emission --

    def _emit(self, handle: JobHandle, step: str, status: str,
              detail: str = "", elapsed_sec: float = 0.0,
              frame_id: int = 0) -> None:
        if self._on_step is not None:
            event = StepEvent(
                job_id=handle.job_id,
                member_id=handle.job.member_id,
                step=step,
                status=status,
                detail=detail,
                elapsed_sec=elapsed_sec,
                job_index=getattr(handle, "_batch_index", 0),
                job_total=getattr(handle, "_batch_total", 1),
                video_name=getattr(handle, "_video_name", ""),
                frame_id=frame_id,
            )
            try:
                self._on_step(event)
            except Exception:
                logger.debug("on_step callback error", exc_info=True)

    # -- Core execution --

    def _execute_job(self, handle: JobHandle) -> None:
        """Full job execution: scan -> lookup -> generate (single thread)."""
        job = handle.job
        timing = StepTiming()
        total_start = time.monotonic()

        scan_result = None
        generation_result = None
        ref_count = 0
        gen_error: Optional[str] = None

        try:
            # Step 1: SCAN
            if not job.generate_only:
                scan_result = self._do_scan(handle, timing)
            else:
                self._emit(handle, "scan", "skipped")

            # Step 2: LOOKUP + Step 3: GENERATE
            if not job.scan_only:
                ref_paths, ref_count, lookup_elapsed = self._do_lookup(handle)
                timing.lookup_sec = lookup_elapsed

                if ref_paths:
                    generation_result, gen_error, gen_elapsed = self._do_generate(
                        handle, ref_paths,
                    )
                    timing.generate_sec = gen_elapsed
                else:
                    self._emit(handle, "generate", "skipped",
                               detail="no refs available")
            else:
                self._emit(handle, "lookup", "skipped")
                self._emit(handle, "generate", "skipped")

            timing.total_sec = time.monotonic() - total_start
            handle._set_result(JobResult(
                job=job,
                status=JobStatus.PARTIAL if gen_error else JobStatus.DONE,
                scan_result=scan_result,
                generation_result=generation_result,
                ref_count=ref_count,
                error=gen_error,
                timing=timing,
            ))

        except Exception as e:
            timing.total_sec = time.monotonic() - total_start
            current_step = {
                JobStatus.SCANNING: "scan",
                JobStatus.GENERATING: "generate",
            }.get(handle.status, "scan")
            self._emit(handle, current_step, "failed", detail=str(e))

            handle._set_result(JobResult(
                job=job,
                status=JobStatus.FAILED,
                scan_result=scan_result,
                error=str(e),
                timing=timing,
            ))

    def _execute_scan_and_lookup(self, handle: JobHandle) -> None:
        """Run scan + lookup in calling thread (for batch phase 1)."""
        job = handle.job
        handle._timing = StepTiming()
        handle._total_start = time.monotonic()
        handle._scan_result = None
        handle._pending_generate = None

        try:
            # SCAN
            if not job.generate_only:
                handle._scan_result = self._do_scan(handle, handle._timing)
                # Check interrupt after scan completes
                if self._interrupted:
                    handle._timing.total_sec = time.monotonic() - handle._total_start
                    handle._set_result(JobResult(
                        job=job,
                        status=JobStatus.DONE,
                        scan_result=handle._scan_result,
                        timing=handle._timing,
                    ))
                    return
            else:
                self._emit(handle, "scan", "skipped")

            # LOOKUP
            if not job.scan_only:
                ref_paths, ref_count, lookup_elapsed = self._do_lookup(handle)
                handle._timing.lookup_sec = lookup_elapsed

                if ref_paths:
                    handle._pending_generate = _PendingGenerate(
                        ref_paths=ref_paths, ref_count=ref_count,
                    )
                else:
                    self._emit(handle, "generate", "skipped",
                               detail="no refs available")
                    handle._pending_generate = _PendingGenerate(
                        ref_paths=[], ref_count=0,
                    )
            else:
                self._emit(handle, "lookup", "skipped")
                self._emit(handle, "generate", "skipped")

        except KeyboardInterrupt:
            self._interrupted = True
            handle._timing.total_sec = time.monotonic() - handle._total_start
            # Scan data may have been partially saved — preserve it
            handle._set_result(JobResult(
                job=job,
                status=JobStatus.DONE if handle._scan_result else JobStatus.FAILED,
                scan_result=handle._scan_result,
                error="interrupted by user",
                timing=handle._timing,
            ))

        except Exception as e:
            handle._timing.total_sec = time.monotonic() - handle._total_start
            current_step = {
                JobStatus.SCANNING: "scan",
                JobStatus.GENERATING: "generate",
            }.get(handle.status, "scan")
            self._emit(handle, current_step, "failed", detail=str(e))
            handle._set_result(JobResult(
                job=job,
                status=JobStatus.FAILED,
                scan_result=handle._scan_result,
                error=str(e),
                timing=handle._timing,
            ))

    def _execute_generate(self, handle: JobHandle, state: _PendingGenerate) -> None:
        """Run generate step in thread pool (for batch phase 2)."""
        job = handle.job
        timing = handle._timing

        if not state.ref_paths:
            timing.total_sec = time.monotonic() - handle._total_start
            handle._set_result(JobResult(
                job=job, status=JobStatus.DONE,
                scan_result=handle._scan_result,
                ref_count=state.ref_count, timing=timing,
            ))
            return

        try:
            generation_result, gen_error, gen_elapsed = self._do_generate(
                handle, state.ref_paths,
            )
            timing.generate_sec = gen_elapsed
            timing.total_sec = time.monotonic() - handle._total_start

            handle._set_result(JobResult(
                job=job,
                status=JobStatus.PARTIAL if gen_error else JobStatus.DONE,
                scan_result=handle._scan_result,
                generation_result=generation_result,
                ref_count=state.ref_count,
                error=gen_error,
                timing=timing,
            ))
        except Exception as e:
            timing.total_sec = time.monotonic() - handle._total_start
            self._emit(handle, "generate", "failed", detail=str(e))
            handle._set_result(JobResult(
                job=job, status=JobStatus.PARTIAL,
                scan_result=handle._scan_result,
                ref_count=state.ref_count,
                error=str(e), timing=timing,
            ))

    # -- Step primitives --

    def _do_scan(self, handle: JobHandle, timing: StepTiming) -> object:
        """Run momentscan. Returns scan_result."""
        handle.status = JobStatus.SCANNING
        self._emit(handle, "scan", "started")
        scan_start = time.monotonic()

        scan_output_dir = handle.job.output_dir
        if scan_output_dir is None:
            scan_output_dir = tempfile.mkdtemp(prefix="p981_scan_")

        # Per-frame progress callback → emits "progress" events
        frame_counter = [0]

        def _on_frame(frame, results):
            frame_counter[0] += 1
            elapsed = time.monotonic() - scan_start
            self._emit(
                handle, "scan", "progress",
                detail=f"frame {frame.frame_id}",
                elapsed_sec=elapsed,
                frame_id=frame.frame_id,
            )
            # Propagate interrupt into momentscan
            return not self._interrupted

        with self._member_locks[handle.job.member_id]:
            scan_result = ms.run(
                handle.job.video_path,
                member_id=handle.job.member_id or None,
                output_dir=scan_output_dir,
                collection_path=handle.job.collection_path,
                fps=self._config.scan_fps,
                backend=self._config.scan_backend,
                on_frame=_on_frame,
            )

        timing.scan_sec = time.monotonic() - scan_start
        handle.status = JobStatus.INGESTED

        highlights = getattr(scan_result, "highlights", [])
        frame_count = getattr(scan_result, "frame_count", 0)
        self._emit(
            handle, "scan", "completed",
            detail=f"{frame_count} frames, {len(highlights)} highlights",
            elapsed_sec=timing.scan_sec,
        )
        return scan_result

    def _do_lookup(self, handle: JobHandle) -> tuple:
        """Run lookup_frames. Returns (ref_paths, ref_count, elapsed_sec)."""
        self._emit(handle, "lookup", "started")
        lookup_start = time.monotonic()
        frames = lookup_frames(
            handle.job.member_id,
            pose=handle.job.pose,
            category=handle.job.category,
            top_k=handle.job.top_k,
        )
        ref_paths = [f["path"] for f in frames]
        ref_count = len(ref_paths)
        elapsed = time.monotonic() - lookup_start

        self._emit(
            handle, "lookup", "completed",
            detail=f"{ref_count} refs found",
            elapsed_sec=elapsed,
        )
        return ref_paths, ref_count, elapsed

    def _do_generate(self, handle: JobHandle, ref_paths: list) -> tuple:
        """Run PortraitGenerator. Returns (result, error_str, elapsed_sec).

        Acquires a ComfyUI node from the pool; on failure marks it unhealthy
        so subsequent requests route to a healthy node.
        """
        job = handle.job
        ref_count = len(ref_paths)
        handle.status = JobStatus.GENERATING
        self._emit(handle, "generate", "started",
                   detail=f"{ref_count} refs, workflow={job.workflow}")
        gen_start = time.monotonic()

        comfy_url = self._node_pool.acquire()

        gen_config = GenerationConfig(
            comfy_url=comfy_url,
            api_key=self._config.api_key,
            workflow_template=job.workflow,
        )
        request = GenerationRequest(
            person_id=0,
            ref_paths=ref_paths,
            style_prompt=job.prompt,
            workflow_template=job.workflow,
        )

        gen_error: Optional[str] = None
        generation_result = None
        try:
            generation_result = PortraitGenerator(gen_config).generate(request)
            self._node_pool.release(comfy_url)
        except Exception as e:
            gen_error = str(e)
            self._node_pool.mark_unhealthy(comfy_url)
            self._node_pool.release(comfy_url)

        elapsed = time.monotonic() - gen_start

        if gen_error:
            self._emit(handle, "generate", "failed",
                       detail=gen_error, elapsed_sec=elapsed)
        elif generation_result and not getattr(generation_result, "success", True):
            gen_error = getattr(generation_result, "error", None) or "generation returned success=False"
            self._emit(handle, "generate", "failed",
                       detail=gen_error, elapsed_sec=elapsed)
        else:
            out_paths = getattr(generation_result, "output_paths", [])
            self._emit(handle, "generate", "completed",
                       detail=f"{len(out_paths)} outputs",
                       elapsed_sec=elapsed)

        return generation_result, gen_error, elapsed


class _PendingGenerate:
    """Intermediate state between scan/lookup and generate phases."""
    __slots__ = ("ref_paths", "ref_count")

    def __init__(self, ref_paths: list, ref_count: int) -> None:
        self.ref_paths = ref_paths
        self.ref_count = ref_count
