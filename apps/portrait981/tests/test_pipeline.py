"""Pipeline tests for portrait981."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from portrait981.pipeline import Portrait981Pipeline
from portrait981.types import JobResult, JobSpec, JobStatus, PipelineConfig, StepEvent


@dataclass
class _FakeScanResult:
    highlights: List[Any] = field(default_factory=lambda: [{"window_id": 0}])
    frame_count: int = 100
    duration_sec: float = 10.0
    actual_backend: str = "simple"
    stats: Dict[str, Any] = field(default_factory=dict)


class TestRunOneFull:
    def test_full_pipeline(self, mock_ms_run, mock_lookup, mock_generator):
        """scan -> lookup -> generate 전체 흐름."""
        job = JobSpec(video_path="v.mp4", member_id="test_1")
        pipeline = Portrait981Pipeline()
        try:
            result = pipeline.run_one(job)
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.DONE
        mock_ms_run.run.assert_called_once()
        mock_lookup.assert_called_once()
        mock_generator.generate.assert_called_once()
        assert result.scan_result is not None
        assert result.generation_result is not None
        assert result.ref_count == 2
        assert result.error is None
        assert result.timing.total_sec > 0

    def test_scan_only(self, mock_ms_run, mock_lookup, mock_generator):
        """scan_only=True -> generate 미호출."""
        job = JobSpec(video_path="v.mp4", member_id="test_1", scan_only=True)
        pipeline = Portrait981Pipeline()
        try:
            result = pipeline.run_one(job)
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.DONE
        mock_ms_run.run.assert_called_once()
        mock_lookup.assert_not_called()
        mock_generator.generate.assert_not_called()
        assert result.scan_result is not None
        assert result.generation_result is None

    def test_generate_only(self, mock_ms_run, mock_lookup, mock_generator):
        """generate_only=True -> ms.run 미호출."""
        job = JobSpec(member_id="test_1", generate_only=True)
        pipeline = Portrait981Pipeline()
        try:
            result = pipeline.run_one(job)
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.DONE
        mock_ms_run.run.assert_not_called()
        mock_lookup.assert_called_once()
        mock_generator.generate.assert_called_once()

    def test_no_refs_in_bank(self, mock_ms_run, mock_lookup_empty, mock_generator):
        """lookup 빈 결과 -> DONE, generation 건너뜀."""
        job = JobSpec(video_path="v.mp4", member_id="test_1")
        pipeline = Portrait981Pipeline()
        try:
            result = pipeline.run_one(job)
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.DONE
        assert result.ref_count == 0
        mock_generator.generate.assert_not_called()

    def test_scan_fails(self, mock_lookup, mock_generator):
        """ms.run 예외 -> FAILED."""
        with patch("portrait981.pipeline.ms") as m:
            m.run.side_effect = RuntimeError("GPU OOM")
            job = JobSpec(video_path="v.mp4", member_id="test_1")
            pipeline = Portrait981Pipeline()
            try:
                result = pipeline.run_one(job)
            finally:
                pipeline.shutdown()

        assert result.status == JobStatus.FAILED
        assert "GPU OOM" in result.error
        mock_generator.generate.assert_not_called()


class TestPartialFailure:
    def test_generation_exception_is_partial(self, mock_ms_run, mock_lookup, mock_generator_fail):
        """ComfyUI 예외 -> PARTIAL, scan_result 보존, error 기록."""
        job = JobSpec(video_path="v.mp4", member_id="test_1")
        pipeline = Portrait981Pipeline()
        try:
            result = pipeline.run_one(job)
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.PARTIAL
        assert result.scan_result is not None
        assert result.generation_result is None
        assert "ComfyUI timeout" in result.error

    def test_generation_unsuccessful_is_partial(self, mock_ms_run, mock_lookup, mock_generator_unsuccessful):
        """generate returns success=False -> PARTIAL."""
        job = JobSpec(video_path="v.mp4", member_id="test_1")
        pipeline = Portrait981Pipeline()
        try:
            result = pipeline.run_one(job)
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.PARTIAL
        assert result.scan_result is not None
        assert result.generation_result is not None
        assert "Model not found" in result.error

    def test_partial_then_retry_generate_only(self, mock_ms_run, mock_lookup):
        """PARTIAL 후 generate_only로 재시도 -> DONE."""
        # First run: generation fails
        with patch("portrait981.pipeline.PortraitGenerator") as cls:
            inst = cls.return_value
            inst.generate.side_effect = ConnectionError("ComfyUI refused")
            pipeline = Portrait981Pipeline()
            try:
                r1 = pipeline.run_one(JobSpec(video_path="v.mp4", member_id="m1"))
            finally:
                pipeline.shutdown()

        assert r1.status == JobStatus.PARTIAL

        # Second run: generate_only succeeds (mock_lookup still active)
        with patch("portrait981.pipeline.PortraitGenerator") as cls:
            inst = cls.return_value
            inst.generate.return_value = MagicMock(success=True, output_paths=["/out/p.png"])
            pipeline = Portrait981Pipeline()
            try:
                r2 = pipeline.run_one(JobSpec(member_id="m1", generate_only=True))
            finally:
                pipeline.shutdown()

        assert r2.status == JobStatus.DONE
        assert r2.generation_result is not None
        assert r2.error is None


class TestStepEvents:
    def test_full_pipeline_emits_all_steps(self, mock_ms_run, mock_lookup, mock_generator):
        """전체 파이프라인 -> scan/lookup/generate 이벤트 발행."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="test_1"))
        finally:
            pipeline.shutdown()

        steps = [(e.step, e.status) for e in events]
        assert ("scan", "started") in steps
        assert ("scan", "completed") in steps
        assert ("lookup", "started") in steps
        assert ("lookup", "completed") in steps
        assert ("generate", "started") in steps
        assert ("generate", "completed") in steps

    def test_scan_completed_has_detail(self, mock_ms_run, mock_lookup, mock_generator):
        """scan completed 이벤트에 frame/highlight 정보."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="test_1"))
        finally:
            pipeline.shutdown()

        scan_done = [e for e in events if e.step == "scan" and e.status == "completed"][0]
        assert "100 frames" in scan_done.detail
        assert "1 highlights" in scan_done.detail
        assert scan_done.elapsed_sec > 0

    def test_lookup_completed_has_ref_count(self, mock_ms_run, mock_lookup, mock_generator):
        """lookup completed 이벤트에 ref 수."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="test_1"))
        finally:
            pipeline.shutdown()

        lookup_done = [e for e in events if e.step == "lookup" and e.status == "completed"][0]
        assert "2 refs" in lookup_done.detail

    def test_scan_only_skips_generate(self, mock_ms_run, mock_lookup, mock_generator):
        """scan_only -> lookup/generate skipped."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="t1", scan_only=True))
        finally:
            pipeline.shutdown()

        steps = [(e.step, e.status) for e in events]
        assert ("scan", "completed") in steps
        assert ("lookup", "skipped") in steps
        assert ("generate", "skipped") in steps

    def test_generate_only_skips_scan(self, mock_ms_run, mock_lookup, mock_generator):
        """generate_only -> scan skipped."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(member_id="t1", generate_only=True))
        finally:
            pipeline.shutdown()

        steps = [(e.step, e.status) for e in events]
        assert ("scan", "skipped") in steps
        assert ("lookup", "completed") in steps
        assert ("generate", "completed") in steps

    def test_generate_failed_event(self, mock_ms_run, mock_lookup, mock_generator_fail):
        """generate 실패 -> failed 이벤트."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="t1"))
        finally:
            pipeline.shutdown()

        gen_fail = [e for e in events if e.step == "generate" and e.status == "failed"][0]
        assert "ComfyUI timeout" in gen_fail.detail

    def test_no_refs_skips_generate(self, mock_ms_run, mock_lookup_empty, mock_generator):
        """lookup 결과 0건 -> generate skipped."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="t1"))
        finally:
            pipeline.shutdown()

        steps = [(e.step, e.status) for e in events]
        assert ("generate", "skipped") in steps

    def test_scan_failed_event(self, mock_lookup, mock_generator):
        """scan 예외 -> scan failed 이벤트."""
        events: List[StepEvent] = []
        with patch("portrait981.pipeline.ms") as m:
            m.run.side_effect = RuntimeError("GPU OOM")
            pipeline = Portrait981Pipeline(on_step=events.append)
            try:
                pipeline.run_one(JobSpec(video_path="v.mp4", member_id="t1"))
            finally:
                pipeline.shutdown()

        scan_fail = [e for e in events if e.step == "scan" and e.status == "failed"][0]
        assert "GPU OOM" in scan_fail.detail

    def test_events_include_job_id_and_member(self, mock_ms_run, mock_lookup, mock_generator):
        """이벤트에 job_id, member_id 포함."""
        events: List[StepEvent] = []
        pipeline = Portrait981Pipeline(on_step=events.append)
        try:
            pipeline.run_one(JobSpec(video_path="v.mp4", member_id="person_7"))
        finally:
            pipeline.shutdown()

        assert all(e.member_id == "person_7" for e in events)
        assert all(len(e.job_id) == 12 for e in events)

    def test_callback_error_does_not_break_pipeline(self, mock_ms_run, mock_lookup, mock_generator):
        """콜백 예외가 파이프라인 실행을 중단시키지 않음."""
        def bad_callback(event):
            raise ValueError("callback bug")

        pipeline = Portrait981Pipeline(on_step=bad_callback)
        try:
            result = pipeline.run_one(JobSpec(video_path="v.mp4", member_id="t1"))
        finally:
            pipeline.shutdown()

        assert result.status == JobStatus.DONE


class TestRunBatch:
    def test_multiple_jobs(self, mock_ms_run, mock_lookup, mock_generator):
        """여러 job 완료."""
        jobs = [
            JobSpec(video_path="v1.mp4", member_id="m1"),
            JobSpec(video_path="v2.mp4", member_id="m2"),
        ]
        pipeline = Portrait981Pipeline()
        try:
            results = pipeline.run_batch(jobs)
        finally:
            pipeline.shutdown()

        assert len(results) == 2
        assert all(r.status == JobStatus.DONE for r in results)


class TestSubmitAsync:
    def test_handle_result(self, mock_ms_run, mock_lookup, mock_generator):
        """handle.result() blocks until done."""
        job = JobSpec(video_path="v.mp4", member_id="test_1")
        pipeline = Portrait981Pipeline()
        try:
            handle = pipeline.submit(job)
            result = handle.result(timeout=10)
            assert handle.done
            assert result.status == JobStatus.DONE
        finally:
            pipeline.shutdown()


class TestConcurrency:
    def test_member_lock_serialization(self, mock_lookup, mock_generator):
        """같은 member_id -> scan 직렬화."""
        call_order = []

        def slow_scan(*args, **kwargs):
            call_order.append(("start", kwargs.get("member_id")))
            time.sleep(0.05)
            call_order.append(("end", kwargs.get("member_id")))
            return _FakeScanResult()

        with patch("portrait981.pipeline.ms") as m:
            m.run.side_effect = slow_scan
            config = PipelineConfig(max_scan_workers=2)
            pipeline = Portrait981Pipeline(config)
            try:
                jobs = [
                    JobSpec(video_path="v1.mp4", member_id="same"),
                    JobSpec(video_path="v2.mp4", member_id="same"),
                ]
                results = pipeline.run_batch(jobs)
            finally:
                pipeline.shutdown()

        assert len(results) == 2
        starts = [i for i, (action, _) in enumerate(call_order) if action == "start"]
        ends = [i for i, (action, _) in enumerate(call_order) if action == "end"]
        assert len(starts) == 2
        assert ends[0] < starts[1], f"Expected serial execution, got: {call_order}"

    def test_batch_scans_are_sequential(self, mock_lookup, mock_generator):
        """run_batch에서 scan은 메인 스레드에서 순차 실행 (signal 호환)."""
        call_order = []

        def tracked_scan(*args, **kwargs):
            call_order.append(("start", kwargs.get("member_id")))
            time.sleep(0.02)
            call_order.append(("end", kwargs.get("member_id")))
            return _FakeScanResult()

        with patch("portrait981.pipeline.ms") as m:
            m.run.side_effect = tracked_scan
            pipeline = Portrait981Pipeline()
            try:
                jobs = [
                    JobSpec(video_path="v1.mp4", member_id="m1"),
                    JobSpec(video_path="v2.mp4", member_id="m2"),
                ]
                results = pipeline.run_batch(jobs)
            finally:
                pipeline.shutdown()

        assert len(results) == 2
        starts = [i for i, (action, _) in enumerate(call_order) if action == "start"]
        ends = [i for i, (action, _) in enumerate(call_order) if action == "end"]
        assert len(starts) == 2
        # Scans must be sequential: second starts after first ends
        assert ends[0] < starts[1], f"Expected sequential scans, got: {call_order}"

    def test_batch_generates_are_parallel(self, mock_ms_run, mock_lookup):
        """run_batch에서 generate는 스레드풀에서 병렬 실행."""
        active = {"count": 0, "max": 0}
        lock = threading.Lock()

        def tracked_generate(request):
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.05)
            with lock:
                active["count"] -= 1
            return MagicMock(success=True, output_paths=[], error=None)

        with patch("portrait981.pipeline.PortraitGenerator") as cls:
            cls.return_value.generate.side_effect = tracked_generate
            config = PipelineConfig(max_generate_workers=2)
            pipeline = Portrait981Pipeline(config)
            try:
                jobs = [
                    JobSpec(video_path="v1.mp4", member_id="m1"),
                    JobSpec(video_path="v2.mp4", member_id="m2"),
                ]
                results = pipeline.run_batch(jobs)
            finally:
                pipeline.shutdown()

        assert len(results) == 2
        assert all(r.status == JobStatus.DONE for r in results)
        assert active["max"] == 2, f"Expected parallel generates, max concurrent: {active['max']}"


class TestShutdown:
    def test_rejects_new_jobs(self, mock_ms_run, mock_lookup, mock_generator):
        """shutdown 후 submit -> RuntimeError."""
        pipeline = Portrait981Pipeline()
        pipeline.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            pipeline.submit(JobSpec(video_path="v.mp4", member_id="test"))


class TestPassedArguments:
    def test_scan_args_forwarded(self, mock_lookup, mock_generator):
        """ms.run()에 올바른 인자 전달."""
        with patch("portrait981.pipeline.ms") as m:
            m.run.return_value = _FakeScanResult()
            config = PipelineConfig(scan_fps=5, scan_backend="worker")
            job = JobSpec(
                video_path="video.mp4",
                member_id="person_1",
                collection_path="/cat/v1",
                output_dir="/out",
            )
            pipeline = Portrait981Pipeline(config)
            try:
                pipeline.run_one(job)
            finally:
                pipeline.shutdown()

            m.run.assert_called_once()
            call_kwargs = m.run.call_args
            assert call_kwargs[0] == ("video.mp4",)
            assert call_kwargs[1]["member_id"] == "person_1"
            assert call_kwargs[1]["output_dir"] == "/out"
            assert call_kwargs[1]["collection_path"] == "/cat/v1"
            assert call_kwargs[1]["fps"] == 5
            assert call_kwargs[1]["backend"] == "worker"
            assert callable(call_kwargs[1]["on_frame"])

    def test_lookup_args_forwarded(self, mock_ms_run, mock_generator):
        """lookup_frames에 올바른 인자 전달."""
        with patch("portrait981.pipeline.lookup_frames") as m:
            m.return_value = [{"path": "/f.jpg"}]
            job = JobSpec(
                video_path="v.mp4",
                member_id="p1",
                pose="left30",
                category="smile",
                top_k=5,
            )
            pipeline = Portrait981Pipeline()
            try:
                pipeline.run_one(job)
            finally:
                pipeline.shutdown()

            m.assert_called_once_with("p1", pose="left30", category="smile", top_k=5)
