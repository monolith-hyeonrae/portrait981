"""Integration tests — real personmemory I/O, mocked scan/generate.

Tests the full portrait981 orchestration with:
- Momentscan.scan(): mocked (no GPU needed)
- personmemory: real file I/O (tmp_path isolation via PORTRAIT981_HOME)
- PortraitGenerator: mocked (no ComfyUI needed)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from portrait981.pipeline import Portrait981Pipeline
from portrait981.types import JobSpec, JobStatus, PipelineConfig, StepEvent

from conftest import FakeFrameResult


@pytest.fixture
def bank_home(tmp_path, monkeypatch):
    """Redirect PORTRAIT981_HOME to tmp_path for test isolation."""
    monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path))
    return tmp_path


def _seed_bank(bank_home: Path, member_id: str, entries: List[dict]) -> Path:
    """Directly write a frames manifest to simulate ingested frames."""
    from personmemory.paths import get_bank_dir

    frames_dir = get_bank_dir(member_id) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy image files
    for entry in entries:
        (frames_dir / entry["file"]).write_bytes(b"\xff\xd8dummy-jpg")

    # Write manifest
    manifest_path = frames_dir / "frames.json"
    with open(manifest_path, "w") as f:
        json.dump(entries, f)

    return frames_dir


# ── Test: scan → bank seed → lookup → generate ──


class TestScanThenGenerate:
    """Full pipeline with real personmemory I/O."""

    def test_scan_seeds_bank_then_generate_uses_it(self, bank_home):
        """scan saves to bank → pipeline lookup finds refs → generate called."""
        member_id = "integ_test_1"

        # Prepare: seed the bank as if scan+ingest had completed
        entries = [
            {
                "file": "clip_frontal_smile_10.jpg",
                "frame_idx": 10,
                "timestamp_ms": 1000.0,
                "pose_name": "frontal",
                "category": "smile",
                "cell_key": "frontal|smile",
                "quality": 0.85,
                "cell_score": 0.92,
                "video": "clip",
            },
            {
                "file": "clip_frontal_neutral_20.jpg",
                "frame_idx": 20,
                "timestamp_ms": 2000.0,
                "pose_name": "frontal",
                "category": "neutral",
                "cell_key": "frontal|neutral",
                "quality": 0.80,
                "cell_score": 0.88,
                "video": "clip",
            },
            {
                "file": "clip_left30_smile_30.jpg",
                "frame_idx": 30,
                "timestamp_ms": 3000.0,
                "pose_name": "left30",
                "category": "smile",
                "cell_key": "left30|smile",
                "quality": 0.75,
                "cell_score": 0.70,
                "video": "clip",
            },
        ]
        _seed_bank(bank_home, member_id, entries)

        # Mock PortraitGenerator — capture what refs were passed
        captured_request = {}

        def capture_generate(request):
            captured_request["ref_paths"] = request.ref_paths
            captured_request["prompt"] = request.style_prompt
            return MagicMock(success=True, output_paths=["/out/portrait.png"], error=None)

        events: List[StepEvent] = []

        with patch("portrait981.pipeline.Momentscan") as MockScanner, \
             patch("portrait981.pipeline.PortraitGenerator") as mock_gen_cls:
            scanner = MockScanner.return_value
            scanner.scan.return_value = [FakeFrameResult() for _ in range(150)]
            scanner.initialize.return_value = None
            scanner.shutdown.return_value = None
            mock_gen_cls.return_value.generate.side_effect = capture_generate

            pipeline = Portrait981Pipeline(
                PipelineConfig(),
                on_step=events.append,
            )
            try:
                result = pipeline.run_one(JobSpec(
                    video_path="clip.mp4",
                    member_id=member_id,
                    pose="frontal",
                    category="smile",
                    prompt="oil painting portrait",
                    top_k=3,
                ))
            finally:
                pipeline.shutdown()

        # Verify full pipeline completed
        assert result.status == JobStatus.DONE
        assert result.error is None

        # Verify scan was called
        scanner.scan.assert_called_once()

        # Verify lookup found the right ref (frontal + smile only)
        assert result.ref_count == 1
        assert len(captured_request["ref_paths"]) == 1
        assert "frontal_smile" in captured_request["ref_paths"][0]

        # Verify prompt forwarded
        assert captured_request["prompt"] == "oil painting portrait"

        # Verify step events
        steps = [(e.step, e.status) for e in events]
        assert ("scan", "started") in steps
        assert ("scan", "completed") in steps
        assert ("lookup", "completed") in steps
        assert ("generate", "completed") in steps

    def test_lookup_all_refs_no_filter(self, bank_home):
        """No pose/category filter → all 3 refs used."""
        member_id = "integ_test_2"
        entries = [
            {"file": f"f{i}.jpg", "frame_idx": i, "timestamp_ms": i * 100.0,
             "pose_name": "frontal", "category": "smile",
             "cell_key": f"frontal|smile", "quality": 0.8, "cell_score": 0.9 - i * 0.1,
             "video": "v"}
            for i in range(5)
        ]
        _seed_bank(bank_home, member_id, entries)

        with patch("portrait981.pipeline.Momentscan") as MockScanner, \
             patch("portrait981.pipeline.PortraitGenerator") as mock_gen_cls:
            scanner = MockScanner.return_value
            scanner.scan.return_value = [FakeFrameResult() for _ in range(100)]
            scanner.initialize.return_value = None
            scanner.shutdown.return_value = None
            mock_gen_cls.return_value.generate.return_value = MagicMock(
                success=True, output_paths=[], error=None,
            )

            pipeline = Portrait981Pipeline()
            try:
                result = pipeline.run_one(JobSpec(
                    video_path="v.mp4",
                    member_id=member_id,
                    top_k=3,  # only top 3 of 5
                ))
            finally:
                pipeline.shutdown()

        assert result.status == JobStatus.DONE
        assert result.ref_count == 3  # top_k=3

    def test_empty_bank_skips_generate(self, bank_home):
        """No frames in bank → generate skipped, DONE."""
        member_id = "integ_empty"

        events: List[StepEvent] = []
        with patch("portrait981.pipeline.Momentscan") as MockScanner, \
             patch("portrait981.pipeline.PortraitGenerator") as mock_gen_cls:
            scanner = MockScanner.return_value
            scanner.scan.return_value = [FakeFrameResult() for _ in range(100)]
            scanner.initialize.return_value = None
            scanner.shutdown.return_value = None

            pipeline = Portrait981Pipeline(on_step=events.append)
            try:
                result = pipeline.run_one(JobSpec(
                    video_path="v.mp4",
                    member_id=member_id,
                ))
            finally:
                pipeline.shutdown()

        assert result.status == JobStatus.DONE
        assert result.ref_count == 0
        mock_gen_cls.return_value.generate.assert_not_called()

        # Verify generate was skipped
        steps = [(e.step, e.status) for e in events]
        assert ("generate", "skipped") in steps


class TestPartialFailureAndRetry:
    """Test PARTIAL → retry workflow with real bank."""

    def test_scan_ok_generate_fails_then_retry(self, bank_home):
        """scan seeds bank → generate fails (PARTIAL) → retry generate_only (DONE)."""
        member_id = "integ_retry"

        # Seed bank
        entries = [
            {"file": "ref.jpg", "frame_idx": 1, "timestamp_ms": 100.0,
             "pose_name": "frontal", "category": "neutral",
             "cell_key": "frontal|neutral", "quality": 0.9, "cell_score": 0.95,
             "video": "v"},
        ]
        _seed_bank(bank_home, member_id, entries)

        # Run 1: scan ok, generate fails
        events_1: List[StepEvent] = []
        with patch("portrait981.pipeline.Momentscan") as MockScanner, \
             patch("portrait981.pipeline.PortraitGenerator") as mock_gen_cls:
            scanner = MockScanner.return_value
            scanner.scan.return_value = [FakeFrameResult() for _ in range(50)]
            scanner.initialize.return_value = None
            scanner.shutdown.return_value = None
            mock_gen_cls.return_value.generate.side_effect = ConnectionError(
                "Connection refused: ComfyUI not running"
            )

            pipeline = Portrait981Pipeline(on_step=events_1.append)
            try:
                r1 = pipeline.run_one(JobSpec(
                    video_path="v.mp4",
                    member_id=member_id,
                ))
            finally:
                pipeline.shutdown()

        assert r1.status == JobStatus.PARTIAL
        assert r1.scan_result is not None
        assert "Connection refused" in r1.error
        assert r1.ref_count == 1

        # Verify events show scan ok + generate failed
        steps_1 = [(e.step, e.status) for e in events_1]
        assert ("scan", "completed") in steps_1
        assert ("generate", "failed") in steps_1

        # Run 2: generate_only retry — ComfyUI now available
        events_2: List[StepEvent] = []
        with patch("portrait981.pipeline.Momentscan") as MockScanner, \
             patch("portrait981.pipeline.PortraitGenerator") as mock_gen_cls:
            scanner = MockScanner.return_value
            scanner.initialize.return_value = None
            scanner.shutdown.return_value = None
            mock_gen_cls.return_value.generate.return_value = MagicMock(
                success=True,
                output_paths=["/out/portrait.png"],
                error=None,
            )

            pipeline = Portrait981Pipeline(on_step=events_2.append)
            try:
                r2 = pipeline.run_one(JobSpec(
                    member_id=member_id,
                    generate_only=True,
                ))
            finally:
                pipeline.shutdown()

        assert r2.status == JobStatus.DONE
        assert r2.error is None
        assert r2.ref_count == 1
        scanner.scan.assert_not_called()  # scan skipped

        # Verify events show scan skipped + generate ok
        steps_2 = [(e.step, e.status) for e in events_2]
        assert ("scan", "skipped") in steps_2
        assert ("generate", "completed") in steps_2


class TestStatusQuery:
    """Test p981 status command with real bank."""

    def test_status_shows_coverage(self, bank_home, capsys):
        """p981 status → pose x category 테이블 출력."""
        member_id = "integ_status"
        entries = [
            {"file": "a.jpg", "frame_idx": 1, "timestamp_ms": 100.0,
             "pose_name": "frontal", "category": "smile",
             "cell_key": "frontal|smile", "quality": 0.9, "cell_score": 0.95,
             "video": "v"},
            {"file": "b.jpg", "frame_idx": 2, "timestamp_ms": 200.0,
             "pose_name": "frontal", "category": "neutral",
             "cell_key": "frontal|neutral", "quality": 0.8, "cell_score": 0.85,
             "video": "v"},
            {"file": "c.jpg", "frame_idx": 3, "timestamp_ms": 300.0,
             "pose_name": "left30", "category": "smile",
             "cell_key": "left30|smile", "quality": 0.7, "cell_score": 0.75,
             "video": "v"},
        ]
        _seed_bank(bank_home, member_id, entries)

        from portrait981.cli import main
        main(["status", member_id])

        captured = capsys.readouterr()
        assert "3 frames" in captured.out
        assert "frontal" in captured.out
        assert "smile" in captured.out
        assert "neutral" in captured.out
        assert "left30" in captured.out


class TestBatchWithBank:
    """Batch processing with real bank."""

    def test_batch_multiple_members(self, bank_home, tmp_path):
        """2명의 member → 각각 독립적으로 bank에 저장 → 각각 조회 가능."""
        # Seed banks for two members
        for mid in ("member_a", "member_b"):
            entries = [
                {"file": f"{mid}_ref.jpg", "frame_idx": 1, "timestamp_ms": 100.0,
                 "pose_name": "frontal", "category": "neutral",
                 "cell_key": "frontal|neutral", "quality": 0.9, "cell_score": 0.9,
                 "video": mid},
            ]
            _seed_bank(bank_home, mid, entries)

        with patch("portrait981.pipeline.Momentscan") as MockScanner, \
             patch("portrait981.pipeline.PortraitGenerator") as mock_gen_cls:
            scanner = MockScanner.return_value
            scanner.scan.return_value = [FakeFrameResult() for _ in range(100)]
            scanner.initialize.return_value = None
            scanner.shutdown.return_value = None
            mock_gen_cls.return_value.generate.return_value = MagicMock(
                success=True, output_paths=[], error=None,
            )

            pipeline = Portrait981Pipeline(PipelineConfig(max_scan_workers=2))
            try:
                results = pipeline.run_batch([
                    JobSpec(video_path="a.mp4", member_id="member_a"),
                    JobSpec(video_path="b.mp4", member_id="member_b"),
                ])
            finally:
                pipeline.shutdown()

        assert len(results) == 2
        assert all(r.status == JobStatus.DONE for r in results)
        assert all(r.ref_count == 1 for r in results)

        # Verify each member's bank is independently queryable
        from personmemory.ingest import lookup_frames
        frames_a = lookup_frames("member_a")
        frames_b = lookup_frames("member_b")
        assert len(frames_a) == 1
        assert len(frames_b) == 1
        assert "member_a" in frames_a[0]["file"]
        assert "member_b" in frames_b[0]["file"]
