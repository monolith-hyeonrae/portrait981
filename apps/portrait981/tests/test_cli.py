"""CLI tests for portrait981."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portrait981.cli import _build_parser, _derive_member_id, _discover_videos, main
from portrait981.types import JobStatus


class TestArgParsing:
    def test_run_subcommand_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            "run", "video.mp4", "--member-id", "test_3",
            "--pose", "frontal", "--category", "warm_smile",
            "--prompt", "portrait", "--workflow", "custom",
            "--top-k", "5", "--scan-only",
        ])
        assert args.command == "run"
        assert args.video == "video.mp4"
        assert args.member_id == "test_3"
        assert args.pose == "frontal"
        assert args.category == "warm_smile"
        assert args.prompt == "portrait"
        assert args.workflow == "custom"
        assert args.top_k == 5
        assert args.scan_only is True

    def test_scan_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["scan", "v.mp4", "--member-id", "m1"])
        assert args.command == "scan"
        assert args.video == "v.mp4"
        assert args.member_id == "m1"

    def test_generate_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args([
            "generate", "test_3", "--pose", "left30", "--prompt", "oil painting",
        ])
        assert args.command == "generate"
        assert args.member_id == "test_3"
        assert args.pose == "left30"
        assert args.prompt == "oil painting"

    def test_status_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["status", "test_3"])
        assert args.command == "status"
        assert args.member_id == "test_3"

    def test_batch_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args([
            "batch", "/videos", "--member-id-from", "parent",
            "--workers", "4", "--scan-only",
        ])
        assert args.command == "batch"
        assert args.directory == "/videos"
        assert args.member_id_from == "parent"
        assert args.workers == 4
        assert args.scan_only is True

    def test_verbose_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "v.mp4", "--member-id", "m1", "-v"])
        assert args.verbose is True


class TestMemberIdDerivation:
    def test_from_filename(self):
        assert _derive_member_id(Path("/dir/person_1.mp4"), "filename") == "person_1"

    def test_from_parent(self):
        assert _derive_member_id(Path("/videos/person_1/clip.mp4"), "parent") == "person_1"


class TestDiscoverVideos:
    def test_discovers_videos(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.avi").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.mov").touch()
        videos = _discover_videos(str(tmp_path))
        names = [v.name for v in videos]
        assert "a.mp4" in names
        assert "b.avi" in names
        assert "d.mov" in names
        assert "c.txt" not in names

    def test_nonexistent_dir(self):
        with pytest.raises(SystemExit):
            _discover_videos("/nonexistent/path")


class TestMainImport:
    def test_main_importable(self):
        from portrait981.cli import main
        assert callable(main)

    def test_no_command_shows_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1


class TestCLIExecution:
    def test_run_invokes_pipeline(self):
        from portrait981.types import JobResult, JobSpec as JS, StepTiming
        with patch("portrait981.cli.Portrait981Pipeline") as cls:
            inst = cls.return_value
            mock_result = JobResult(
                job=JS(video_path="v.mp4", member_id="t1"),
                status=JobStatus.DONE,
                timing=StepTiming(total_sec=1.0),
            )
            inst.run_one.return_value = mock_result
            main(["run", "v.mp4", "--member-id", "t1"])
            inst.run_one.assert_called_once()
            inst.shutdown.assert_called_once()

    def test_status_invokes_lookup(self):
        with patch("momentbank.ingest.lookup_frames") as m:
            m.return_value = [
                {"pose_name": "frontal", "category": "smile"},
                {"pose_name": "frontal", "category": "smile"},
            ]
            main(["status", "test_3"])
            m.assert_called_once_with("test_3")

    def test_partial_result_shows_retry_hint(self, capsys):
        from portrait981.types import JobResult, JobSpec as JS, StepTiming
        with patch("portrait981.cli.Portrait981Pipeline") as cls:
            inst = cls.return_value
            mock_result = JobResult(
                job=JS(video_path="v.mp4", member_id="test_5"),
                status=JobStatus.PARTIAL,
                scan_result=MagicMock(frame_count=200, highlights=[1, 2]),
                ref_count=3,
                error="ComfyUI connection refused",
                timing=StepTiming(scan_sec=5.0, lookup_sec=0.1, generate_sec=1.0, total_sec=6.1),
            )
            inst.run_one.return_value = mock_result
            main(["run", "v.mp4", "--member-id", "test_5"])

        captured = capsys.readouterr()
        assert "PARTIAL" in captured.out
        assert "p981 generate test_5" in captured.out
        assert "ComfyUI connection refused" in captured.out
