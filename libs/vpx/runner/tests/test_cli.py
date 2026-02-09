"""Tests for vpx CLI argument parsing."""

import pytest
from unittest.mock import patch, MagicMock

from vpx.runner.cli import _build_parser


class TestCLIParser:
    def test_run_basic(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "face.detect", "--input", "video.mp4"])
        assert args.command == "run"
        assert args.analyzers == "face.detect"
        assert args.input == "video.mp4"
        assert args.viz == "text"
        assert args.fps is None
        assert args.max_frames is None

    def test_run_with_fps(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "face.detect", "--input", "video.mp4", "--fps", "5"]
        )
        assert args.fps == 5.0

    def test_run_with_max_frames(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "face.detect", "--input", "video.mp4", "--max-frames", "100"]
        )
        assert args.max_frames == 100

    def test_run_multiple_analyzers(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "face.detect,face.expression", "--input", "video.mp4"]
        )
        assert args.analyzers == "face.detect,face.expression"

    def test_run_viz_live(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "face.detect", "--input", "video.mp4", "--viz", "live"]
        )
        assert args.viz == "live"

    def test_run_viz_save(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "face.detect", "-i", "video.mp4", "--viz", "save", "-o", "out.mp4"]
        )
        assert args.viz == "save"
        assert args.output == "out.mp4"

    def test_run_camera_input(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "face.detect", "--input", "0"])
        assert args.input == "0"

    def test_list_basic(self):
        parser = _build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_list_verbose(self):
        parser = _build_parser()
        args = parser.parse_args(["list", "--verbose"])
        assert args.verbose is True

    def test_no_command(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestCLIList:
    def test_list_shows_analyzers(self, capsys):
        from vpx.runner.cli import _cmd_list

        mock_ep = MagicMock()
        mock_ep.value = "vpx.face_detect:FaceDetectionAnalyzer"

        with patch(
            "visualpath.plugin.discovery.discover_analyzers",
            return_value={"face.detect": mock_ep},
        ):
            args = _build_parser().parse_args(["list"])
            _cmd_list(args)

        captured = capsys.readouterr()
        assert "face.detect" in captured.out

    def test_list_verbose_shows_path(self, capsys):
        from vpx.runner.cli import _cmd_list

        mock_ep = MagicMock()
        mock_ep.value = "vpx.face_detect:FaceDetectionAnalyzer"

        with patch(
            "visualpath.plugin.discovery.discover_analyzers",
            return_value={"face.detect": mock_ep},
        ):
            args = _build_parser().parse_args(["list", "--verbose"])
            _cmd_list(args)

        captured = capsys.readouterr()
        assert "FaceDetectionAnalyzer" in captured.out

    def test_list_empty(self, capsys):
        from vpx.runner.cli import _cmd_list

        with patch("visualpath.plugin.discovery.discover_analyzers", return_value={}):
            args = _build_parser().parse_args(["list"])
            _cmd_list(args)

        captured = capsys.readouterr()
        assert "No analyzers" in captured.out
