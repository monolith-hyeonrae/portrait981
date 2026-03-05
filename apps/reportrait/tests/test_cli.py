"""Tests for reportrait CLI."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from reportrait.cli import main


class TestMainImport:
    """Verify CLI entry point is importable."""

    def test_main_callable(self):
        assert callable(main)

    def test_no_command_returns_zero(self):
        """No subcommand → print help, exit 0."""
        assert main([]) == 0


class TestGenerateDryRun:
    """Test --dry-run mode: lookup → inject → print JSON."""

    @patch("momentbank.ingest.lookup_frames")
    def test_dry_run_outputs_workflow_json(self, mock_lookup, capsys):
        mock_lookup.return_value = [
            {"path": "/tmp/ref1.jpg", "cell_score": 0.9},
            {"path": "/tmp/ref2.jpg", "cell_score": 0.8},
        ]

        ret = main(["generate", "test_member", "--dry-run"])

        assert ret == 0
        captured = capsys.readouterr()
        workflow = json.loads(captured.out)
        # Reference images should be injected
        ref_nodes = [
            n for n in workflow.values()
            if isinstance(n, dict)
            and n.get("class_type") == "LoadImage"
            and n.get("_meta", {}).get("role") == "reference"
        ]
        assert len(ref_nodes) == 2
        injected_paths = [n["inputs"]["image"] for n in ref_nodes]
        assert "/tmp/ref1.jpg" in injected_paths
        assert "/tmp/ref2.jpg" in injected_paths

    @patch("momentbank.ingest.lookup_frames")
    def test_dry_run_with_prompt(self, mock_lookup, capsys):
        mock_lookup.return_value = [
            {"path": "/tmp/ref1.jpg", "cell_score": 0.9},
        ]

        ret = main([
            "generate", "test_member",
            "--prompt", "cinematic portrait",
            "--dry-run",
        ])

        assert ret == 0
        workflow = json.loads(capsys.readouterr().out)
        positive_nodes = [
            n for n in workflow.values()
            if isinstance(n, dict)
            and n.get("class_type") == "CLIPTextEncode"
            and n.get("_meta", {}).get("role") == "positive"
        ]
        assert len(positive_nodes) >= 1
        assert positive_nodes[0]["inputs"]["text"] == "cinematic portrait"

    @patch("momentbank.ingest.lookup_frames")
    def test_dry_run_passes_pose_and_category(self, mock_lookup):
        mock_lookup.return_value = [{"path": "/tmp/ref.jpg", "cell_score": 0.5}]

        main([
            "generate", "m1",
            "--pose", "left30",
            "--category", "warm_smile",
            "--top", "5",
            "--dry-run",
        ])

        mock_lookup.assert_called_once_with(
            "m1", pose="left30", category="warm_smile", top_k=5
        )


class TestGenerateWithRef:
    """Test --ref direct reference image injection."""

    def test_ref_dry_run(self, tmp_path, capsys):
        """--ref with existing files injects paths into workflow."""
        img1 = tmp_path / "face1.jpg"
        img2 = tmp_path / "face2.jpg"
        img1.write_bytes(b"fake")
        img2.write_bytes(b"fake")

        ret = main([
            "generate",
            "--ref", str(img1), str(img2),
            "--dry-run",
        ])

        assert ret == 0
        workflow = json.loads(capsys.readouterr().out)
        ref_nodes = [
            n for n in workflow.values()
            if isinstance(n, dict)
            and n.get("class_type") == "LoadImage"
            and n.get("_meta", {}).get("role") == "reference"
        ]
        injected = [n["inputs"]["image"] for n in ref_nodes]
        assert str(img1.resolve()) in injected
        assert str(img2.resolve()) in injected

    def test_ref_missing_file_error(self, capsys):
        """--ref with non-existent file → error."""
        ret = main(["generate", "--ref", "/no/such/file.jpg", "--dry-run"])

        assert ret == 1
        assert "not found" in capsys.readouterr().err

    def test_ref_skips_lookup(self, tmp_path):
        """--ref should not call lookup_frames at all."""
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"fake")

        with patch("momentbank.ingest.lookup_frames") as mock_lookup:
            main(["generate", "--ref", str(img), "--dry-run"])
            mock_lookup.assert_not_called()

    def test_ref_with_workflow_file(self, tmp_path, capsys):
        """--ref + --workflow <file.json> loads external workflow and injects."""
        img = tmp_path / "face.jpg"
        img.write_bytes(b"fake")

        # Create a custom I2I workflow JSON
        custom_wf = {
            "10": {
                "class_type": "LoadImage",
                "inputs": {"image": "placeholder.png"},
                "_meta": {"role": "reference", "title": "Source"},
            },
            "20": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "default", "clip": ["30", 0]},
                "_meta": {"role": "positive"},
            },
        }
        wf_file = tmp_path / "i2i_workflow.json"
        wf_file.write_text(json.dumps(custom_wf))

        ret = main([
            "generate",
            "--ref", str(img),
            "--workflow", str(wf_file),
            "--prompt", "cinematic",
            "--dry-run",
        ])

        assert ret == 0
        result = json.loads(capsys.readouterr().out)
        assert result["10"]["inputs"]["image"] == str(img.resolve())
        assert result["20"]["inputs"]["text"] == "cinematic"

    def test_no_member_id_no_ref_error(self, capsys):
        """Neither member_id nor --ref → error."""
        ret = main(["generate", "--dry-run"])

        assert ret == 1
        assert "member_id or --ref required" in capsys.readouterr().err


class TestGenerateNoRefs:
    """Test error when no reference frames found."""

    @patch("momentbank.ingest.lookup_frames")
    def test_no_frames_error(self, mock_lookup, capsys):
        mock_lookup.return_value = []

        ret = main(["generate", "unknown_member"])

        assert ret == 1
        captured = capsys.readouterr()
        assert "No frames found" in captured.err
        assert "unknown_member" in captured.err

    @patch("momentbank.ingest.lookup_frames")
    def test_no_frames_with_filters_in_error(self, mock_lookup, capsys):
        mock_lookup.return_value = []

        main(["generate", "m1", "--pose", "frontal", "--category", "smile"])

        captured = capsys.readouterr()
        assert "pose=frontal" in captured.err
        assert "category=smile" in captured.err


class TestGenerateCall:
    """Test actual generate path (mocked ComfyUI)."""

    @patch("momentbank.ingest.lookup_frames")
    @patch("reportrait.generator.PortraitGenerator")
    def test_generate_success(self, mock_gen_cls, mock_lookup, capsys):
        mock_lookup.return_value = [
            {"path": "/tmp/ref1.jpg", "cell_score": 0.9},
        ]

        from reportrait.types import GenerationResult

        mock_gen = MagicMock()
        mock_gen.generate.return_value = GenerationResult(
            success=True,
            output_paths=["/tmp/out/portrait_001.png"],
            elapsed_sec=5.2,
        )
        mock_gen_cls.return_value = mock_gen

        ret = main(["generate", "test_member", "--prompt", "portrait"])

        assert ret == 0
        captured = capsys.readouterr()
        assert "1 image(s)" in captured.out
        assert "portrait_001.png" in captured.out

        # Verify generator was called with correct request
        call_args = mock_gen.generate.call_args
        request = call_args[0][0]
        assert request.ref_paths == ["/tmp/ref1.jpg"]
        assert request.style_prompt == "portrait"

    @patch("momentbank.ingest.lookup_frames")
    @patch("reportrait.generator.PortraitGenerator")
    def test_generate_failure(self, mock_gen_cls, mock_lookup, capsys):
        mock_lookup.return_value = [
            {"path": "/tmp/ref1.jpg", "cell_score": 0.9},
        ]

        from reportrait.types import GenerationResult

        mock_gen = MagicMock()
        mock_gen.generate.return_value = GenerationResult(
            success=False,
            error="Connection refused",
        )
        mock_gen_cls.return_value = mock_gen

        ret = main(["generate", "test_member"])

        assert ret == 1
        captured = capsys.readouterr()
        assert "Connection refused" in captured.err
