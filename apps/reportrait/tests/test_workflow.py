"""Tests for workflow template loading and injection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reportrait.workflow import load_template, inject_references, inject_prompt


class TestLoadTemplate:
    def test_load_existing(self, templates_dir):
        wf = load_template("default", templates_dir=templates_dir)
        assert "1" in wf
        assert wf["1"]["class_type"] == "LoadImage"

    def test_load_from_file_path(self, templates_dir):
        """load_template accepts a direct .json file path."""
        file_path = str(templates_dir / "default.json")
        wf = load_template(file_path)
        assert "1" in wf
        assert wf["1"]["class_type"] == "LoadImage"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_template("nonexistent", templates_dir=tmp_path)


class TestInjectReferences:
    def test_injects_single_path(self, sample_workflow):
        result = inject_references(sample_workflow, ["/path/to/ref.jpg"])

        assert result["1"]["inputs"]["image"] == "/path/to/ref.jpg"
        assert result["2"]["inputs"]["image"] == "/path/to/ref.jpg"  # cycled

    def test_injects_multiple_paths(self, sample_workflow):
        result = inject_references(sample_workflow, ["/a.jpg", "/b.jpg"])

        assert result["1"]["inputs"]["image"] == "/a.jpg"
        assert result["2"]["inputs"]["image"] == "/b.jpg"

    def test_cycles_when_more_paths(self, sample_workflow):
        result = inject_references(sample_workflow, ["/a.jpg", "/b.jpg", "/c.jpg"])

        # Only 2 reference nodes, so /c.jpg is not used
        assert result["1"]["inputs"]["image"] == "/a.jpg"
        assert result["2"]["inputs"]["image"] == "/b.jpg"

    def test_does_not_modify_original(self, sample_workflow):
        original_image = sample_workflow["1"]["inputs"]["image"]
        inject_references(sample_workflow, ["/new.jpg"])
        assert sample_workflow["1"]["inputs"]["image"] == original_image

    def test_empty_paths_returns_unchanged(self, sample_workflow):
        result = inject_references(sample_workflow, [])
        assert result is sample_workflow  # no copy needed

    def test_no_reference_nodes(self):
        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}, "_meta": {}},
        }
        result = inject_references(wf, ["/ref.jpg"])
        # No role="reference" → no changes
        assert result["1"]["inputs"]["image"] == "x.png"


class TestInjectPrompt:
    def test_injects_into_positive_node(self, sample_workflow):
        result = inject_prompt(sample_workflow, "beautiful portrait, high quality")

        assert result["3"]["inputs"]["text"] == "beautiful portrait, high quality"
        # Negative node unchanged
        assert result["4"]["inputs"]["text"] == "bad quality"

    def test_does_not_modify_original(self, sample_workflow):
        original_text = sample_workflow["3"]["inputs"]["text"]
        inject_prompt(sample_workflow, "new prompt")
        assert sample_workflow["3"]["inputs"]["text"] == original_text

    def test_empty_prompt_returns_unchanged(self, sample_workflow):
        result = inject_prompt(sample_workflow, "")
        assert result is sample_workflow
