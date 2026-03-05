"""Shared fixtures for reportrait tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SAMPLE_WORKFLOW = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {"image": "placeholder.png"},
        "_meta": {"role": "reference"},
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {"image": "placeholder2.png"},
        "_meta": {"role": "reference"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a portrait photo", "clip": ["4", 0]},
        "_meta": {"role": "positive"},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad quality", "clip": ["4", 0]},
        "_meta": {"role": "negative"},
    },
    "5": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "portrait"},
    },
}


@pytest.fixture
def sample_workflow():
    """Return a sample ComfyUI workflow dict."""
    return json.loads(json.dumps(SAMPLE_WORKFLOW))  # deep copy


@pytest.fixture
def templates_dir(tmp_path: Path, sample_workflow):
    """Create a temp templates directory with a default.json workflow."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "default.json").write_text(json.dumps(sample_workflow))
    return tpl_dir


@pytest.fixture
def mock_comfy_client():
    """Create a mock ComfyClient that simulates queue -> complete -> download."""
    client = MagicMock()
    client.queue_prompt.return_value = "test-prompt-id"
    client.wait_for_completion.return_value = {
        "outputs": {
            "5": {
                "images": [
                    {"filename": "portrait_00001.png", "subfolder": "", "type": "output"},
                ],
            },
        },
    }
    client.get_image.return_value = b"\x89PNG\r\n\x1a\nfake_image_data"
    return client
