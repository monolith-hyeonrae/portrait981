"""Test fixtures for portrait981-serve."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from portrait981_serve.app import create_app
from portrait981_serve.config import ServeConfig


@pytest.fixture
def config():
    """Minimal test config with no real S3/ComfyUI."""
    return ServeConfig(
        default_workflow="test-workflow",
        default_prompt="test prompt",
        comfy_urls=["http://comfy1:8188", "http://comfy2:8188"],
        s3_output_bucket="test-bucket",
        s3_access_key="FAKE_KEY",
        s3_secret_key="FAKE_SECRET",
    )


@pytest.fixture
def mock_boto3():
    with patch("portrait981_serve.s3.boto3") as m:
        yield m


@pytest.fixture
def mock_pipeline():
    with patch("portrait981_serve.routes.Portrait981Pipeline") as cls:
        inst = cls.return_value
        yield inst


@pytest.fixture
def mock_lookup():
    with patch("portrait981_serve.routes.lookup_frames") as m:
        yield m


@pytest.fixture
def client(config, mock_boto3, mock_pipeline):
    """FastAPI test client with mocked dependencies."""
    app = create_app(config)
    return TestClient(app)
