"""Tests for server configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from portrait981_serve.config import ServeConfig


class TestServeConfig:
    def test_defaults(self):
        config = ServeConfig()
        assert config.top_k == 3
        assert config.s3_region == "ap-northeast-2"
        assert config.comfy_urls == ["http://127.0.0.1:8188"]

    def test_from_env(self):
        env = {
            "P981_DEFAULT_WORKFLOW": "spring-2026",
            "P981_TOP_K": "5",
            "P981_COMFY_URLS": "http://a:8188,http://b:8188",
            "P981_S3_ACCESS_KEY": "AKID",
            "P981_S3_SECRET_KEY": "SECRET",
            "P981_S3_OUTPUT_BUCKET": "my-bucket",
        }
        with patch.dict(os.environ, env, clear=False):
            config = ServeConfig.from_env()

        assert config.default_workflow == "spring-2026"
        assert config.top_k == 5
        assert config.comfy_urls == ["http://a:8188", "http://b:8188"]
        assert config.s3_access_key == "AKID"
        assert config.s3_output_bucket == "my-bucket"

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ServeConfig.from_env()
        assert config.default_workflow == "default"
        assert config.scan_fps == 10
