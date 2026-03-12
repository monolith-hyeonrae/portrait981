"""Tests for S3 client."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portrait981_serve.config import ServeConfig
from portrait981_serve.s3 import S3Client


class TestParseS3Uri:
    def test_valid(self):
        bucket, key = S3Client.parse_s3_uri("s3://my-bucket/path/to/video.mp4")
        assert bucket == "my-bucket"
        assert key == "path/to/video.mp4"

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Not an S3 URI"):
            S3Client.parse_s3_uri("http://example.com/file.mp4")

    def test_root_key(self):
        bucket, key = S3Client.parse_s3_uri("s3://bucket/file.mp4")
        assert bucket == "bucket"
        assert key == "file.mp4"


class TestS3Client:
    @patch("portrait981_serve.s3.boto3")
    def test_download(self, mock_boto3, tmp_path):
        config = ServeConfig(s3_access_key="KEY", s3_secret_key="SECRET")
        client = S3Client(config)

        local = client.download("s3://bucket/video.mp4", local_dir=str(tmp_path))
        assert local.name == "video.mp4"
        mock_boto3.client.return_value.download_file.assert_called_once_with(
            "bucket", "video.mp4", str(local),
        )

    @patch("portrait981_serve.s3.boto3")
    def test_upload(self, mock_boto3, tmp_path):
        config = ServeConfig(
            s3_output_bucket="out-bucket",
            s3_output_prefix="portrait/",
            s3_access_key="KEY",
            s3_secret_key="SECRET",
        )
        client = S3Client(config)
        local_file = tmp_path / "result.png"
        local_file.touch()

        uri = client.upload(local_file, key="portrait/123/result.png")
        assert uri == "s3://out-bucket/portrait/123/result.png"
        mock_boto3.client.return_value.upload_file.assert_called_once()

    @patch("portrait981_serve.s3.boto3")
    def test_upload_no_bucket_raises(self, mock_boto3):
        config = ServeConfig(s3_output_bucket="")
        client = S3Client(config)
        with pytest.raises(RuntimeError, match="P981_S3_OUTPUT_BUCKET"):
            client.upload(Path("/fake/file.png"))

    @patch("portrait981_serve.s3.boto3")
    def test_cleanup(self, mock_boto3, tmp_path):
        config = ServeConfig()
        client = S3Client(config)
        f = tmp_path / "temp.mp4"
        f.touch()
        assert f.exists()
        client.cleanup(f)
        assert not f.exists()
