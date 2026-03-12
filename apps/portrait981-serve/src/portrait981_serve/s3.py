"""S3 client for video download and result upload."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3

from portrait981_serve.config import ServeConfig

logger = logging.getLogger(__name__)


class S3Client:
    """Thin wrapper around boto3 for portrait981-serve S3 operations."""

    def __init__(self, config: ServeConfig) -> None:
        self._config = config
        kwargs: dict = {"region_name": config.s3_region}
        if config.s3_access_key and config.s3_secret_key:
            kwargs["aws_access_key_id"] = config.s3_access_key
            kwargs["aws_secret_access_key"] = config.s3_secret_key
        self._client = boto3.client("s3", **kwargs)

    @staticmethod
    def parse_s3_uri(uri: str) -> tuple[str, str]:
        """Parse 's3://bucket/key' into (bucket, key)."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Not an S3 URI: {uri}")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    def download(self, s3_uri: str, local_dir: Optional[str] = None) -> Path:
        """Download an S3 object to a local temp path. Returns local file path."""
        bucket, key = self.parse_s3_uri(s3_uri)
        filename = Path(key).name

        if local_dir is None:
            local_dir = tempfile.mkdtemp(prefix="p981_fetch_")
        local_path = Path(local_dir) / filename

        logger.info("S3 download: s3://%s/%s -> %s", bucket, key, local_path)
        self._client.download_file(bucket, key, str(local_path))
        return local_path

    def upload(self, local_path: Path, key: Optional[str] = None) -> str:
        """Upload a local file to S3. Returns the S3 URI."""
        bucket = self._config.s3_output_bucket
        if not bucket:
            raise RuntimeError("P981_S3_OUTPUT_BUCKET not configured")

        if key is None:
            key = self._config.s3_output_prefix + local_path.name

        logger.info("S3 upload: %s -> s3://%s/%s", local_path, bucket, key)
        self._client.upload_file(str(local_path), bucket, key)
        return f"s3://{bucket}/{key}"

    def cleanup(self, local_path: Path) -> None:
        """Delete a local file after processing."""
        try:
            if local_path.exists():
                local_path.unlink()
                logger.debug("Cleaned up: %s", local_path)
        except OSError as e:
            logger.warning("Failed to clean up %s: %s", local_path, e)
