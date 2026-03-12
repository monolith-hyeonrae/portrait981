"""Server configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServeConfig:
    """portrait981-serve configuration. All values read from env vars."""

    # --- Generate defaults ---
    default_workflow: str = ""
    default_prompt: str = ""
    top_k: int = 3

    # --- ComfyUI node pool ---
    comfy_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8188"])
    comfy_api_key: Optional[str] = None

    # --- Scan ---
    scan_fps: int = 10
    scan_backend: str = "simple"
    collection_path: Optional[str] = None

    # --- S3 ---
    s3_output_bucket: str = ""
    s3_output_prefix: str = "portrait/"
    s3_region: str = "ap-northeast-2"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> ServeConfig:
        """Build config from P981_* environment variables."""
        urls_raw = os.environ.get("P981_COMFY_URLS", "http://127.0.0.1:8188")
        comfy_urls = [u.strip() for u in urls_raw.split(",") if u.strip()]

        return cls(
            default_workflow=os.environ.get("P981_DEFAULT_WORKFLOW", "default"),
            default_prompt=os.environ.get("P981_DEFAULT_PROMPT", ""),
            top_k=int(os.environ.get("P981_TOP_K", "3")),
            comfy_urls=comfy_urls,
            comfy_api_key=os.environ.get("P981_COMFY_API_KEY"),
            scan_fps=int(os.environ.get("P981_SCAN_FPS", "10")),
            scan_backend=os.environ.get("P981_SCAN_BACKEND", "simple"),
            collection_path=os.environ.get("P981_COLLECTION_PATH"),
            s3_output_bucket=os.environ.get("P981_S3_OUTPUT_BUCKET", ""),
            s3_output_prefix=os.environ.get("P981_S3_OUTPUT_PREFIX", "portrait/"),
            s3_region=os.environ.get("P981_S3_REGION", "ap-northeast-2"),
            s3_access_key=os.environ.get("P981_S3_ACCESS_KEY"),
            s3_secret_key=os.environ.get("P981_S3_SECRET_KEY"),
        )
