"""ComfyUI REST API client.

Uses urllib only — no external HTTP dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional


class ComfyClient:
    """Minimal ComfyUI REST API client."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", api_key: Optional[str] = None):
        url = base_url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        self.base_url = url
        self.api_key = api_key

    def _request(self, url: str, *, data: Optional[bytes] = None, method: Optional[str] = None) -> bytes:
        """Send an HTTP request with auth headers if configured."""
        headers: Dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow prompt for execution.

        Args:
            workflow: ComfyUI workflow dict (API format).

        Returns:
            prompt_id string.

        Raises:
            urllib.error.URLError: On connection failure.
        """
        payload = json.dumps({"prompt": workflow}).encode("utf-8")
        raw = self._request(f"{self.base_url}/prompt", data=payload, method="POST")
        data = json.loads(raw.decode("utf-8"))
        return data["prompt_id"]

    def get_history(self, prompt_id: str) -> Optional[dict]:
        """Get execution history for a prompt.

        Args:
            prompt_id: The prompt ID from queue_prompt.

        Returns:
            History entry dict if completed, None if still pending.
        """
        url = f"{self.base_url}/history/{prompt_id}"
        raw = self._request(url)
        data = json.loads(raw.decode("utf-8"))
        entry = data.get(prompt_id)
        if entry and entry.get("outputs"):
            return entry
        return None

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download a single image from ComfyUI.

        Args:
            filename: Image filename.
            subfolder: Subfolder within the output directory.
            folder_type: "output", "input", or "temp".

        Returns:
            Raw image bytes.
        """
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        })
        url = f"{self.base_url}/view?{params}"
        return self._request(url)

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> dict:
        """Poll history until prompt completes.

        Args:
            prompt_id: The prompt ID to wait for.
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between polls.

        Returns:
            History entry dict with outputs.

        Raises:
            TimeoutError: If not completed within timeout.
        """
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"ComfyUI prompt {prompt_id} did not complete within {timeout}s"
                )
            entry = self.get_history(prompt_id)
            if entry is not None:
                return entry
            time.sleep(poll_interval)

    def download_images(self, history_entry: dict, output_dir: Path) -> List[Path]:
        """Download all output images from a completed prompt.

        Args:
            history_entry: Completed history entry from wait_for_completion.
            output_dir: Local directory to save images.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        outputs = history_entry.get("outputs", {})

        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            for img_info in images:
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                folder_type = img_info.get("type", "output")

                image_data = self.get_image(filename, subfolder, folder_type)
                out_path = output_dir / filename
                out_path.write_bytes(image_data)
                saved.append(out_path)

        return saved
