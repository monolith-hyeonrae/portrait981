"""Tests for ComfyClient."""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from reportrait.comfy_client import ComfyClient


def _mock_urlopen(response_data: dict | bytes, status: int = 200):
    """Create a mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    if isinstance(response_data, bytes):
        mock_resp.read.return_value = response_data
    else:
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestQueuePrompt:
    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_returns_prompt_id(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"prompt_id": "abc-123"})
        client = ComfyClient()

        pid = client.queue_prompt({"1": {"class_type": "Test"}})

        assert pid == "abc-123"
        mock_urlopen.assert_called_once()

    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_sends_json_payload(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"prompt_id": "x"})
        client = ComfyClient("http://localhost:9999")

        workflow = {"1": {"class_type": "LoadImage"}}
        client.queue_prompt(workflow)

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == "http://localhost:9999/prompt"
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["prompt"] == workflow


class TestGetHistory:
    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_returns_entry_when_complete(self, mock_urlopen):
        entry = {"outputs": {"5": {"images": []}}}
        mock_urlopen.return_value = _mock_urlopen({"abc": entry})
        client = ComfyClient()

        result = client.get_history("abc")

        assert result == entry

    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_returns_none_when_pending(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"abc": {}})
        client = ComfyClient()

        result = client.get_history("abc")

        assert result is None

    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_returns_none_when_not_found(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({})
        client = ComfyClient()

        result = client.get_history("missing")

        assert result is None


class TestGetImage:
    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_returns_bytes(self, mock_urlopen):
        image_bytes = b"\x89PNG\r\nfake"
        mock_urlopen.return_value = _mock_urlopen(image_bytes)
        client = ComfyClient()

        data = client.get_image("test.png", subfolder="sub", folder_type="output")

        assert data == image_bytes


class TestWaitForCompletion:
    @patch("reportrait.comfy_client.time.sleep")
    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_returns_on_completion(self, mock_urlopen, mock_sleep):
        entry = {"outputs": {"5": {"images": []}}}
        # First call: pending, second call: complete
        mock_urlopen.side_effect = [
            _mock_urlopen({"abc": {}}),
            _mock_urlopen({"abc": entry}),
        ]
        client = ComfyClient()

        result = client.wait_for_completion("abc", timeout=10, poll_interval=0.01)

        assert result == entry

    @patch("reportrait.comfy_client.time.sleep")
    @patch("reportrait.comfy_client.time.monotonic")
    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_raises_timeout(self, mock_urlopen, mock_monotonic, mock_sleep):
        mock_urlopen.return_value = _mock_urlopen({"abc": {}})
        # Simulate time passing beyond timeout
        mock_monotonic.side_effect = [0.0, 0.0, 301.0]
        client = ComfyClient()

        with pytest.raises(TimeoutError, match="did not complete"):
            client.wait_for_completion("abc", timeout=300.0, poll_interval=0.01)


class TestDownloadImages:
    @patch("reportrait.comfy_client.urllib.request.urlopen")
    def test_downloads_and_saves(self, mock_urlopen, tmp_path):
        image_bytes = b"\x89PNG\r\nfake_data"
        mock_urlopen.return_value = _mock_urlopen(image_bytes)
        client = ComfyClient()

        history = {
            "outputs": {
                "5": {
                    "images": [
                        {"filename": "out_001.png", "subfolder": "", "type": "output"},
                    ],
                },
            },
        }

        paths = client.download_images(history, tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "out_001.png"
        assert paths[0].read_bytes() == image_bytes

    def test_empty_outputs(self, tmp_path):
        client = ComfyClient()
        paths = client.download_images({"outputs": {}}, tmp_path)
        assert paths == []
