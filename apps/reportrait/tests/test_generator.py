"""Tests for PortraitGenerator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reportrait.types import GenerationConfig, GenerationRequest, GenerationResult
from reportrait.generator import PortraitGenerator


class TestGenerate:
    def test_successful_generation(self, templates_dir, mock_comfy_client, tmp_path):
        config = GenerationConfig(
            templates_dir=templates_dir,
            output_dir=tmp_path / "output",
        )
        gen = PortraitGenerator(config)
        gen.client = mock_comfy_client

        # Mock download_images to create actual files
        def fake_download(history, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "portrait_00001.png"
            p.write_bytes(b"fake")
            return [p]

        mock_comfy_client.download_images.side_effect = fake_download

        request = GenerationRequest(
            person_id=0,
            ref_paths=["/path/ref1.jpg", "/path/ref2.jpg"],
            workflow_template="default",
            style_prompt="beautiful portrait",
        )

        result = gen.generate(request)

        assert result.success is True
        assert len(result.output_paths) == 1
        assert result.error is None
        assert result.elapsed_sec > 0

        mock_comfy_client.queue_prompt.assert_called_once()
        mock_comfy_client.wait_for_completion.assert_called_once()

    def test_template_not_found(self, tmp_path):
        config = GenerationConfig(
            templates_dir=tmp_path / "empty",
            output_dir=tmp_path / "output",
        )
        gen = PortraitGenerator(config)

        request = GenerationRequest(
            person_id=0,
            ref_paths=["/path/ref.jpg"],
            workflow_template="missing",
        )

        result = gen.generate(request)

        assert result.success is False
        assert "not found" in result.error

    def test_timeout_error(self, templates_dir, tmp_path):
        config = GenerationConfig(
            templates_dir=templates_dir,
            output_dir=tmp_path / "output",
            timeout_sec=0.01,
        )
        gen = PortraitGenerator(config)

        mock_client = MagicMock()
        mock_client.queue_prompt.return_value = "test-id"
        mock_client.wait_for_completion.side_effect = TimeoutError("timeout")
        gen.client = mock_client

        request = GenerationRequest(
            person_id=0,
            ref_paths=["/path/ref.jpg"],
        )

        result = gen.generate(request)

        assert result.success is False
        assert "timeout" in result.error


class TestGenerateFromBank:
    def test_end_to_end(self, templates_dir, mock_comfy_client, tmp_path):
        # Create a fake memory bank
        from momentbank import MemoryBank, save_bank

        bank = MemoryBank(person_id=0)
        emb = np.random.RandomState(42).randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        bank.update(emb, 0.9, {"yaw": "[-5,5]", "expression": "smile"}, "/ref1.jpg")
        bank.update(emb, 0.8, {"yaw": "[-5,5]", "expression": "neutral"}, "/ref2.jpg")

        bank_path = tmp_path / "bank" / "memory_bank.json"
        save_bank(bank, bank_path)

        config = GenerationConfig(
            templates_dir=templates_dir,
            output_dir=tmp_path / "output",
        )
        gen = PortraitGenerator(config)
        gen.client = mock_comfy_client

        def fake_download(history, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "result.png"
            p.write_bytes(b"fake")
            return [p]

        mock_comfy_client.download_images.side_effect = fake_download

        result = gen.generate_from_bank(
            bank_path,
            query={"yaw": "[-5,5]"},
            style_prompt="test prompt",
        )

        assert result.success is True
        assert len(result.output_paths) == 1

    def test_empty_bank(self, templates_dir, tmp_path):
        from momentbank import MemoryBank, save_bank

        bank = MemoryBank(person_id=0)
        bank_path = tmp_path / "bank" / "memory_bank.json"
        save_bank(bank, bank_path)

        config = GenerationConfig(templates_dir=templates_dir)
        gen = PortraitGenerator(config)

        result = gen.generate_from_bank(bank_path)

        assert result.success is False
        assert "No reference images" in result.error
