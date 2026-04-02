"""Tests for REST API routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from portrait981.types import JobResult, JobSpec, JobStatus, StepTiming
from portrait981_serve.app import create_app
from portrait981_serve.config import ServeConfig


@pytest.fixture
def config():
    return ServeConfig(
        default_workflow="test-wf",
        default_prompt="test prompt",
        comfy_urls=["http://comfy:8188"],
        s3_output_bucket="test-bucket",
        s3_access_key="FAKE",
        s3_secret_key="FAKE",
    )


def _make_frame_result(is_shoot=True):
    r = MagicMock()
    r.is_shoot = is_shoot
    return r


def _make_scan_result(frame_count=100, shoot_count=3):
    """Return list[FrameResult]-like for v2 pipeline."""
    results = [_make_frame_result(is_shoot=True) for _ in range(shoot_count)]
    results += [_make_frame_result(is_shoot=False) for _ in range(frame_count - shoot_count)]
    return results


def _make_gen_result(success=True, output_paths=None):
    r = MagicMock()
    r.success = success
    r.output_paths = output_paths or []
    return r


class TestScanEndpoint:
    @patch("portrait981_serve.routes.Portrait981Pipeline")
    @patch("portrait981_serve.s3.boto3")
    def test_scan_success(self, mock_boto3, mock_cls, config):
        inst = mock_cls.return_value
        inst.run_one.return_value = JobResult(
            job=JobSpec(video_path="/tmp/v.mp4", member_id="1423", scan_only=True),
            status=JobStatus.DONE,
            scan_result=_make_scan_result(200, 5),
            timing=StepTiming(scan_sec=12.3, total_sec=12.5),
        )

        app = create_app(config)
        client = TestClient(app)
        resp = client.post("/portrait/scan", json={
            "member_id": 1423,
            "workflow_id": 20260309001423,
            "video_uri": "s3://bucket/video.mp4",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["member_id"] == 1423
        assert data["frame_count"] == 200
        assert data["shoot_count"] == 5


class TestGenerateEndpoint:
    @patch("portrait981_serve.routes.Portrait981Pipeline")
    @patch("portrait981_serve.s3.boto3")
    def test_generate_success(self, mock_boto3, mock_cls, config):
        inst = mock_cls.return_value
        inst.run_one.return_value = JobResult(
            job=JobSpec(member_id="1423", generate_only=True),
            status=JobStatus.DONE,
            generation_result=_make_gen_result(success=True, output_paths=[]),
            ref_count=3,
            timing=StepTiming(generate_sec=60.0, total_sec=61.0),
        )

        app = create_app(config)
        client = TestClient(app)
        resp = client.post("/portrait/generate", json={
            "member_id": 1423,
            "generate": {"workflow": "spring-2026"},
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["ref_count"] == 3

    @patch("portrait981_serve.routes.Portrait981Pipeline")
    @patch("portrait981_serve.s3.boto3")
    def test_generate_uses_server_defaults(self, mock_boto3, mock_cls, config):
        inst = mock_cls.return_value
        inst.run_one.return_value = JobResult(
            job=JobSpec(member_id="1423", generate_only=True),
            status=JobStatus.DONE,
            timing=StepTiming(),
        )

        app = create_app(config)
        client = TestClient(app)
        resp = client.post("/portrait/generate", json={
            "member_id": 1423,
        })

        assert resp.status_code == 200
        # Verify pipeline was called with server defaults
        call_args = inst.run_one.call_args
        job = call_args[0][0]
        assert job.workflow == "test-wf"
        assert job.prompt == "test prompt"


class TestTestEndpoint:
    @patch("portrait981_serve.routes.Portrait981Pipeline")
    @patch("portrait981_serve.s3.boto3")
    def test_scan_only_test(self, mock_boto3, mock_cls, config):
        inst = mock_cls.return_value
        inst.run_one.return_value = JobResult(
            job=JobSpec(video_path="/tmp/v.mp4", member_id="__test__", scan_only=True),
            status=JobStatus.DONE,
            scan_result=_make_scan_result(50, 2),
            timing=StepTiming(total_sec=5.0),
        )

        app = create_app(config)
        client = TestClient(app)
        resp = client.post("/portrait/test", json={
            "video_uri": "s3://bucket/test.mp4",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["scan_result"]["frame_count"] == 50


class TestStatusEndpoint:
    @patch("portrait981_serve.routes.PersonMemory")
    @patch("portrait981_serve.routes.Portrait981Pipeline")
    @patch("portrait981_serve.s3.boto3")
    def test_status(self, mock_boto3, mock_cls, mock_mem_cls, config):
        mock_profile = MagicMock()
        mock_profile.n_total_frames = 12
        mock_mem_cls.return_value.profile.return_value = mock_profile

        app = create_app(config)
        client = TestClient(app)
        resp = client.get("/portrait/status/1423")

        assert resp.status_code == 200
        data = resp.json()
        assert data["member_id"] == 1423
        assert data["frame_count"] == 12


class TestHealthEndpoint:
    @patch("portrait981_serve.routes.Portrait981Pipeline")
    @patch("portrait981_serve.s3.boto3")
    def test_health(self, mock_boto3, mock_cls, config):
        app = create_app(config)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
