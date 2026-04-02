"""REST API routes for portrait981-serve."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from portrait981 import JobSpec, PipelineConfig, Portrait981Pipeline
from portrait981.types import JobStatus
from personmemory import PersonMemory

from portrait981_serve.config import ServeConfig
from portrait981_serve.s3 import S3Client
from portrait981_serve.schemas import (
    GenerateRequest,
    GenerateResponse,
    ScanRequest,
    ScanResponse,
    StatusResponse,
    TestRequest,
    TestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portrait")

# Shared state — initialized by init_routes()
_config: ServeConfig
_s3: S3Client
_pipeline: Portrait981Pipeline


def init_routes(config: ServeConfig) -> None:
    """Wire dependencies into route handlers."""
    global _config, _s3, _pipeline
    _config = config
    _s3 = S3Client(config)
    _pipeline = Portrait981Pipeline(
        config=PipelineConfig(
            comfy_urls=config.comfy_urls,
            api_key=config.comfy_api_key,
            scan_fps=config.scan_fps,
            scan_backend=config.scan_backend,
        ),
    )


def _resolve_workflow(override: Optional[str]) -> str:
    """Event override > server default."""
    return override or _config.default_workflow


def _resolve_prompt(override: Optional[str]) -> str:
    return override or _config.default_prompt


def _extract_scan_counts(scan_result) -> tuple[int, int]:
    """Extract frame_count and shoot_count from scan result (v2: list[FrameResult])."""
    if isinstance(scan_result, list):
        frame_count = len(scan_result)
        shoot_count = sum(1 for r in scan_result if getattr(r, "is_shoot", False))
    else:
        frame_count = getattr(scan_result, "frame_count", 0)
        shoot_count = 0
    return frame_count, shoot_count


# -- POST /portrait/scan --

@router.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    """Scan a video and optionally ingest SHOOT frames into personmemory."""
    local_path: Optional[Path] = None
    try:
        local_path = _s3.download(req.video_uri)

        result = _pipeline.run_one(JobSpec(
            video_path=str(local_path),
            member_id=str(req.member_id),
            scan_only=True,
            ingest=req.ingest,
        ))

        frame_count, shoot_count = _extract_scan_counts(result.scan_result)

        return ScanResponse(
            status=result.status.value,
            member_id=req.member_id,
            workflow_id=req.workflow_id,
            frame_count=frame_count,
            shoot_count=shoot_count,
            timing_sec=result.timing.total_sec,
        )
    except Exception as e:
        logger.exception("Scan failed for member_id=%s", req.member_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if local_path:
            _s3.cleanup(local_path)


# -- POST /portrait/generate --

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate portraits from existing personmemory data."""
    result = _pipeline.run_one(JobSpec(
        member_id=str(req.member_id),
        workflow=_resolve_workflow(req.generate.workflow),
        prompt=_resolve_prompt(req.generate.prompt),
        generate_only=True,
    ))

    # Upload outputs to S3
    output_urls: list[str] = []
    out_paths = getattr(result.generation_result, "output_paths", [])
    for path in out_paths:
        local = Path(path)
        if local.exists():
            prefix = _config.s3_output_prefix
            key = f"{prefix}{req.member_id}/{local.name}"
            s3_url = _s3.upload(local, key=key)
            output_urls.append(s3_url)

    return GenerateResponse(
        status=result.status.value,
        member_id=req.member_id,
        output_urls=output_urls,
        ref_count=result.ref_count,
        timing_sec=result.timing.total_sec,
        error=result.error,
    )


# -- POST /portrait/test --

@router.post("/test", response_model=TestResponse)
def test_pipeline(req: TestRequest):
    """Pipeline verification — does NOT persist to personmemory."""
    local_path: Optional[Path] = None
    try:
        scan_only = req.generate is None
        generate_only = req.video_uri is None

        if req.video_uri:
            local_path = _s3.download(req.video_uri)

        job = JobSpec(
            video_path=str(local_path) if local_path else "",
            member_id="__test__",
            workflow=_resolve_workflow(
                req.generate.workflow if req.generate else None
            ),
            prompt=_resolve_prompt(
                req.generate.prompt if req.generate else None
            ),
            scan_only=scan_only,
            generate_only=generate_only,
            output_dir=tempfile.mkdtemp(prefix="p981_test_"),
        )
        result = _pipeline.run_one(job)

        scan_info = None
        if result.scan_result:
            frame_count, shoot_count = _extract_scan_counts(result.scan_result)
            scan_info = {
                "frame_count": frame_count,
                "shoot_count": shoot_count,
            }

        gen_info = None
        if result.generation_result:
            gen_info = {
                "success": getattr(result.generation_result, "success", False),
                "output_count": len(
                    getattr(result.generation_result, "output_paths", [])
                ),
            }

        return TestResponse(
            status=result.status.value,
            scan_result=scan_info,
            generate_result=gen_info,
            timing_sec=result.timing.total_sec,
            error=result.error,
        )
    except Exception as e:
        logger.exception("Test pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if local_path:
            _s3.cleanup(local_path)


# -- GET /portrait/status/{member_id} --

@router.get("/status/{member_id}", response_model=StatusResponse)
def status(member_id: int):
    """Query personmemory status for a member."""
    try:
        mem = PersonMemory(str(member_id))
        profile = mem.profile()
        return StatusResponse(
            member_id=member_id,
            frame_count=profile.n_total_frames,
            frames=[],
        )
    except Exception as e:
        logger.exception("Status lookup failed for member_id=%s", member_id)
        raise HTTPException(status_code=500, detail=str(e))
