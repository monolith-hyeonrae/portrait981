"""Request/response schemas for REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# -- Shared --

class GenerateOptions(BaseModel):
    """Generation style options (shared across endpoints)."""
    workflow: Optional[str] = None
    prompt: Optional[str] = None
    purpose: Optional[str] = None
    ref_images: Optional[list[str]] = None


# -- POST /portrait/scan --

class ScanRequest(BaseModel):
    member_id: int
    workflow_id: int
    video_uri: str


class ScanResponse(BaseModel):
    status: str
    member_id: int
    workflow_id: int
    frame_count: int = 0
    highlight_count: int = 0
    timing_sec: float = 0.0


# -- POST /portrait/generate --

class GenerateRequest(BaseModel):
    member_id: int
    generate: GenerateOptions = GenerateOptions()


class GenerateResponse(BaseModel):
    status: str
    member_id: int
    output_urls: list[str] = []
    ref_count: int = 0
    timing_sec: float = 0.0
    error: Optional[str] = None


# -- POST /portrait/test --

class TestRequest(BaseModel):
    video_uri: Optional[str] = None
    generate: Optional[GenerateOptions] = None


class TestResponse(BaseModel):
    status: str
    scan_result: Optional[dict] = None
    generate_result: Optional[dict] = None
    timing_sec: float = 0.0
    error: Optional[str] = None


# -- GET /portrait/status/{member_id} --

class StatusResponse(BaseModel):
    member_id: int
    frame_count: int = 0
    frames: list[dict] = []
