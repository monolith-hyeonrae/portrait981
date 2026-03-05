"""Reportrait data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GenerationConfig:
    """Configuration for portrait generation."""

    comfy_url: str = "http://127.0.0.1:8188"
    api_key: Optional[str] = None
    workflow_template: str = "default"
    templates_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    poll_interval_sec: float = 1.0
    timeout_sec: float = 300.0


@dataclass
class GenerationRequest:
    """Request for portrait generation."""

    person_id: int
    ref_paths: List[str]
    workflow_template: str = "default"
    style_prompt: str = ""
    node_ids: Optional[List[str]] = None


@dataclass
class GenerationResult:
    """Result of portrait generation."""

    success: bool
    output_paths: List[str] = field(default_factory=list)
    workflow_used: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_sec: float = 0.0
