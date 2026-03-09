"""Mock factories for portrait981 tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest


@dataclass
class FakeScanResult:
    highlights: List[Any] = field(default_factory=lambda: [{"window_id": 0}])
    collection: Any = None
    identity: Any = None
    bank: Any = None
    frame_count: int = 100
    duration_sec: float = 10.0
    actual_backend: str = "simple"
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeGenerationResult:
    success: bool = True
    output_paths: List[str] = field(default_factory=lambda: ["/out/portrait.png"])
    workflow_used: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_sec: float = 5.0


@pytest.fixture
def mock_ms_run():
    """Mock momentscan.run() to return a fake result."""
    with patch("portrait981.pipeline.ms") as m:
        m.run.return_value = FakeScanResult()
        yield m


@pytest.fixture
def mock_lookup():
    """Mock lookup_frames to return fake frame entries."""
    with patch("portrait981.pipeline.lookup_frames") as m:
        m.return_value = [
            {"path": "/frames/f1.jpg", "pose_name": "frontal", "category": "smile", "cell_score": 0.9},
            {"path": "/frames/f2.jpg", "pose_name": "frontal", "category": "smile", "cell_score": 0.8},
        ]
        yield m


@pytest.fixture
def mock_lookup_empty():
    """Mock lookup_frames to return empty results."""
    with patch("portrait981.pipeline.lookup_frames") as m:
        m.return_value = []
        yield m


@pytest.fixture
def mock_generator():
    """Mock PortraitGenerator to return a fake result."""
    with patch("portrait981.pipeline.PortraitGenerator") as cls:
        inst = cls.return_value
        inst.generate.return_value = FakeGenerationResult()
        yield inst


@pytest.fixture
def mock_generator_fail():
    """Mock PortraitGenerator to raise an exception."""
    with patch("portrait981.pipeline.PortraitGenerator") as cls:
        inst = cls.return_value
        inst.generate.side_effect = RuntimeError("ComfyUI timeout")
        yield inst


@pytest.fixture
def mock_generator_unsuccessful():
    """Mock PortraitGenerator to return success=False."""
    with patch("portrait981.pipeline.PortraitGenerator") as cls:
        inst = cls.return_value
        inst.generate.return_value = FakeGenerationResult(
            success=False, error="Model not found: sd_xl_base", output_paths=[],
        )
        yield inst
