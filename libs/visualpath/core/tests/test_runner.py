"""Tests for runner.py (process, run, get_backend)."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import Module, Observation
from visualpath.runner import (
    ProcessResult,
    get_backend,
    resolve_modules,
)
from visualpath.backends.base import PipelineResult
from visualpath.backends.simple import SimpleBackend


# Check if Pathway is available (both the pathway package and visualpath-pathway)
try:
    import pathway as pw
    from visualpath.backends.pathway import PathwayBackend as _PB
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================


class CountingAnalyzer(Module):
    """Analyzer that counts calls."""

    def __init__(self, name: str, value: float = 0.5):
        self._name = name
        self._value = value
        self._process_count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


# =============================================================================
# ProcessResult Tests
# =============================================================================


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_default_values(self):
        result = ProcessResult()
        assert result.triggers == []
        assert result.frame_count == 0
        assert result.duration_sec == 0.0

    def test_with_values(self):
        result = ProcessResult(triggers=["t1"], frame_count=10, duration_sec=1.5)
        assert result.triggers == ["t1"]
        assert result.frame_count == 10
        assert result.duration_sec == 1.5


# =============================================================================
# get_backend Tests
# =============================================================================


class TestGetBackend:
    """Tests for get_backend factory."""

    def test_simple_backend(self):
        backend = get_backend("simple")
        assert isinstance(backend, SimpleBackend)

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown")

    @pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
    def test_pathway_backend(self):
        from visualpath.backends.pathway import PathwayBackend

        backend = get_backend("pathway")
        assert isinstance(backend, PathwayBackend)


# =============================================================================
# resolve_modules Tests
# =============================================================================


class TestResolveModules:
    """Tests for resolve_modules."""

    def test_resolve_instance(self):
        ext = CountingAnalyzer("test")
        result = resolve_modules([ext])
        assert result == [ext]

    def test_resolve_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown module"):
            resolve_modules(["nonexistent_module"])

    def test_resolve_registered_name(self):
        import visualpath as vp
        from visualpath.api import _analyzer_registry

        # Register an analyzer
        @vp.analyzer("test_resolve_ext")
        def my_ext(frame):
            return {"x": 1.0}

        try:
            result = resolve_modules(["test_resolve_ext"])
            assert len(result) == 1
            assert result[0].name == "test_resolve_ext"
        finally:
            _analyzer_registry.pop("test_resolve_ext", None)

    def test_resolve_mixed(self):
        import visualpath as vp
        from visualpath.api import _analyzer_registry

        @vp.analyzer("test_resolve_mix")
        def my_ext(frame):
            return {"x": 1.0}

        instance = CountingAnalyzer("inst")

        try:
            result = resolve_modules(["test_resolve_mix", instance])
            assert len(result) == 2
        finally:
            _analyzer_registry.pop("test_resolve_mix", None)

    def test_resolve_fusion_registered_name(self):
        import visualpath as vp
        from visualpath.api import _fusion_registry

        @vp.fusion(sources=["a"], name="test_resolve_fus")
        def my_fus(a):
            pass

        try:
            result = resolve_modules(["test_resolve_fus"])
            assert len(result) == 1
            assert result[0].name == "test_resolve_fus"
        finally:
            _fusion_registry.pop("test_resolve_fus", None)


# =============================================================================
# Public API via visualpath namespace
# =============================================================================


class TestPublicAPI:
    """Tests that process/run are accessible from visualpath namespace."""

    def test_process_video_importable(self):
        import visualpath as vp
        assert hasattr(vp, "process_video")
        assert callable(vp.process_video)

    def test_process_video_importable_as_main_api(self):
        import visualpath as vp
        assert hasattr(vp, "process_video")
        assert callable(vp.process_video)

    def test_process_result_importable(self):
        import visualpath as vp
        assert hasattr(vp, "ProcessResult")
        result = vp.ProcessResult()
        assert result.triggers == []
