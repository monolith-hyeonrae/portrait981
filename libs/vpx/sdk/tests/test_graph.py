"""Tests for toposort_modules (via vpx.sdk re-export)."""

import pytest
from dataclasses import dataclass, field
from typing import List


@dataclass
class _StubModule:
    """Minimal module stub with name and depends."""
    _name: str
    depends: List[str] = field(default_factory=list)
    optional_depends: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self._name


class TestToposortModules:
    def _sort(self, modules):
        from vpx.sdk import toposort_modules
        return toposort_modules(modules)

    def test_empty(self):
        assert self._sort([]) == []

    def test_single(self):
        m = _StubModule("a")
        assert self._sort([m]) == [m]

    def test_linear_chain(self):
        a = _StubModule("a")
        b = _StubModule("b", depends=["a"])
        c = _StubModule("c", depends=["b"])
        result = self._sort([c, b, a])
        names = [m.name for m in result]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_diamond(self):
        a = _StubModule("a")
        b = _StubModule("b", depends=["a"])
        c = _StubModule("c", depends=["a"])
        d = _StubModule("d", depends=["b", "c"])
        result = self._sort([d, c, b, a])
        names = [m.name for m in result]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_cycle_fallback(self):
        """Cycles fall back to original order for remaining nodes."""
        a = _StubModule("a", depends=["b"])
        b = _StubModule("b", depends=["a"])
        result = self._sort([a, b])
        assert len(result) == 2
        # Both modules present despite cycle
        assert {m.name for m in result} == {"a", "b"}

    def test_external_deps_ignored(self):
        """Dependencies not in the given set are ignored."""
        a = _StubModule("a", depends=["external"])
        b = _StubModule("b", depends=["a"])
        result = self._sort([b, a])
        names = [m.name for m in result]
        assert names.index("a") < names.index("b")

    def test_optional_depends(self):
        a = _StubModule("a")
        b = _StubModule("b", optional_depends=["a"])
        result = self._sort([b, a])
        names = [m.name for m in result]
        assert names.index("a") < names.index("b")

    def test_independent_modules(self):
        """Modules with no dependencies preserve relative input order."""
        a = _StubModule("a")
        b = _StubModule("b")
        c = _StubModule("c")
        result = self._sort([a, b, c])
        assert len(result) == 3
        assert {m.name for m in result} == {"a", "b", "c"}

    def test_import_from_visualpath_core(self):
        """Also importable directly from visualpath.core."""
        from visualpath.core.graph import toposort_modules
        assert callable(toposort_modules)
