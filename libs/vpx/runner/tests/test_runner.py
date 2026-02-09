"""Tests for LiteRunner."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from vpx.sdk.testing import FakeFrame
from vpx.sdk.observation import Observation


@dataclass
class _MockModule:
    """Mock module for runner tests."""
    _name: str
    depends: List[str] = field(default_factory=list)
    optional_depends: List[str] = field(default_factory=list)
    _initialized: bool = False
    _cleaned: bool = False
    _process_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    def initialize(self):
        self._initialized = True

    def cleanup(self):
        self._cleaned = True

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._process_count += 1
        return Observation(
            source=self._name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"count": self._process_count},
        )


class TestLiteRunner:
    def test_run_with_fake_frames(self):
        from vpx.runner import LiteRunner

        mod = _MockModule("mock.test")
        runner = LiteRunner(mod)
        frames = FakeFrame.sequence(5)
        result = runner.run(frames)

        assert result.frame_count == 5
        assert result.module_names == ["mock.test"]
        assert len(result.frames) == 5
        assert "mock.test" in result.frames[0]

    def test_run_multiple_modules(self):
        from vpx.runner import LiteRunner

        a = _MockModule("a")
        b = _MockModule("b", depends=["a"])
        runner = LiteRunner([b, a])
        frames = FakeFrame.sequence(3)
        result = runner.run(frames)

        assert result.frame_count == 3
        # Toposorted: a before b
        assert result.module_names == ["a", "b"]

    def test_dependency_passing(self):
        """Dependencies are passed to downstream modules."""
        from vpx.runner import LiteRunner

        received_deps = []

        @dataclass
        class DepCapture:
            _name: str
            depends: List[str] = field(default_factory=lambda: ["upstream"])
            optional_depends: List[str] = field(default_factory=list)

            @property
            def name(self) -> str:
                return self._name

            def initialize(self): pass
            def cleanup(self): pass

            def process(self, frame, deps=None):
                received_deps.append(deps)
                return Observation(
                    source=self._name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={},
                )

        upstream = _MockModule("upstream")
        downstream = DepCapture("downstream")
        runner = LiteRunner([downstream, upstream])
        frames = FakeFrame.sequence(1)
        runner.run(frames)

        assert len(received_deps) == 1
        assert "upstream" in received_deps[0]

    def test_max_frames(self):
        from vpx.runner import LiteRunner

        mod = _MockModule("mock.test")
        runner = LiteRunner(mod)
        frames = FakeFrame.sequence(10)
        result = runner.run(frames, max_frames=3)

        assert result.frame_count == 3

    def test_on_observation_callback(self):
        from vpx.runner import LiteRunner

        received = []
        mod = _MockModule("mock.test")
        runner = LiteRunner(mod, on_observation=lambda n, o: received.append((n, o)))
        frames = FakeFrame.sequence(2)
        runner.run(frames)

        assert len(received) == 2
        assert received[0][0] == "mock.test"

    def test_on_frame_callback(self):
        from vpx.runner import LiteRunner

        received = []
        mod = _MockModule("mock.test")
        runner = LiteRunner(mod, on_frame=lambda f, obs: received.append(obs))
        frames = FakeFrame.sequence(3)
        runner.run(frames)

        assert len(received) == 3
        assert "mock.test" in received[0]

    def test_initialize_and_cleanup(self):
        from vpx.runner import LiteRunner

        mod = _MockModule("mock.test")
        runner = LiteRunner(mod)
        frames = FakeFrame.sequence(1)
        runner.run(frames)

        assert mod._initialized
        assert mod._cleaned

    def test_cleanup_on_error(self):
        """Cleanup runs even if processing raises."""
        from vpx.runner import LiteRunner

        @dataclass
        class FailModule:
            _name: str = "fail"
            depends: List[str] = field(default_factory=list)
            optional_depends: List[str] = field(default_factory=list)
            _cleaned: bool = False

            @property
            def name(self) -> str:
                return self._name

            def initialize(self): pass

            def cleanup(self):
                self._cleaned = True

            def process(self, frame, deps=None):
                raise RuntimeError("boom")

        mod = FailModule()
        runner = LiteRunner(mod)
        frames = FakeFrame.sequence(1)

        with pytest.raises(RuntimeError, match="boom"):
            runner.run(frames)

        assert mod._cleaned

    def test_run_callback_override(self):
        """Per-run callbacks override constructor callbacks."""
        from vpx.runner import LiteRunner

        ctor_received = []
        run_received = []

        mod = _MockModule("mock.test")
        runner = LiteRunner(
            mod,
            on_observation=lambda n, o: ctor_received.append(n),
        )
        runner.run(
            FakeFrame.sequence(1),
            on_observation=lambda n, o: run_received.append(n),
        )

        assert len(ctor_received) == 0
        assert len(run_received) == 1

    def test_module_returning_none(self):
        """Modules that return None are skipped in deps."""
        from vpx.runner import LiteRunner

        @dataclass
        class NoneModule:
            _name: str = "none.mod"
            depends: List[str] = field(default_factory=list)
            optional_depends: List[str] = field(default_factory=list)

            @property
            def name(self) -> str:
                return self._name

            def initialize(self): pass
            def cleanup(self): pass

            def process(self, frame, deps=None):
                return None

        mod = NoneModule()
        runner = LiteRunner(mod)
        result = runner.run(FakeFrame.sequence(2))

        assert result.frame_count == 2
        # No observations since module returns None
        assert result.frames[0] == {}

    def test_string_analyzer_resolution(self):
        """String analyzer names are resolved via entry points."""
        from vpx.runner.runner import LiteRunner

        mock_cls = MagicMock()
        mock_instance = _MockModule("mock.dummy")
        mock_cls.return_value = mock_instance

        with patch("vpx.runner.runner.load_analyzer", return_value=mock_cls) as mock_load:
            runner = LiteRunner("mock.dummy")
            result = runner.run(FakeFrame.sequence(1))

        mock_load.assert_called_once_with("mock.dummy")
        assert result.frame_count == 1
