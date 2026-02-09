"""Tests for Phase 2: Interface stabilization (warmup, PortSchema, ErrorPolicy)."""

import pytest
from unittest.mock import MagicMock, patch

from visualpath.core.module import Module
from visualpath.core.capabilities import PortSchema
from visualpath.core.error_policy import ErrorPolicy
from visualpath.core.observation import Observation


class TestWarmup:
    """Test warmup() lifecycle hook."""

    def test_warmup_default_noop(self):
        """Default warmup() does nothing."""

        class SimpleModule(Module):
            @property
            def name(self):
                return "test"

            def process(self, frame, deps=None):
                return None

        mod = SimpleModule()
        mod.warmup()  # Should not raise

    def test_warmup_with_sample_frame(self):
        """warmup() can accept a sample frame."""

        class GpuModule(Module):
            warmup_called = False
            warmup_frame = None

            @property
            def name(self):
                return "gpu"

            def warmup(self, sample_frame=None):
                self.warmup_called = True
                self.warmup_frame = sample_frame

            def process(self, frame, deps=None):
                return None

        mod = GpuModule()
        mock_frame = MagicMock()
        mod.warmup(sample_frame=mock_frame)
        assert mod.warmup_called
        assert mod.warmup_frame is mock_frame

    def test_warmup_called_by_path_node(self):
        """PathNode.initialize() calls warmup() on all modules."""
        from visualpath.flow.nodes.path import PathNode

        class TrackedModule(Module):
            init_called = False
            warmup_called = False

            @property
            def name(self):
                return "tracked"

            def initialize(self):
                self.init_called = True

            def warmup(self, sample_frame=None):
                self.warmup_called = True

            def process(self, frame, deps=None):
                return None

        mod = TrackedModule()
        node = PathNode(name="test", modules=[mod])
        node.initialize()

        assert mod.init_called
        assert mod.warmup_called

    def test_lifecycle_order(self):
        """Lifecycle order: initialize -> warmup -> [process] -> cleanup."""

        class OrderTracker(Module):
            calls = []

            @property
            def name(self):
                return "order"

            def initialize(self):
                self.calls.append("init")

            def warmup(self, sample_frame=None):
                self.calls.append("warmup")

            def process(self, frame, deps=None):
                self.calls.append("process")
                return None

            def cleanup(self):
                self.calls.append("cleanup")

        mod = OrderTracker()
        mod.calls = []  # Reset

        with mod:  # __enter__ calls initialize()
            mod.warmup()
            mod.process(MagicMock())
        # __exit__ calls cleanup()

        assert mod.calls == ["init", "warmup", "process", "cleanup"]


class TestPortSchema:
    """Test PortSchema."""

    def test_defaults(self):
        schema = PortSchema()
        assert schema.version == "1.0"
        assert schema.input_signals == frozenset()
        assert schema.output_signals == frozenset()
        assert schema.output_data_type == ""

    def test_frozen(self):
        schema = PortSchema(version="1.0")
        with pytest.raises(AttributeError):
            schema.version = "2.0"  # type: ignore[misc]

    def test_module_port_schema_default_none(self):
        class SimpleModule(Module):
            @property
            def name(self):
                return "test"

            def process(self, frame, deps=None):
                return None

        assert SimpleModule.port_schema is None

    def test_module_port_schema_declared(self):
        class TypedModule(Module):
            port_schema = PortSchema(
                input_signals=frozenset({"face_count"}),
                output_signals=frozenset({"expression_score"}),
                output_data_type="vpx.face_expression.output.ExpressionOutput",
            )

            @property
            def name(self):
                return "typed"

            def process(self, frame, deps=None):
                return None

        schema = TypedModule.port_schema
        assert "face_count" in schema.input_signals
        assert "expression_score" in schema.output_signals


class TestErrorPolicy:
    """Test ErrorPolicy."""

    def test_defaults(self):
        policy = ErrorPolicy()
        assert policy.max_retries == 0
        assert policy.timeout_sec == 0.0
        assert policy.on_timeout == "skip"
        assert policy.on_error == "skip"
        assert policy.fallback_signals == {}

    def test_frozen(self):
        policy = ErrorPolicy()
        with pytest.raises(AttributeError):
            policy.max_retries = 5  # type: ignore[misc]

    def test_module_error_policy_default_none(self):
        class SimpleModule(Module):
            @property
            def name(self):
                return "test"

            def process(self, frame, deps=None):
                return None

        assert SimpleModule.error_policy is None

    def test_module_error_policy_declared(self):
        class ResilientModule(Module):
            error_policy = ErrorPolicy(
                max_retries=2,
                on_error="fallback",
                fallback_signals={"score": 0.0},
            )

            @property
            def name(self):
                return "resilient"

            def process(self, frame, deps=None):
                return None

        policy = ResilientModule.error_policy
        assert policy.max_retries == 2
        assert policy.on_error == "fallback"


class TestErrorPolicyInInterpreter:
    """Test ErrorPolicy integration in SimpleInterpreter."""

    def _make_failing_module(self, name, fail_count, policy):
        """Create a module that fails `fail_count` times then succeeds."""
        call_count = [0]

        class _Mod(Module):
            error_policy = policy

            @property
            def name(self_):
                return name

            def process(self_, frame, deps=None):
                call_count[0] += 1
                if call_count[0] <= fail_count:
                    raise RuntimeError(f"Fail #{call_count[0]}")
                return Observation(
                    source=name,
                    frame_id=getattr(frame, 'frame_id', 0),
                    t_ns=getattr(frame, 't_src_ns', 0),
                    signals={"ok": 1.0},
                )

        return _Mod(), call_count

    def _make_path_node(self, mod):
        from visualpath.flow.nodes.path import PathNode
        return PathNode(name="n", modules=[mod])

    def test_retry_success(self):
        """Module succeeds after retries."""
        from visualpath.flow.interpreter import SimpleInterpreter
        from visualpath.flow.node import FlowData

        policy = ErrorPolicy(max_retries=2, on_error="skip")
        mod, call_count = self._make_failing_module("retry_test", 1, policy)

        interpreter = SimpleInterpreter()
        node = self._make_path_node(mod)
        frame = MagicMock()
        frame.frame_id = 1
        frame.t_src_ns = 0
        data = FlowData(frame=frame)

        outputs = interpreter.interpret(node, data)
        assert len(outputs) == 1
        assert call_count[0] == 2  # Failed once, succeeded on retry

    def test_all_retries_fail_skip(self):
        """Module fails all retries with skip policy -> None output."""
        from visualpath.flow.interpreter import SimpleInterpreter
        from visualpath.flow.node import FlowData

        policy = ErrorPolicy(max_retries=1, on_error="skip")
        mod, call_count = self._make_failing_module("skip_test", 10, policy)

        interpreter = SimpleInterpreter()
        node = self._make_path_node(mod)
        frame = MagicMock()
        frame.frame_id = 1
        frame.t_src_ns = 0
        data = FlowData(frame=frame)

        outputs = interpreter.interpret(node, data)
        assert len(outputs) == 1  # FlowData still returned, but no observations added
        assert call_count[0] == 2  # Original + 1 retry

    def test_all_retries_fail_raise(self):
        """Module fails all retries with raise policy -> exception."""
        from visualpath.flow.interpreter import SimpleInterpreter
        from visualpath.flow.node import FlowData

        policy = ErrorPolicy(max_retries=0, on_error="raise")
        mod, _ = self._make_failing_module("raise_test", 10, policy)

        interpreter = SimpleInterpreter()
        node = self._make_path_node(mod)
        frame = MagicMock()
        frame.frame_id = 1
        frame.t_src_ns = 0
        data = FlowData(frame=frame)

        with pytest.raises(RuntimeError, match="Fail #1"):
            interpreter.interpret(node, data)

    def test_fallback_policy(self):
        """Module fails with fallback policy -> returns fallback signals."""
        from visualpath.flow.interpreter import SimpleInterpreter
        from visualpath.flow.node import FlowData

        policy = ErrorPolicy(
            max_retries=0,
            on_error="fallback",
            fallback_signals={"score": 0.0, "error": 1.0},
        )
        mod, _ = self._make_failing_module("fb_test", 10, policy)

        interpreter = SimpleInterpreter()
        node = self._make_path_node(mod)
        frame = MagicMock()
        frame.frame_id = 42
        frame.t_src_ns = 100
        data = FlowData(frame=frame)

        outputs = interpreter.interpret(node, data)
        assert len(outputs) == 1
        obs_list = outputs[0].observations
        assert len(obs_list) == 1
        assert obs_list[0].signals["score"] == 0.0
        assert obs_list[0].signals["error"] == 1.0
        assert obs_list[0].source == "fb_test"
