"""Tests for vpx-sdk package."""

import numpy as np
import pytest

from vpx.sdk import (
    Module,
    Observation,
    ProcessingStep,
    processing_step,
    get_processing_steps,
    Capability,
    ModuleCapabilities,
    PortSchema,
    ErrorPolicy,
    check_compatibility,
    CompatibilityReport,
)
from vpx.sdk.testing import (
    PluginTestHarness,
    PluginCheckReport,
    FakeFrame,
    assert_valid_observation,
)


class TestSDKImports:
    """Verify all re-exports from vpx.sdk work."""

    def test_module(self):
        assert Module is not None

    def test_observation(self):
        assert Observation is not None

    def test_capability(self):
        assert Capability.GPU is not None

    def test_module_capabilities(self):
        caps = ModuleCapabilities()
        assert caps.flags == Capability.NONE

    def test_port_schema(self):
        schema = PortSchema()
        assert schema.version == "1.0"

    def test_error_policy(self):
        policy = ErrorPolicy()
        assert policy.max_retries == 0

    def test_check_compatibility(self):
        report = check_compatibility([])
        assert isinstance(report, CompatibilityReport)

    def test_processing_step_decorator(self):
        assert callable(processing_step)

    def test_processing_step_class(self):
        step = ProcessingStep(name="test", description="A test step")
        assert step.name == "test"

    def test_get_processing_steps(self):
        assert callable(get_processing_steps)


class TestPluginTestHarness:
    """Test PluginTestHarness."""

    def _make_good_module(self):
        class GoodModule(Module):
            @property
            def name(self):
                return "test.good"

            @property
            def capabilities(self):
                return ModuleCapabilities(
                    flags=Capability.DETERMINISTIC,
                )

            def process(self, frame, deps=None):
                return None

        return GoodModule()

    def _make_bad_name_module(self):
        class BadName(Module):
            @property
            def name(self):
                return "badname"  # no dot

            def process(self, frame, deps=None):
                return None

        return BadName()

    def test_good_module(self):
        harness = PluginTestHarness()
        report = harness.check_module(self._make_good_module())
        assert report.valid is True
        assert report.errors == []
        assert report.module_name == "test.good"

    def test_bad_name_warning(self):
        harness = PluginTestHarness()
        report = harness.check_module(self._make_bad_name_module())
        assert report.valid is True  # warnings don't fail
        assert any("dot notation" in w for w in report.warnings)

    def test_empty_name_error(self):
        class EmptyName(Module):
            @property
            def name(self):
                return ""

            def process(self, frame, deps=None):
                return None

        harness = PluginTestHarness()
        report = harness.check_module(EmptyName())
        assert report.valid is False
        assert any("empty" in e for e in report.errors)

    def test_plugin_check_report_defaults(self):
        r = PluginCheckReport()
        assert r.valid is True
        assert r.warnings == []
        assert r.errors == []
        assert r.module_name == ""

    def test_module_has_analyze_alias(self):
        """Module from vpx.sdk should have the analyze() backward-compat alias."""
        class TestMod(Module):
            @property
            def name(self):
                return "test.alias"

            def process(self, frame, deps=None):
                return "result"

        mod = TestMod()
        assert mod.analyze(None) == "result"

    def test_observation_has_trigger_helpers(self):
        """Observation from vpx.sdk should have trigger helper properties."""
        obs = Observation(source="test", frame_id=0, t_ns=0)
        assert obs.should_trigger is False
        assert obs.trigger_score == 0.0


class TestFakeFrame:
    """Test FakeFrame factory."""

    def test_create_defaults(self):
        frame = FakeFrame.create()
        assert frame.frame_id == 0
        assert frame.t_src_ns == 0
        assert frame.width == 640
        assert frame.height == 480
        assert frame.data.shape == (480, 640, 3)
        assert frame.data.dtype == np.uint8

    def test_create_custom(self):
        frame = FakeFrame.create(320, 240, frame_id=7, t_src_ns=999)
        assert frame.width == 320
        assert frame.height == 240
        assert frame.frame_id == 7
        assert frame.t_src_ns == 999
        assert frame.data.shape == (240, 320, 3)

    def test_sequence(self):
        frames = FakeFrame.sequence(4, interval_ns=100_000_000)
        assert len(frames) == 4
        assert [f.frame_id for f in frames] == [0, 1, 2, 3]
        assert [f.t_src_ns for f in frames] == [0, 100_000_000, 200_000_000, 300_000_000]

    def test_sequence_empty(self):
        assert FakeFrame.sequence(0) == []

    def test_duck_type_compatible(self):
        """FakeFrame has the same attributes Module.process() uses."""
        frame = FakeFrame.create()
        # These are the three attributes every analyzer accesses
        assert hasattr(frame, 'data')
        assert hasattr(frame, 'frame_id')
        assert hasattr(frame, 't_src_ns')


class TestAssertValidObservation:
    """Test assert_valid_observation helper."""

    def _good_obs(self, **overrides):
        defaults = dict(source="test.mod", frame_id=0, t_ns=0)
        defaults.update(overrides)
        return Observation(**defaults)

    def test_valid_minimal(self):
        assert_valid_observation(self._good_obs())

    def test_valid_with_all_fields(self):
        obs = self._good_obs(
            signals={"score": 0.9},
            data={"faces": []},
            metadata={"backend": "test"},
            timing={"detect_ms": 12.3, "track_ms": 1},
        )
        assert_valid_observation(obs)

    def test_not_observation(self):
        with pytest.raises(AssertionError, match="Expected Observation"):
            assert_valid_observation({"source": "fake"})

    def test_empty_source(self):
        with pytest.raises(AssertionError, match="non-empty string"):
            assert_valid_observation(Observation(source="", frame_id=0, t_ns=0))

    def test_module_name_match(self):
        class Mod(Module):
            @property
            def name(self):
                return "test.mod"

            def process(self, frame, deps=None):
                return None

        assert_valid_observation(self._good_obs(source="test.mod"), module=Mod())

    def test_module_name_mismatch(self):
        class Mod(Module):
            @property
            def name(self):
                return "test.other"

            def process(self, frame, deps=None):
                return None

        with pytest.raises(AssertionError, match="test.mod.*test.other"):
            assert_valid_observation(self._good_obs(source="test.mod"), module=Mod())

    def test_require_data_present(self):
        assert_valid_observation(self._good_obs(data=[1, 2]), require_data=True)

    def test_require_data_missing(self):
        with pytest.raises(AssertionError, match="require_data"):
            assert_valid_observation(self._good_obs(), require_data=True)

    def test_require_timing_present(self):
        assert_valid_observation(
            self._good_obs(timing={"step_ms": 5.0}), require_timing=True
        )

    def test_require_timing_missing(self):
        with pytest.raises(AssertionError, match="require_timing"):
            assert_valid_observation(self._good_obs(), require_timing=True)

    def test_timing_bad_key_type(self):
        obs = self._good_obs(timing={42: 1.0})
        with pytest.raises(AssertionError, match="timing key must be str"):
            assert_valid_observation(obs)

    def test_timing_bad_value_type(self):
        obs = self._good_obs(timing={"step": "slow"})
        with pytest.raises(AssertionError, match="must be numeric"):
            assert_valid_observation(obs)

    def test_roundtrip_with_module(self):
        """End-to-end: module.process(FakeFrame) -> assert_valid_observation."""

        class EchoModule(Module):
            @property
            def name(self):
                return "test.echo"

            def process(self, frame, deps=None):
                return Observation(
                    source=self.name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"pixel_sum": float(frame.data.sum())},
                )

        mod = EchoModule()
        frame = FakeFrame.create(frame_id=3, t_src_ns=42_000)
        obs = mod.process(frame)
        assert_valid_observation(obs, module=mod)
        assert obs.frame_id == 3
        assert obs.t_ns == 42_000
