"""Testing utilities for vpx plugin developers.

PluginTestHarness validates that a Module implementation follows
the visualpath plugin conventions.

FakeFrame provides a lightweight stand-in for visualbase.Frame
without requiring a visualbase dependency in tests.

Example:
    >>> from vpx.sdk.testing import PluginTestHarness, FakeFrame
    >>> harness = PluginTestHarness()
    >>> report = harness.check_module(my_module)
    >>> assert report.valid, report.errors

    >>> frame = FakeFrame.create(640, 480)
    >>> obs = my_module.process(frame)
    >>> assert_valid_observation(obs, module=my_module)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from visualpath.core.capabilities import ModuleCapabilities
from visualpath.core.compat import CompatibilityReport

if TYPE_CHECKING:
    import numpy as np
    from visualpath.core.module import Module


@dataclass
class FakeFrame:
    """Lightweight fake Frame for testing analyzers.

    Duck-type compatible with visualbase.Frame. Provides the three
    attributes every Module.process() uses: data, frame_id, t_src_ns,
    plus width/height for analyzers that inspect image dimensions.

    Example:
        >>> frame = FakeFrame.create()                  # 640x480 black
        >>> frame = FakeFrame.create(320, 240, frame_id=5)
        >>> frames = FakeFrame.sequence(3)              # 3 sequential frames
    """

    data: "np.ndarray"
    frame_id: int
    t_src_ns: int
    width: int
    height: int

    @classmethod
    def create(
        cls,
        width: int = 640,
        height: int = 480,
        frame_id: int = 0,
        t_src_ns: int = 0,
    ) -> "FakeFrame":
        """Create a single fake frame with a black BGR image.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            frame_id: Frame identifier.
            t_src_ns: Timestamp in nanoseconds.
        """
        import numpy as np

        data = np.zeros((height, width, 3), dtype=np.uint8)
        return cls(
            data=data,
            frame_id=frame_id,
            t_src_ns=t_src_ns,
            width=width,
            height=height,
        )

    @classmethod
    def sequence(
        cls,
        count: int,
        width: int = 640,
        height: int = 480,
        interval_ns: int = 33_333_333,
    ) -> List["FakeFrame"]:
        """Create a sequence of fake frames with incrementing IDs.

        Args:
            count: Number of frames.
            width: Image width in pixels.
            height: Image height in pixels.
            interval_ns: Nanoseconds between frames (default ~30fps).
        """
        return [
            cls.create(
                width=width,
                height=height,
                frame_id=i,
                t_src_ns=i * interval_ns,
            )
            for i in range(count)
        ]


def assert_valid_observation(
    obs: Any,
    *,
    module: Optional["Module"] = None,
    require_data: bool = False,
    require_timing: bool = False,
) -> None:
    """Assert that an Observation follows vpx conventions.

    Checks structural validity: correct types, non-empty source,
    and optionally matches against a module's name.

    Args:
        obs: Observation instance to validate.
        module: If provided, asserts obs.source == module.name.
        require_data: If True, asserts obs.data is not None.
        require_timing: If True, asserts obs.timing is present.

    Raises:
        AssertionError: If any check fails.

    Example:
        >>> obs = my_module.process(FakeFrame.create())
        >>> assert_valid_observation(obs, module=my_module)
    """
    from vpx.sdk.observation import Observation

    assert isinstance(obs, Observation), (
        f"Expected Observation, got {type(obs).__name__}"
    )
    assert isinstance(obs.source, str) and obs.source, (
        "Observation.source must be a non-empty string"
    )
    assert isinstance(obs.frame_id, int), (
        f"Observation.frame_id must be int, got {type(obs.frame_id).__name__}"
    )
    assert isinstance(obs.t_ns, int), (
        f"Observation.t_ns must be int, got {type(obs.t_ns).__name__}"
    )
    assert isinstance(obs.signals, dict), (
        f"Observation.signals must be dict, got {type(obs.signals).__name__}"
    )
    assert isinstance(obs.metadata, dict), (
        f"Observation.metadata must be dict, got {type(obs.metadata).__name__}"
    )

    if module is not None:
        assert obs.source == module.name, (
            f"Observation.source '{obs.source}' != module.name '{module.name}'"
        )

    if require_data:
        assert obs.data is not None, "Observation.data is None (require_data=True)"

    if obs.timing is not None:
        assert isinstance(obs.timing, dict), (
            f"Observation.timing must be dict or None, got {type(obs.timing).__name__}"
        )
        for key, val in obs.timing.items():
            assert isinstance(key, str), (
                f"timing key must be str, got {type(key).__name__}"
            )
            assert isinstance(val, (int, float)), (
                f"timing['{key}'] must be numeric, got {type(val).__name__}"
            )
    elif require_timing:
        raise AssertionError("Observation.timing is None (require_timing=True)")


@dataclass
class PluginCheckReport:
    """Result of a plugin convention check.

    Attributes:
        valid: True if all checks passed.
        warnings: Non-fatal issues.
        errors: Fatal convention violations.
        module_name: Name of the checked module.
    """

    valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    module_name: str = ""


class PluginTestHarness:
    """Validates module plugin conventions.

    Checks:
    - Module has a non-empty name
    - Module name uses dot notation (domain.action)
    - process() method exists and is callable
    - capabilities property returns ModuleCapabilities
    - port_schema is None or a PortSchema instance
    - error_policy is None or an ErrorPolicy instance
    """

    def check_module(self, module: "Module") -> PluginCheckReport:
        """Run all convention checks on a module.

        Args:
            module: Module instance to check.

        Returns:
            PluginCheckReport with results.
        """
        report = PluginCheckReport()

        # Check name
        try:
            name = module.name
            report.module_name = name
        except Exception as e:
            report.errors.append(f"module.name raised: {e}")
            report.valid = False
            return report

        if not name:
            report.errors.append("module.name is empty")
            report.valid = False

        if name and "." not in name:
            report.warnings.append(
                f"module.name '{name}' does not use dot notation (e.g. 'domain.action')"
            )

        # Check process()
        if not callable(getattr(module, 'process', None)):
            report.errors.append("module.process is not callable")
            report.valid = False

        # Check capabilities
        try:
            caps = module.capabilities
            if not isinstance(caps, ModuleCapabilities):
                report.errors.append(
                    f"module.capabilities returned {type(caps).__name__}, "
                    "expected ModuleCapabilities"
                )
                report.valid = False
        except Exception as e:
            report.errors.append(f"module.capabilities raised: {e}")
            report.valid = False

        # Check port_schema
        schema = getattr(module, 'port_schema', None)
        if schema is not None:
            from visualpath.core.capabilities import PortSchema
            if not isinstance(schema, PortSchema):
                report.warnings.append(
                    f"module.port_schema is {type(schema).__name__}, expected PortSchema or None"
                )

        # Check error_policy
        policy = getattr(module, 'error_policy', None)
        if policy is not None:
            from visualpath.core.error_policy import ErrorPolicy
            if not isinstance(policy, ErrorPolicy):
                report.warnings.append(
                    f"module.error_policy is {type(policy).__name__}, expected ErrorPolicy or None"
                )

        return report


    def assert_batch_correctness(
        self,
        module: "Module",
        batch_size: int = 4,
    ) -> None:
        """Verify process_batch() results match sequential process() calls.

        Runs the module in both modes and compares: source, frame_id,
        and signal keys/values. Timing data is excluded from comparison
        since it varies between runs.

        This is most useful for stateless modules. Stateful modules
        (with tracking, etc.) may produce different results due to
        state accumulation order.

        Args:
            module: Module instance to test. Must be initialized.
            batch_size: Number of frames to test with.

        Raises:
            AssertionError: If batch results don't match sequential.
        """
        frames = FakeFrame.sequence(batch_size)
        deps_list = [None] * batch_size

        # Sequential
        sequential = [module.process(f, d) for f, d in zip(frames, deps_list)]

        # Reset module state for fair comparison
        module.reset()

        # Batch
        batch = module.process_batch(frames, deps_list)

        assert len(batch) == len(sequential), (
            f"Batch length {len(batch)} != sequential length {len(sequential)}"
        )

        for i, (s, b) in enumerate(zip(sequential, batch)):
            if s is None and b is None:
                continue
            assert (s is None) == (b is None), (
                f"Frame {i}: sequential={s is None}, batch={b is None}"
            )
            if s is not None and b is not None:
                assert s.source == b.source, (
                    f"Frame {i}: source mismatch: {s.source!r} vs {b.source!r}"
                )
                assert s.frame_id == b.frame_id, (
                    f"Frame {i}: frame_id mismatch: {s.frame_id} vs {b.frame_id}"
                )
                # Compare signal keys and values
                for key in s.signals:
                    assert key in b.signals, (
                        f"Frame {i}: signal '{key}' missing in batch result"
                    )
                    assert s.signals[key] == b.signals[key], (
                        f"Frame {i}: signal '{key}' mismatch: "
                        f"{s.signals[key]} vs {b.signals[key]}"
                    )


__all__ = [
    "FakeFrame",
    "assert_valid_observation",
    "PluginTestHarness",
    "PluginCheckReport",
]
