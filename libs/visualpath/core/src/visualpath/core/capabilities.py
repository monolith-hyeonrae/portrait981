"""Capability declarations for modules.

Modules declare their capabilities as metadata, enabling:
- Compatibility checking between modules (resource group conflicts)
- GPU memory estimation
- Profile-based isolation decisions

Example:
    >>> class FaceDetector(Module):
    ...     @property
    ...     def capabilities(self) -> ModuleCapabilities:
    ...         return ModuleCapabilities(
    ...             flags=Capability.STATEFUL | Capability.GPU,
    ...             gpu_memory_mb=512,
    ...             resource_groups=frozenset({"onnxruntime"}),
    ...         )
"""

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import FrozenSet


class Capability(Flag):
    """Capability flags for modules.

    Combine with bitwise OR: ``Capability.GPU | Capability.STATEFUL``.
    """

    NONE = 0
    STATEFUL = auto()          # Maintains state across frames
    GPU = auto()               # Requires CUDA/ROCm
    BATCHING = auto()          # Supports batch processing
    CROSS_HOST_OK = auto()     # Safe for remote/distributed execution
    NEEDS_ZERO_COPY = auto()   # Requires shared memory frame access
    DETERMINISTIC = auto()     # Same input -> same output
    THREAD_SAFE = auto()       # Safe for multi-threaded calls


@dataclass(frozen=True)
class ModuleCapabilities:
    """Declarative capability metadata for a module.

    All fields have safe defaults so existing modules work unchanged.

    Attributes:
        flags: Bitwise combination of Capability flags.
        gpu_memory_mb: Estimated GPU memory usage in megabytes.
        init_time_sec: Estimated initialization time in seconds.
        max_batch_size: Maximum batch size (1 = no batching).
        resource_groups: Runtime groups that may conflict (e.g. {"onnxruntime"}, {"torch"}).
            Modules in different groups may need process isolation.
        required_extras: Package extras needed at install time.
    """

    flags: Capability = Capability.NONE
    gpu_memory_mb: int = 0
    init_time_sec: float = 0.0
    max_batch_size: int = 1
    resource_groups: FrozenSet[str] = field(default_factory=frozenset)
    required_extras: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PortSchema:
    """Declarative port type schema for a module.

    Declares expected input/output signal keys and data types,
    enabling static validation of module connections.

    Attributes:
        version: Schema version string (e.g. "1.0").
        input_signals: Signal keys required from dependency modules.
        output_signals: Signal keys guaranteed in output Observation.
        output_data_type: Qualified name of Observation.data type (e.g. "vpx.face_detect.output.FaceDetectOutput").
    """

    version: str = "1.0"
    input_signals: FrozenSet[str] = field(default_factory=frozenset)
    output_signals: FrozenSet[str] = field(default_factory=frozenset)
    output_data_type: str = ""


__all__ = ["Capability", "ModuleCapabilities", "PortSchema"]
