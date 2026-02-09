"""Base module interface for feature analysis (B modules).

This module re-exports the base classes from visualpath and provides
analyzer-specific observation types.

Usage:
    Inherit from Module and implement process():
    - Return Observation from process()
    - For trigger modules, set signals["should_trigger"] = True
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Re-export Module from visualpath
from visualpath.core.module import Module as VisualPathModule
from visualpath.core.isolation import IsolationLevel

class Module(VisualPathModule):
    """Module with backwards-compatible analyze() alias.

    This class extends visualpath's Module to provide the legacy `analyze()`
    method as an alias for `process()`.
    """

    def analyze(self, frame: "Frame", deps=None) -> Optional["Observation"]:
        """Backwards-compatible alias for process().

        Args:
            frame: Input frame to analyze.
            deps: Optional dependency observations.

        Returns:
            Observation from processing.
        """
        return self.process(frame, deps)


# Backwards compatibility alias
BaseAnalyzer = Module

# Re-export for backward compatibility
from visualbase import Frame  # noqa: F401


@dataclass
class ProcessingStep:
    """Describes a single processing step within an analyzer.

    Used to define and visualize the internal data flow of analyzers.
    Steps are registered via the @processing_step decorator.

    Attributes:
        name: Short identifier for the step (e.g., "detection", "tracking").
        description: Human-readable description of what this step does.
        backend: Backend/library used (e.g., "InsightFace SCRFD", "MediaPipe").
        input_type: Description of input data type.
        output_type: Description of output data type.
        optional: Whether this step can be skipped.
        depends_on: List of step names this step depends on.
        method_name: Name of the method implementing this step.
    """

    name: str
    description: str
    backend: Optional[str] = None
    input_type: str = "Frame"
    output_type: str = "Data"
    optional: bool = False
    depends_on: List[str] = field(default_factory=list)
    method_name: Optional[str] = None

    def __str__(self) -> str:
        backend_str = f" ({self.backend})" if self.backend else ""
        return f"{self.name}: {self.description}{backend_str}"


def processing_step(
    name: str,
    description: str = "",
    backend: Optional[str] = None,
    input_type: str = "Any",
    output_type: str = "Any",
    optional: bool = False,
    depends_on: Optional[List[str]] = None,
):
    """Decorator to register a method as a processing step.

    This decorator:
    1. Registers the step in the analyzer's step registry
    2. Can optionally wrap the method for timing tracking
    3. Enables DAG visualization and runtime monitoring

    Args:
        name: Unique identifier for this step within the analyzer.
        description: Human-readable description.
        backend: Backend/library name (shown in visualizations).
        input_type: Description of input data type.
        output_type: Description of output data type.
        optional: Whether this step can be skipped at runtime.
        depends_on: List of step names this step depends on.

    Example:
        class FaceAnalyzer(Module):
            @processing_step(
                "detection",
                description="Detect faces with landmarks",
                backend="InsightFace SCRFD",
                input_type="Frame (BGR)",
                output_type="List[DetectedFace]",
            )
            def _detect_faces(self, image):
                return self._face_backend.detect(image)

            @processing_step(
                "expression",
                description="Analyze facial expressions",
                backend="HSEmotion",
                depends_on=["detection"],
                optional=True,
            )
            def _analyze_expression(self, image, faces):
                return self._expression_backend.analyze(image, faces)
    """
    from functools import wraps
    import time

    def decorator(func):
        # Create step definition
        step_info = ProcessingStep(
            name=name,
            description=description or func.__doc__ or "",
            backend=backend,
            input_type=input_type,
            output_type=output_type,
            optional=optional,
            depends_on=depends_on or [],
            method_name=func.__name__,
        )

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Track timing if analyzer has _step_timings dict
            if hasattr(self, '_step_timings') and self._step_timings is not None:
                start = time.perf_counter_ns()
                result = func(self, *args, **kwargs)
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                self._step_timings[name] = elapsed_ms
                return result
            return func(self, *args, **kwargs)

        # Store step info on the wrapper function
        wrapper._step_info = step_info
        return wrapper

    return decorator


def get_processing_steps(cls_or_instance) -> List[ProcessingStep]:
    """Get all registered processing steps from an analyzer class or instance.

    Args:
        cls_or_instance: Analyzer class or instance.

    Returns:
        List of ProcessingStep in dependency order.
    """
    cls = cls_or_instance if isinstance(cls_or_instance, type) else type(cls_or_instance)
    steps = []

    for attr_name in dir(cls):
        try:
            attr = getattr(cls, attr_name)
            if callable(attr) and hasattr(attr, '_step_info'):
                steps.append(attr._step_info)
        except Exception:
            continue

    # Sort by dependency order (topological sort)
    return _topological_sort_steps(steps)


def _topological_sort_steps(steps: List[ProcessingStep]) -> List[ProcessingStep]:
    """Sort steps by dependency order."""
    if not steps:
        return []

    # Build name -> step mapping
    by_name = {s.name: s for s in steps}
    result = []
    visited = set()
    temp_mark = set()

    def visit(step: ProcessingStep):
        if step.name in temp_mark:
            raise ValueError(f"Circular dependency detected involving {step.name}")
        if step.name in visited:
            return
        temp_mark.add(step.name)
        for dep_name in step.depends_on:
            if dep_name in by_name:
                visit(by_name[dep_name])
        temp_mark.remove(step.name)
        visited.add(step.name)
        result.append(step)

    for step in steps:
        if step.name not in visited:
            visit(step)

    return result


@dataclass
class Observation:
    """Observation output from an analyzer.

    Observations are timestamped feature analyses that flow from
    B modules (analyzers) to C module (fusion).

    For trigger modules, set trigger info in signals:
    - signals["should_trigger"]: Whether to fire a trigger
    - signals["trigger_score"]: Confidence score [0, 1]
    - signals["trigger_reason"]: Reason for the trigger
    - metadata["trigger"]: Trigger object

    Attributes:
        source: Name of the analyzer that produced this observation.
        frame_id: Frame identifier from the source video.
        t_ns: Timestamp in nanoseconds (source timeline).
        signals: Dictionary of extracted signals/features.
        data: Type-safe output data (e.g., PoseOutput, FaceClassifierOutput).
        metadata: Additional metadata about the observation.
        timing: Optional per-component timing in milliseconds.
    """

    source: str
    frame_id: int
    t_ns: int
    signals: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Any] = None  # Type-safe output (PoseOutput, FaceDetectOutput, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Dict[str, float]] = None  # {"detect_ms": 42.3, "expression_ms": 28.1}

    # Trigger helper properties (matches visualpath.core.Observation)

    @property
    def should_trigger(self) -> bool:
        """Check if this observation indicates a trigger should fire."""
        return bool(self.signals.get("should_trigger", False))

    @property
    def trigger_score(self) -> float:
        """Get the trigger confidence score."""
        return float(self.signals.get("trigger_score", 0.0))

    @property
    def trigger_reason(self) -> str:
        """Get the trigger reason."""
        return str(self.signals.get("trigger_reason", ""))

    @property
    def trigger(self) -> Optional[Any]:
        """Get the Trigger object if present."""
        return self.metadata.get("trigger")

    # Backwards-compatible aliases
    @property
    def score(self) -> float:
        """Alias for trigger_score."""
        return self.trigger_score

    @property
    def reason(self) -> str:
        """Alias for trigger_reason."""
        return self.trigger_reason


__all__ = [
    "Module",
    "BaseAnalyzer",  # Alias for Module
    "ProcessingStep",
    "processing_step",
    "get_processing_steps",
    "Observation",
    "IsolationLevel",
]
