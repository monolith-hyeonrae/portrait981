"""LiteRunner - Lightweight analyzer execution engine.

Runs vpx modules on a video source without requiring the full
momentscan pipeline. Handles dependency resolution, topological
sorting, and lifecycle management.

Example:
    >>> from vpx.runner import LiteRunner
    >>> runner = LiteRunner("face.detect")
    >>> result = runner.run("video.mp4", max_frames=10)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from visualpath.core.graph import toposort_modules
from visualpath.plugin.discovery import load_analyzer, discover_analyzers

logger = logging.getLogger(__name__)

# Type alias for callbacks
ObservationCallback = Callable[[str, Any], None]
FrameCallback = Callable[[Any, Dict[str, Any]], None]


@dataclass
class RunResult:
    """Result of a LiteRunner.run() invocation.

    Attributes:
        frames: Per-frame observation dicts (module_name -> Observation).
        frame_count: Total frames processed.
        module_names: Names of modules that were executed (in topo order).
    """

    frames: List[Dict[str, Any]] = field(default_factory=list)
    frame_count: int = 0
    module_names: List[str] = field(default_factory=list)


class LiteRunner:
    """Lightweight runner for vpx analyzer modules.

    Resolves analyzer dependencies, manages lifecycle, and processes
    frames from a video source or list of fake frames.

    Args:
        analyzers: Analyzer name(s) or Module instance(s).
        on_observation: Callback ``(name, obs)`` fired per module per frame.
        on_frame: Callback ``(frame, obs_dict)`` fired per frame after all modules.

    Example:
        >>> runner = LiteRunner("face.detect")
        >>> result = runner.run("video.mp4", fps=5, max_frames=100)
        >>> print(f"Processed {result.frame_count} frames")
    """

    def __init__(
        self,
        analyzers: Union[str, Any, List[Union[str, Any]]],
        *,
        on_observation: Optional[ObservationCallback] = None,
        on_frame: Optional[FrameCallback] = None,
    ):
        if isinstance(analyzers, (str, type)) or (
            not isinstance(analyzers, list) and hasattr(analyzers, "name")
        ):
            analyzers = [analyzers]
        self._analyzer_specs = analyzers
        self._on_observation = on_observation
        self._on_frame = on_frame

    def run(
        self,
        source: Any,
        *,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        on_observation: Optional[ObservationCallback] = None,
        on_frame: Optional[FrameCallback] = None,
    ) -> RunResult:
        """Run analyzers on a source.

        Args:
            source: File path (str), BaseSource instance, or list of frames.
            fps: Target FPS for frame skipping (None = process all).
            max_frames: Stop after this many processed frames.
            on_observation: Per-run callback override.
            on_frame: Per-run callback override.

        Returns:
            RunResult with per-frame observations.
        """
        obs_cb = on_observation or self._on_observation
        frame_cb = on_frame or self._on_frame

        modules = self._resolve_analyzers()
        sorted_modules = toposort_modules(modules)
        module_names = [m.name for m in sorted_modules]

        result = RunResult(module_names=module_names)

        # Initialize modules
        for mod in sorted_modules:
            mod.initialize()

        try:
            source_iter = self._resolve_source(source)
            skip_interval = self._compute_skip_interval(fps, source)

            frame_index = 0
            for frame in source_iter:
                # FPS-based skip
                if skip_interval and frame_index % skip_interval != 0:
                    frame_index += 1
                    continue
                frame_index += 1

                deps: Dict[str, Any] = {}
                frame_obs: Dict[str, Any] = {}

                for mod in sorted_modules:
                    mod_deps = {
                        d: deps[d]
                        for d in list(getattr(mod, "depends", []))
                        + list(getattr(mod, "optional_depends", []))
                        if d in deps
                    }
                    obs = mod.process(frame, mod_deps if mod_deps else None)
                    if obs is not None:
                        deps[mod.name] = obs
                        frame_obs[mod.name] = obs
                        if obs_cb:
                            obs_cb(mod.name, obs)

                if frame_cb:
                    frame_cb(frame, frame_obs)

                result.frames.append(frame_obs)
                result.frame_count += 1

                if max_frames and result.frame_count >= max_frames:
                    break
        finally:
            # Cleanup modules
            for mod in sorted_modules:
                try:
                    mod.cleanup()
                except Exception:
                    logger.debug("Cleanup error for %s", mod.name, exc_info=True)

        return result

    def _resolve_analyzers(self) -> List[Any]:
        """Resolve analyzer specs to Module instances with BFS dep resolution."""
        resolved: Dict[str, Any] = {}
        queue = list(self._analyzer_specs)

        while queue:
            spec = queue.pop(0)
            if isinstance(spec, str):
                if spec in resolved:
                    continue
                cls = load_analyzer(spec)
                instance = cls()
                resolved[instance.name] = instance
                # BFS: enqueue required deps (not optional_depends)
                for dep_name in getattr(instance, "depends", []):
                    if dep_name not in resolved:
                        queue.append(dep_name)
            else:
                # Already a Module instance
                if spec.name not in resolved:
                    resolved[spec.name] = spec
                    for dep_name in getattr(spec, "depends", []):
                        if dep_name not in resolved:
                            queue.append(dep_name)

        return list(resolved.values())

    def _resolve_source(self, source: Any):
        """Convert source to an iterable of frames."""
        if isinstance(source, list):
            return iter(source)
        if isinstance(source, str):
            return _FileSourceIterable(source)
        # Assume BaseSource-like
        return _SourceIterable(source)

    def _compute_skip_interval(
        self, target_fps: Optional[float], source: Any
    ) -> Optional[int]:
        """Compute skip interval for FPS limiting."""
        if target_fps is None:
            return None
        source_fps = getattr(source, "fps", None)
        if source_fps and source_fps > target_fps:
            return max(1, round(source_fps / target_fps))
        return None


class _FileSourceIterable:
    """Wraps FileSource in an iterable with open/close management."""

    def __init__(self, path: str):
        self._path = path

    def __iter__(self):
        from visualbase import FileSource

        source = FileSource(self._path)
        source.open()
        try:
            while True:
                frame = source.read()
                if frame is None:
                    break
                yield frame
        finally:
            source.close()


class _SourceIterable:
    """Wraps a BaseSource in an iterable with context management."""

    def __init__(self, source):
        self._source = source

    def __iter__(self):
        self._source.open()
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break
                yield frame
        finally:
            self._source.close()
