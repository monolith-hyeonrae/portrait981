"""Pipeline runner for visualpath.

This module provides the ``run()`` entry point for running video
analysis pipelines. It handles:
- Module resolution from registry
- Isolation resolution (profile, explicit, or auto-detect)
- Video source opening (visualbase or OpenCV fallback)
- Backend selection and FlowGraph construction
- Pipeline execution via ``backend.execute(frames, graph)``

Example:
    >>> import visualpath as vp
    >>>
    >>> result = vp.run("video.mp4", modules=[face_detector, smile_trigger])
    >>> print(f"Found {len(result.triggers)} triggers")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from visualbase import Frame, Trigger

from visualpath.core.observation import Observation
from visualpath.core.module import Module

logger = logging.getLogger(__name__)

# Backend type alias
BackendType = Literal["simple", "pathway", "worker"]


@dataclass
class ProcessResult:
    """Result from processing a video."""
    triggers: List[Trigger] = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0
    actual_backend: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


def get_backend(backend: BackendType, *, batch_size: int = 1) -> "ExecutionBackend":
    """Get execution backend by name.

    Args:
        backend: Backend name ("simple", "pathway", or "worker").
        batch_size: Batch size for GPU module processing.

    Returns:
        ExecutionBackend instance.

    Raises:
        ValueError: If backend is unknown.
        ImportError: If Pathway or Worker backend is requested but not installed.
    """
    from visualpath.backends.base import ExecutionBackend

    if backend == "simple":
        from visualpath.backends.simple import SimpleBackend
        return SimpleBackend(batch_size=batch_size)
    elif backend == "pathway":
        try:
            from visualpath.backends.pathway import PathwayBackend
            return PathwayBackend()
        except ImportError as e:
            raise ImportError(
                "Pathway backend requires pathway package. "
                "Install with: pip install visualpath[pathway]"
            ) from e
    elif backend == "worker":
        try:
            from visualpath.backends.worker import WorkerBackend
            return WorkerBackend(batch_size=batch_size)
        except ImportError as e:
            raise ImportError(
                "Worker backend requires visualpath-isolation package. "
                "Install with: pip install visualpath-isolation"
            ) from e
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'simple', 'pathway', or 'worker'.")


def resolve_modules(
    modules: Sequence[Union[str, Module]],
) -> List[Module]:
    """Resolve module names or instances to instances.

    Modules can be either module names (from registry) or Module instances.

    Args:
        modules: List of module names or instances.

    Returns:
        List of Module instances.

    Raises:
        ValueError: If a module name is not found.
    """
    from visualpath.api import get_module

    instances: List[Module] = []
    for mod in modules:
        if isinstance(mod, str):
            instance = get_module(mod)
            if instance is None:
                raise ValueError(f"Unknown module: {mod}")
            instances.append(instance)
        else:
            instances.append(mod)
    return instances


def _open_video_source(video: Union[str, Path], fps: int):
    """Open a video source, returning (frames_iterator, cleanup_fn).

    Tries visualbase first, falls back to OpenCV.

    Args:
        video: Path to video file.
        fps: Frames per second to process.

    Returns:
        Tuple of (frames_iterator, optional_cleanup_function).
    """
    vb = None
    try:
        from visualbase import VideoBase
        vb = VideoBase()
        source = vb.open(str(video))
        frames = source.stream(fps=fps)
        return frames, lambda: vb.disconnect()
    except Exception:
        if vb is not None:
            try:
                vb.disconnect()
            except Exception:
                pass
        # Fallback to OpenCV
        return _opencv_frames(str(video), fps), None


def _opencv_frames(video_path: str, fps: int) -> Iterator[Frame]:
    """Fallback frame iterator using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(src_fps / fps))
        frame_id = 0
        read_count = 0

        while True:
            ret, data = cap.read()
            if not ret:
                break

            read_count += 1
            if read_count % skip != 0:
                continue

            t_ns = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)

            yield Frame.from_array(
                data,
                frame_id=frame_id,
                t_src_ns=t_ns,
            )
            frame_id += 1

    finally:
        cap.release()


# Default values (imported from api for consistency)
from visualpath.api import DEFAULT_FPS


def run(
    video: Union[str, Path],
    modules: Sequence[Union[str, Module]],
    *,
    fps: int = DEFAULT_FPS,
    batch_size: int = 1,
    backend: BackendType = "simple",
    isolation: Optional[Any] = None,
    profile: Optional[str] = None,
    on_trigger: Optional[Callable[[Trigger], None]] = None,
    on_frame: Optional[Callable] = None,
) -> ProcessResult:
    """Level 1: One-liner pipeline execution. Thin wrapper around App.

    Args:
        video: Path to video file.
        modules: List of module names or instances.
        fps: Frames per second to process.
        batch_size: Batch size for GPU module processing (default: 1).
        backend: Execution backend ("simple", "pathway", or "worker").
        isolation: Optional IsolationConfig for module execution.
            When provided, backend is auto-switched to "worker".
        profile: Execution profile name (e.g. "lite", "platform").
            Overrides backend/isolation defaults.
        on_trigger: Callback when a trigger fires.
        on_frame: Optional per-frame callback ``(frame, terminal_results) -> bool``.
            Return True to continue, False to stop.

    Returns:
        ProcessResult with triggers, frame_count, duration_sec, actual_backend, stats.

    Example:
        >>> result = vp.run("video.mp4", modules=[face_detector, smile_trigger])
        >>> result = vp.run("video.mp4", modules=["face.detect"], profile="lite")
    """
    from visualpath.app import App

    app = App()
    return app.run(
        video,
        modules=modules,
        fps=fps,
        batch_size=batch_size,
        backend=backend,
        isolation=isolation,
        profile=profile,
        on_trigger=on_trigger,
        on_frame=on_frame,
    )


# Alias for backward compatibility
process_video = run

__all__ = [
    "ProcessResult",
    "BackendType",
    "run",
    "process_video",
    "get_backend",
    "resolve_modules",
]
