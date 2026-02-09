"""High-level API for facemoment.

    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

All execution goes through a single path:
    fm.run() → build_graph(isolation_config) → Backend.execute()
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

from visualbase import Trigger

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10
DEFAULT_COOLDOWN = 2.0
DEFAULT_BACKEND = 'pathway'

# CUDA conflict groups: analyzers sharing the same CUDA runtime binding.
# If analyzers from 2+ groups are active, the minority group must run in
# a subprocess to avoid symbol conflicts (e.g. onnxruntime-gpu vs torch).
# NOTE: This is the legacy fallback. Prefer module.capabilities.resource_groups.
_CUDA_GROUPS: Dict[str, Set[str]] = {
    "onnxruntime": {"face.detect", "face.expression"},
    "torch": {"body.pose"},
}


@dataclass
class Result:
    """Result from fm.run()."""
    triggers: List[Trigger] = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0
    clips_extracted: int = 0
    actual_backend: str = ""


def build_modules(analyzers=None, *, cooldown=2.0, main_only=True):
    """facemoment 도메인 로직: analyzer 이름 → Module 리스트 구성.

    - FaceClassifier 자동 주입 (face/face_detect 사용 시)
    - HighlightFusion 생성
    - 문자열 이름은 visualpath 플러그인 레지스트리에서 해석됨

    Args:
        analyzers: Analyzer names. None = ["face.detect", "face.expression", "body.pose", "hand.gesture"].
        cooldown: Seconds between triggers.
        main_only: Only trigger for main face.

    Returns:
        List of module names (str) and Module instances (fusion).
    """
    from facemoment.moment_detector.fusion import HighlightFusion

    names = list(analyzers) if analyzers else ["face.detect", "face.expression", "body.pose", "hand.gesture"]

    if "face.detect" in names:
        if "face.classify" not in names:
            names.append("face.classify")

    fusion = HighlightFusion(cooldown_sec=cooldown, main_only=main_only)

    return names + [fusion]


def _build_isolation_config(analyzer_names):
    """Build IsolationConfig based on CUDA conflict detection.

    When analyzers from 2+ CUDA groups are active (e.g. onnxruntime-gpu
    for face + torch for pose), the minority group is configured for
    PROCESS isolation.

    Args:
        analyzer_names: List of analyzer names.

    Returns:
        IsolationConfig if isolation is needed, None otherwise.
    """
    from visualpath.core.isolation import IsolationConfig, IsolationLevel

    conflicts = _detect_cuda_conflicts(analyzer_names)
    if not conflicts:
        return None

    return IsolationConfig(
        default_level=IsolationLevel.INLINE,
        overrides={name: IsolationLevel.PROCESS for name in conflicts},
    )


def _detect_cuda_conflicts(names):
    """Detect CUDA conflicts among active analyzers.

    When analyzers from 2+ CUDA groups are active, the minority
    group is returned for subprocess isolation.

    Returns empty set if pyzmq is unavailable (falls back to ordering).

    Args:
        names: List of analyzer names to check.

    Returns:
        Set of analyzer names that should run in a subprocess.
    """
    try:
        import zmq  # noqa: F401
    except ImportError:
        logger.debug("pyzmq not available, skipping CUDA conflict detection")
        return set()

    # Map each analyzer to its CUDA group
    active_groups: Dict[str, List[str]] = {}
    for name in names:
        for group, members in _CUDA_GROUPS.items():
            if name in members:
                active_groups.setdefault(group, []).append(name)
                break

    if len(active_groups) < 2:
        return set()

    # Isolate the smallest group (fewest analyzers).
    # On tie, prefer isolating "torch" so onnxruntime stays in-process.
    _ISOLATE_PREFERENCE = {"torch": 0, "onnxruntime": 1}
    minority_group = min(
        active_groups,
        key=lambda g: (len(active_groups[g]), _ISOLATE_PREFERENCE.get(g, 99)),
    )
    isolated = set(active_groups[minority_group])
    logger.info(
        "CUDA conflict detected: groups %s. Isolating %s analyzers %s to subprocess.",
        list(active_groups.keys()), minority_group, isolated,
    )
    return isolated


def build_graph(modules, *, isolation=None, on_trigger=None):
    """Build a FlowGraph with optional isolation configuration.

    Args:
        modules: List of module names (str) or Module instances.
        isolation: Optional IsolationConfig for module execution.
        on_trigger: Optional callback for trigger events.

    Returns:
        FlowGraph configured for the pipeline.
    """
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.nodes.source import SourceNode
    from visualpath.flow.nodes.path import PathNode
    from visualpath.runner import resolve_modules

    module_instances = resolve_modules(modules)

    graph = FlowGraph(entry_node="source")
    graph.add_node(SourceNode(name="source"))
    graph.add_node(PathNode(
        name="pipeline",
        modules=module_instances,
        isolation=isolation,
    ))
    graph.add_edge("source", "pipeline")
    if on_trigger:
        graph.on_trigger(on_trigger)
    return graph


def _get_backend(backend):
    """Select the appropriate backend.

    Args:
        backend: Requested backend name.

    Returns:
        ExecutionBackend instance.
    """
    from visualpath.runner import get_backend

    return get_backend(backend)


def run(
    video: Union[str, Path],
    *,
    analyzers: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    fps: int = DEFAULT_FPS,
    cooldown: float = DEFAULT_COOLDOWN,
    backend: str = DEFAULT_BACKEND,
    profile: Optional[str] = None,
    on_trigger: Optional[Callable[[Trigger], None]] = None,
) -> Result:
    """Process a video and return results.

    All execution goes through a single path:
        run() → build_graph(isolation) → Backend.execute()

    Args:
        video: Path to video file.
        analyzers: Analyzer names ["face.detect", "face.expression", "body.pose", "hand.gesture", "quality"].
                   None = use all ML analyzers.
        output_dir: Directory for extracted clips. None = no clips.
        fps: Frames per second to process.
        cooldown: Seconds between triggers.
        backend: Execution backend ("pathway", "simple", or "worker").
        profile: Execution profile ("lite" or "platform"). Overrides backend/isolation defaults.
        on_trigger: Callback when a trigger fires.

    Returns:
        Result with triggers, frame_count, duration_sec, clips_extracted.

    Example:
        >>> result = fm.run("video.mp4")
        >>> result = fm.run("video.mp4", output_dir="./clips")
        >>> result = fm.run("video.mp4", analyzers=["face", "body.pose"])
        >>> result = fm.run("video.mp4", backend="pathway")
    """
    from facemoment.cli.utils import create_video_stream

    # 1. Build analyzer name list
    analyzer_names = list(analyzers) if analyzers else ["face.detect", "face.expression", "body.pose", "hand.gesture"]

    # 2. Build modules (facemoment domain logic)
    modules = build_modules(analyzer_names, cooldown=cooldown)

    # 3. Resolve profile or build isolation config
    effective_backend = backend
    if profile is not None:
        from visualpath.core.profile import ExecutionProfile, resolve_profile
        from visualpath.runner import resolve_modules as _resolve
        exec_profile = ExecutionProfile.from_name(profile)
        resolved = _resolve(modules)
        isolation_config = resolve_profile(exec_profile, resolved)
        effective_backend = exec_profile.backend
    else:
        isolation_config = _build_isolation_config(analyzer_names)

    # 4. Build FlowGraph (with isolation info in ModuleSpec)
    graph = build_graph(modules, isolation=isolation_config, on_trigger=on_trigger)

    # 5. Select backend (WorkerBackend if isolation needed)
    engine = _get_backend(effective_backend)

    # 6. Open video source
    vb, source, stream = create_video_stream(str(video), fps=fps)

    # 7. Execute (single path: FlowGraph → Backend.execute())
    try:
        pipeline_result = engine.execute(stream, graph)
    finally:
        vb.disconnect()

    # 8. Clip extraction (post-processing)
    clips_extracted = 0
    if output_dir:
        clips_extracted = _extract_clips(str(video), pipeline_result.triggers, output_dir)

    return Result(
        triggers=pipeline_result.triggers,
        frame_count=pipeline_result.frame_count,
        duration_sec=pipeline_result.frame_count / fps if pipeline_result.frame_count > 0 else 0.0,
        clips_extracted=clips_extracted,
        actual_backend=engine.name,
    )


def _extract_clips(video_path, triggers, output_dir):
    """트리거 기반 클립 추출 (visualbase)."""
    from visualbase import VisualBase, FileSource

    clip_vb = VisualBase(clip_output_dir=Path(output_dir))
    clip_vb.connect(FileSource(video_path))
    clips = sum(1 for t in triggers if clip_vb.trigger(t).success)
    clip_vb.disconnect()
    return clips


# Legacy compatibility
def _needs_cuda_isolation(analyzer_names):
    """Check if the analyzer combination requires CUDA subprocess isolation.

    .. deprecated::
        Use ``_build_isolation_config()`` instead.
    """
    return bool(_detect_cuda_conflicts(analyzer_names))


if __name__ == "__main__":
    pass
