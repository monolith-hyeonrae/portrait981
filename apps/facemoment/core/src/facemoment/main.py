"""High-level API for facemoment.

    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from visualbase import Trigger

DEFAULT_FPS = 10
DEFAULT_COOLDOWN = 2.0
DEFAULT_BACKEND = 'pathway'


@dataclass
class Result:
    """Result from fm.run()."""
    triggers: List[Trigger] = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0
    clips_extracted: int = 0
    actual_backend: str = ""


def build_modules(extractors=None, *, cooldown=2.0, main_only=True):
    """facemoment 도메인 로직: extractor 이름 → Module 리스트 구성.

    - FaceClassifier 자동 주입 (face/face_detect 사용 시)
    - HighlightFusion 생성
    - 문자열 이름은 visualpath 플러그인 레지스트리에서 해석됨

    Args:
        extractors: Extractor names. None = ["face", "pose", "gesture"].
        cooldown: Seconds between triggers.
        main_only: Only trigger for main face.

    Returns:
        List of module names (str) and Module instances (fusion).
    """
    from facemoment.moment_detector.fusion import HighlightFusion

    names = list(extractors) if extractors else ["face", "pose", "gesture"]

    if any(n in ("face", "face_detect") for n in names):
        if "face_classifier" not in names:
            names.append("face_classifier")

    fusion = HighlightFusion(cooldown_sec=cooldown, main_only=main_only)

    return names + [fusion]


def _needs_cuda_isolation(extractor_names):
    """Check if the extractor combination requires CUDA subprocess isolation."""
    from facemoment.pipeline.pathway_pipeline import FacemomentPipeline
    return bool(FacemomentPipeline._detect_cuda_conflicts(extractor_names))


def run(
    video: Union[str, Path],
    *,
    extractors: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    fps: int = DEFAULT_FPS,
    cooldown: float = DEFAULT_COOLDOWN,
    backend: str = DEFAULT_BACKEND,
    on_trigger: Optional[Callable[[Trigger], None]] = None,
) -> Result:
    """Process a video and return results.

    Args:
        video: Path to video file.
        extractors: Extractor names ["face", "pose", "gesture", "quality"].
                   None = use all ML extractors.
        output_dir: Directory for extracted clips. None = no clips.
        fps: Frames per second to process.
        cooldown: Seconds between triggers.
        backend: Execution backend ("pathway" or "simple").
        on_trigger: Callback when a trigger fires.

    Returns:
        Result with triggers, frame_count, duration_sec, clips_extracted.

    Example:
        >>> result = fm.run("video.mp4")
        >>> result = fm.run("video.mp4", output_dir="./clips")
        >>> result = fm.run("video.mp4", extractors=["face", "pose"])
        >>> result = fm.run("video.mp4", backend="pathway")
    """
    # Build extractor name list for conflict check
    extractor_names = list(extractors) if extractors else ["face", "pose", "gesture"]

    # CUDA conflict → FacemomentPipeline (ProcessWorker 격리 필요)
    if _needs_cuda_isolation(extractor_names):
        return _run_with_pipeline(video, extractor_names, fps=fps, cooldown=cooldown,
                                  output_dir=output_dir, on_trigger=on_trigger)

    # No conflict → FlowGraph + backend.execute() (깔끔한 3계층)
    return _run_with_flowgraph(video, extractor_names, fps=fps, cooldown=cooldown,
                               backend=backend, output_dir=output_dir, on_trigger=on_trigger)


def _run_with_flowgraph(video, extractor_names, *, fps, cooldown, backend, output_dir, on_trigger):
    """FlowGraph + backend.execute() 경로 (CUDA 충돌 없을 때)."""
    from visualpath.flow.graph import FlowGraph
    from visualpath.runner import resolve_modules, get_backend
    from facemoment.cli.utils import create_video_stream

    # 1. 모듈 구성 (facemoment 도메인)
    modules = build_modules(extractor_names, cooldown=cooldown)
    module_instances = resolve_modules(modules)

    # 2. FlowGraph 구성 (visualpath DAG)
    graph = FlowGraph.from_modules(module_instances, on_trigger=on_trigger)

    # 3. 비디오 소스 (visualbase 미디어 I/O)
    vb, source, stream = create_video_stream(str(video), fps=fps)

    # 4. 백엔드 실행 (visualpath)
    try:
        engine = get_backend(backend)
        pipeline_result = engine.execute(stream, graph)
    finally:
        vb.disconnect()

    # 5. 클립 추출 (facemoment 후처리)
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


def _run_with_pipeline(video, extractor_names, *, fps, cooldown, output_dir, on_trigger):
    """FacemomentPipeline 경로 (CUDA 충돌 시 ProcessWorker 격리)."""
    from facemoment.pipeline.pathway_pipeline import FacemomentPipeline
    from facemoment.cli.utils import create_video_stream

    pipeline = FacemomentPipeline(
        extractors=extractor_names,
        fusion_config={"cooldown_sec": cooldown, "main_only": True},
    )

    vb, source, stream = create_video_stream(str(video), fps=fps)

    result = Result()
    try:
        frames = list(stream)
        result.frame_count = len(frames)
        result.duration_sec = result.frame_count / fps if fps > 0 else 0

        triggers = pipeline.run(frames, on_trigger=on_trigger)
        result.triggers = triggers
        result.actual_backend = pipeline.actual_backend or "pipeline"

        if output_dir:
            result.clips_extracted = _extract_clips(str(video), triggers, output_dir)
    finally:
        vb.disconnect()

    return result


def _extract_clips(video_path, triggers, output_dir):
    """트리거 기반 클립 추출 (visualbase)."""
    from visualbase import VisualBase, FileSource

    clip_vb = VisualBase(clip_output_dir=Path(output_dir))
    clip_vb.connect(FileSource(video_path))
    clips = sum(1 for t in triggers if clip_vb.trigger(t).success)
    clip_vb.disconnect()
    return clips



if __name__ == "__main__":
    pass
