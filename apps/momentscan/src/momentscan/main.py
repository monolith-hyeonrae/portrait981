"""High-level API for momentscan.

    >>> import momentscan as ms
    >>> result = ms.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

All execution goes through a single path:
    ms.run() → MomentscanApp().run()
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import visualpath as vp
from visualbase import Trigger

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10
DEFAULT_COOLDOWN = 2.0
DEFAULT_BACKEND = "pathway"


@dataclass
class Result:
    """Result from ms.run()."""

    triggers: List[Trigger] = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0
    clips_extracted: int = 0
    actual_backend: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class MomentscanApp(vp.App):
    """981파크 얼굴/장면 분석 앱."""

    fps = DEFAULT_FPS
    backend = DEFAULT_BACKEND

    def __init__(self, *, analyzers=None, cooldown=2.0, main_only=True, output_dir=None):
        self._cooldown = cooldown
        self._main_only = main_only
        self._output_dir = output_dir
        self._vb = None
        self._clips_extracted = 0

    def configure_modules(self, modules):
        from momentscan.algorithm.analyzers.highlight import HighlightFusion
        from vpx.sdk.paths import get_models_dir

        names = list(modules) if modules else [
            "face.detect", "face.expression", "body.pose", "hand.gesture",
            "frame.quality", "frame.scoring",
        ]

        if "face.detect" in names and "face.classify" not in names:
            names.append("face.classify")

        resolved = super().configure_modules(names)

        # Inject centralized models directory into analyzers
        models_dir = get_models_dir()
        for module in resolved:
            if hasattr(module, "_models_dir"):
                module._models_dir = models_dir

        resolved.append(HighlightFusion(
            cooldown_sec=self._cooldown, main_only=self._main_only,
        ))
        return resolved

    def setup(self):
        if self._output_dir:
            from visualbase import VisualBase, FileSource
            self._vb = VisualBase(clip_output_dir=Path(self._output_dir))
            self._vb.connect(FileSource(str(self.video)))
            self._clips_extracted = 0

    def on_trigger(self, trigger):
        if self._vb is not None:
            result = self._vb.trigger(trigger)
            if result.success:
                self._clips_extracted += 1

    def teardown(self):
        if self._vb is not None:
            try:
                self._vb.disconnect()
            except Exception:
                pass
            self._vb = None

    def after_run(self, result):
        return Result(
            triggers=result.triggers,
            frame_count=result.frame_count,
            duration_sec=result.duration_sec,
            clips_extracted=self._clips_extracted,
            actual_backend=result.actual_backend,
            stats=result.stats,
        )



def run(
    video: Union[str, Path],
    *,
    analyzers: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    fps: int = DEFAULT_FPS,
    cooldown: float = DEFAULT_COOLDOWN,
    backend: str = DEFAULT_BACKEND,
    profile: Optional[str] = None,
    isolation: Optional[Any] = None,
    on_trigger: Optional[Callable[[Trigger], None]] = None,
    on_frame: Optional[Callable] = None,
) -> Result:
    """Process a video and return results.

    momentscan one-liner: MomentscanApp().run().

    Args:
        video: Path to video file.
        analyzers: Analyzer names ["face.detect", "face.expression", "body.pose", "hand.gesture", "quality"].
                   None = use all ML analyzers.
        output_dir: Directory for extracted clips. None = no clips.
        fps: Frames per second to process.
        cooldown: Seconds between triggers.
        backend: Execution backend ("pathway", "simple", or "worker").
        profile: Execution profile ("lite" or "platform"). Overrides backend/isolation defaults.
        isolation: Optional IsolationConfig for explicit module isolation.
            When provided, overrides auto-detected CUDA conflict isolation.
            Backend is automatically switched to "worker" if isolation is set.
        on_trigger: Callback when a trigger fires.
        on_frame: Optional per-frame callback ``(frame, terminal_results) -> bool``.
            Called after each frame is processed with the Frame and a list of
            FlowData from terminal nodes. Return True to continue, False to stop.

    Returns:
        Result with triggers, frame_count, duration_sec, clips_extracted.

    Example:
        >>> result = ms.run("video.mp4")
        >>> result = ms.run("video.mp4", output_dir="./clips")
        >>> result = ms.run("video.mp4", analyzers=["face.detect", "body.pose"])
        >>> result = ms.run("video.mp4", backend="pathway")
    """
    app = MomentscanApp(
        analyzers=analyzers, cooldown=cooldown, output_dir=output_dir,
    )
    return app.run(
        video,
        modules=list(analyzers) if analyzers else None,
        fps=fps,
        backend=backend,
        isolation=isolation,
        profile=profile,
        on_trigger=on_trigger,
        on_frame=on_frame,
    )
