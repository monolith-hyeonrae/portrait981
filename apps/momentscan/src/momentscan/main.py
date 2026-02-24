"""High-level API for momentscan.

    >>> import momentscan as ms
    >>> result = ms.run("video.mp4")
    >>> print(f"Found {len(result.highlights)} highlights")

All execution goes through a single path:
    ms.run() → MomentscanApp().run()
"""

import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import visualpath as vp

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10
DEFAULT_BACKEND = "simple"


@dataclass
class Result:
    """Result from ms.run().

    Attributes:
        highlights: Detected highlight windows.
        frame_count: Total frames processed.
        duration_sec: Video duration in seconds.
        actual_backend: Backend that was actually used.
        stats: Backend execution statistics.
    """

    highlights: List[Any] = field(default_factory=list)
    identity: Optional[Any] = None
    frame_count: int = 0
    duration_sec: float = 0.0
    actual_backend: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class MomentscanApp(vp.App):
    """981파크 얼굴/장면 분석 앱.

    배치 모드: on_frame()에서 프레임별 결과를 축적하고,
    after_run()에서 per-video 정규화 + peak detection으로 하이라이트를 찾는다.
    """

    fps = DEFAULT_FPS
    backend = DEFAULT_BACKEND

    def __init__(self, *, analyzers=None, output_dir=None):
        self._output_dir = output_dir
        self._frame_records: list = []
        self._identity_records: list = []
        self._interrupted = False
        self._prev_sigint_handler = None

    def configure_modules(self, modules):
        from vpx.sdk.paths import get_models_dir

        names = list(modules) if modules else [
            "face.detect", "face.expression", "face.au", "head.pose",
            "face.parse", "face.quality", "portrait.score", "frame.quality",
            "face.gate",
        ]

        if "face.detect" in names and "face.classify" not in names:
            names.append("face.classify")

        resolved = super().configure_modules(names)

        # Inject centralized models directory into analyzers
        models_dir = get_models_dir()
        for module in resolved:
            if hasattr(module, "_models_dir"):
                module._models_dir = models_dir

        return resolved

    def setup(self):
        self._frame_records = []
        self._identity_records = []
        self._interrupted = False

        # Reset extract module state for video-level isolation
        from momentscan.algorithm.batch.extract import reset_extract_state
        reset_extract_state()

        # SIGINT를 잡아서 graceful shutdown → after_run() 보장
        self._prev_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        """SIGINT를 잡아 on_frame에서 False를 리턴하게 한다.

        이렇게 하면 engine.execute()가 정상 종료하고,
        after_run()에서 축적된 프레임으로 배치 분석이 실행된다.
        두 번째 Ctrl+C는 원래 핸들러로 전달해서 즉시 종료.
        """
        if self._interrupted:
            # 두 번째 Ctrl+C → 즉시 종료
            if self._prev_sigint_handler and callable(self._prev_sigint_handler):
                self._prev_sigint_handler(signum, frame)
            else:
                raise KeyboardInterrupt
        self._interrupted = True
        logger.info("Interrupted — finishing batch analysis with %d frames", len(self._frame_records))

    def on_frame(self, frame, results):
        """프레임별 analyzer 결과를 축적한다.

        SIGINT 수신 시 False를 리턴하여 정상 종료 → after_run() 실행을 보장.

        Args:
            frame: visualpath Frame 객체.
            results: terminal node의 FlowData 리스트.

        Returns:
            True to continue, False to stop (on interrupt).
        """
        if self._interrupted:
            return False

        from momentscan.algorithm.batch.extract import extract_frame_record
        from momentscan.algorithm.identity.extract import extract_identity_record

        record = extract_frame_record(frame, results)
        if record is not None:
            self._frame_records.append(record)

        id_record = extract_identity_record(frame, results)
        if id_record is not None:
            self._identity_records.append(id_record)

        n = len(self._frame_records)
        if n == 1:
            logger.info("Frame collection started")
        elif n % 100 == 0:
            logger.info("Collected %d frame records", n)

        return True

    def after_run(self, result):
        """전체 비디오 기준 배치 분석을 수행한다."""
        from momentscan.algorithm.batch.highlight import BatchHighlightEngine

        n = len(self._frame_records)
        logger.info(
            "Frame collection done — %d records (%.1fs). Starting batch analysis...",
            n, result.duration_sec,
        )

        engine = BatchHighlightEngine()
        highlight_result = engine.analyze(self._frame_records)

        logger.info(
            "Batch analysis complete — %d highlights detected",
            len(highlight_result.windows),
        )

        # Phase 3: Identity builder
        identity_result = None
        if self._identity_records:
            from momentscan.algorithm.identity import IdentityBuilder
            identity_result = IdentityBuilder().build(self._identity_records)
            logger.info(
                "Identity analysis complete — %d persons detected",
                len(identity_result.persons),
            )

        if self._output_dir:
            output_path = Path(self._output_dir)
            highlight_result.export(output_path)

            video_path = getattr(self, 'video', None)
            if video_path is not None:
                # Peak frame 추출
                from momentscan.algorithm.batch.export_frames import export_highlight_frames
                export_highlight_frames(Path(video_path), highlight_result, output_path)

                # Interactive HTML report
                from momentscan.algorithm.batch.export_report import export_highlight_report
                export_highlight_report(Path(video_path), highlight_result, output_path)

            # Identity metadata + crops → bank → HTML report
            if identity_result is not None:
                from momentscan.algorithm.identity.export import export_identity_metadata
                from momentscan.algorithm.identity.export_crops import export_identity_crops
                from momentscan.algorithm.identity.bank_bridge import register_to_bank
                from momentscan.algorithm.identity.export_report import export_identity_report

                export_identity_metadata(identity_result, output_path)
                export_identity_crops(
                    Path(video_path), identity_result,
                    self._identity_records, output_path,
                )
                register_to_bank(
                    identity_result, self._identity_records, output_path,
                )
                export_identity_report(
                    Path(video_path), identity_result,
                    self._identity_records, output_path,
                )

            logger.info("Results exported to %s", self._output_dir)

        return Result(
            highlights=highlight_result.windows,
            identity=identity_result,
            frame_count=result.frame_count,
            duration_sec=result.duration_sec,
            actual_backend=result.actual_backend,
            stats=result.stats,
        )

    def teardown(self):
        self._frame_records = []
        self._identity_records = []
        # SIGINT 핸들러 복원
        if self._prev_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._prev_sigint_handler)
            self._prev_sigint_handler = None


def run(
    video: Union[str, Path],
    *,
    analyzers: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    fps: int = DEFAULT_FPS,
    batch_size: int = 1,
    backend: str = DEFAULT_BACKEND,
    profile: Optional[str] = None,
    isolation: Optional[Any] = None,
    on_frame: Optional[Callable] = None,
) -> Result:
    """Process a video and return highlight results.

    momentscan one-liner: MomentscanApp().run().

    Args:
        video: Path to video file.
        analyzers: Analyzer names. None = use all analyzers.
        output_dir: Directory for highlight output (windows.json, timeseries.csv, frames/).
            None = no file output.
        fps: Frames per second to process.
        batch_size: Batch size for GPU module processing (default: 1).
        backend: Execution backend ("simple", "pathway", or "worker").
        profile: Execution profile ("lite" or "platform").
        isolation: Optional IsolationConfig for explicit module isolation.
        on_frame: Optional per-frame callback ``(frame, terminal_results) -> bool``.

    Returns:
        Result with highlights, frame_count, duration_sec.

    Example:
        >>> result = ms.run("video.mp4")
        >>> result = ms.run("video.mp4", output_dir="./output")
        >>> result = ms.run("video.mp4", analyzers=["face.detect", "body.pose"])
    """
    app = MomentscanApp(
        analyzers=analyzers, output_dir=output_dir,
    )
    return app.run(
        video,
        modules=list(analyzers) if analyzers else None,
        fps=fps,
        batch_size=batch_size,
        backend=backend,
        isolation=isolation,
        profile=profile,
        on_frame=on_frame,
    )
