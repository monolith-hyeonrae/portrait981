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
        collection: CollectionResult (unified identity + impact selection).
        frame_count: Total frames processed.
        duration_sec: Video duration in seconds.
        actual_backend: Backend that was actually used.
        stats: Backend execution statistics.
    """

    highlights: List[Any] = field(default_factory=list)
    collection: Optional[Any] = None
    identity: Optional[Any] = None  # deprecated — use collection
    bank: Optional[Any] = None  # IngestResult from momentbank
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

    def __init__(self, *, analyzers=None, output_dir=None, collection_path=None, member_id=None):
        self._output_dir = output_dir
        self._collection_path = collection_path
        self._member_id = member_id
        self._frame_records: list = []
        self._collection_records: list = []
        self._interrupted = False
        self._prev_sigint_handler = None

    def configure_modules(self, modules):
        from vpx.sdk.paths import get_models_dir

        names = list(modules) if modules else [
            "face.detect", "face.expression", "face.au", "head.pose",
            "face.parse", "face.quality", "portrait.score", "frame.quality",
            "face.baseline", "face.gate",
        ]

        if "face.detect" in names and "face.classify" not in names:
            names.append("face.classify")

        resolved = super().configure_modules(names)

        # Inject centralized models directory into analyzers
        models_dir = get_models_dir()
        for module in resolved:
            if hasattr(module, "_models_dir"):
                module._models_dir = models_dir

        # Inject catalog CLIP axes into portrait.score analyzer
        # (must happen after modules are resolved, not in setup())
        if self._collection_path and self._clip_axes:
            for module in resolved:
                if getattr(module, "name", "") == "portrait.score":
                    module._clip_axes = self._clip_axes
                    logger.info(
                        "Injected %d catalog CLIP axes into portrait.score",
                        len(self._clip_axes),
                    )
                    break

        return resolved

    def setup(self):
        self._frame_records = []
        self._collection_records = []
        self._ingest_result = None
        self._interrupted = False
        self._clip_axes = None

        # Reset extract module state for video-level isolation
        from momentscan.algorithm.batch.extract import reset_extract_state
        from momentscan.algorithm.collection.extract import (
            reset_extract_state as reset_collection_extract_state,
        )
        reset_extract_state()
        reset_collection_extract_state()

        # Signal-profile catalog loading (from collection directory)
        if self._collection_path:
            from momentscan.algorithm.batch.catalog_scoring import (
                load_profiles, load_clip_axes,
            )
            from momentscan.algorithm.batch.extract import set_catalog_profiles
            from momentscan.algorithm.collection.extract import (
                set_catalog_profiles as set_collection_catalog_profiles,
            )
            catalog_path = Path(self._collection_path)
            profiles = load_profiles(catalog_path)
            set_catalog_profiles(profiles)
            set_collection_catalog_profiles(profiles)
            logger.info("Loaded %d signal-profile catalog categories", len(profiles))

            # Store CLIP axes for injection in configure_modules()
            self._clip_axes = load_clip_axes(catalog_path)

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
        from momentscan.algorithm.collection.extract import extract_collection_record

        # Batch highlight records (kept for timeseries CSV + peak detection)
        record = extract_frame_record(frame, results)
        if record is not None:
            self._frame_records.append(record)

        # Unified collection records (replaces identity_records)
        coll_record = extract_collection_record(frame, results)
        if coll_record is not None:
            self._collection_records.append(coll_record)

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

        # Batch highlight (kept for timeseries + peak detection)
        engine = BatchHighlightEngine()
        highlight_result = engine.analyze(self._frame_records)

        logger.info(
            "Batch analysis complete — %d highlights detected",
            len(highlight_result.windows),
        )

        # Unified collection engine (replaces IdentityBuilder)
        collection_result = None
        n_coll = len(self._collection_records)
        if n_coll > 0:
            from momentscan.algorithm.collection.engine import CollectionEngine
            coll_engine = CollectionEngine.from_collection_path(self._collection_path)

            # Diagnostic: count records with embeddings
            n_with_eid = sum(1 for r in self._collection_records if r.e_id is not None)
            logger.info(
                "Collection: %d records (%d with embedding)",
                n_coll, n_with_eid,
            )

            collection_result = coll_engine.collect(self._collection_records)
            logger.info(
                "Collection analysis complete — %d persons detected",
                len(collection_result.persons),
            )
        else:
            logger.info("No collection records — skipping collection analysis")

        if self._output_dir:
            output_path = Path(self._output_dir)
            highlight_result.export(output_path)

            video_path = getattr(self, 'video', None)

            # Collection exports: metadata + clips
            if collection_result is not None and video_path is not None:
                from momentscan.algorithm.collection.export import (
                    export_metadata,
                    export_clips,
                )
                export_metadata(
                    collection_result, output_path,
                    highlights=highlight_result.windows,
                )
                export_clips(
                    Path(video_path), collection_result,
                    self._collection_records, output_path,
                )

                # Bank ingest: save frames + ingest (optional)
                try:
                    from momentbank.ingest import (
                        save_selected_frames,
                        ingest_collection,
                    )
                    # Resolve member_id: explicit > video stem fallback
                    mid = self._member_id
                    if mid is None and video_path:
                        mid = Path(video_path).stem
                    if mid is None:
                        mid = "unknown"
                    frame_paths = save_selected_frames(
                        video_path, collection_result,
                        self._collection_records, mid,
                    )
                    ingest_result = ingest_collection(
                        collection_result,
                        self._collection_records,
                        frame_paths=frame_paths,
                        member_id=mid,
                    )
                    if ingest_result.stats:
                        logger.info(ingest_result.summary())
                    else:
                        logger.info(
                            "Memory bank: no data ingested "
                            "(persons=%d, check e_id/quality thresholds)",
                            len(collection_result.persons),
                        )
                    self._ingest_result = ingest_result
                except ImportError:
                    logger.debug("momentbank not installed — skipping bank ingest")

            # Unified report (Timeline + Collection tabs)
            if video_path is not None:
                from momentscan.algorithm.report import export_report
                export_report(
                    Path(video_path),
                    highlight_result=highlight_result,
                    collection_result=collection_result,
                    collection_records=self._collection_records,
                    output_dir=output_path,
                )

            logger.info("Results exported to %s", self._output_dir)

        return Result(
            highlights=highlight_result.windows,
            collection=collection_result,
            identity=collection_result,  # backward compat
            bank=getattr(self, "_ingest_result", None),
            frame_count=result.frame_count,
            duration_sec=result.duration_sec,
            actual_backend=result.actual_backend,
            stats=result.stats,
        )

    def teardown(self):
        self._frame_records = []
        self._collection_records = []
        self._ingest_result = None
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
    collection_path: Optional[Union[str, Path]] = None,
    member_id: Optional[str] = None,
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
        collection_path: Path to collection/catalog directory (e.g. catalogs/portrait-v1).
            Loads signal profiles and pose/pivot definitions.
            None = use built-in poses × AU-rule classification.
        member_id: Unique member identifier for cumulative bank storage.
            None = use video file stem as fallback.

    Returns:
        Result with highlights, collection, frame_count, duration_sec.

    Example:
        >>> result = ms.run("video.mp4")
        >>> result = ms.run("video.mp4", output_dir="./output")
        >>> result = ms.run("video.mp4", collection_path="catalogs/portrait-v1")
    """
    app = MomentscanApp(
        analyzers=analyzers, output_dir=output_dir,
        collection_path=str(collection_path) if collection_path else None,
        member_id=member_id,
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
