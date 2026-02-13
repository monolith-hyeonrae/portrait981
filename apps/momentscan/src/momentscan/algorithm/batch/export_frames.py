"""Peak frame image extraction from video.

각 하이라이트 윈도우의 peak frame과 best frame을 비디오에서 추출하여 저장.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Set

import cv2

from momentscan.algorithm.batch.types import HighlightResult

logger = logging.getLogger(__name__)


def export_highlight_frames(
    video_path: Path,
    result: HighlightResult,
    output_dir: Path,
) -> None:
    """하이라이트 윈도우의 peak/best frame을 비디오에서 추출한다.

    Args:
        video_path: 원본 비디오 경로.
        result: BatchHighlightEngine 분석 결과.
        output_dir: 출력 루트 디렉토리. highlight/frames/ 하위에 저장.
    """
    if not result.windows:
        return

    frames_dir = output_dir / "highlight" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 추출할 (timestamp_ms, filename) 쌍 수집
    targets: list[tuple[float, str]] = []
    seen_ms: Set[float] = set()

    for w in result.windows:
        wid = w.window_id

        # peak frame
        if w.peak_ms not in seen_ms:
            targets.append((w.peak_ms, f"w{wid}_peak_{int(w.peak_ms)}ms.jpg"))
            seen_ms.add(w.peak_ms)

        # best frames (selected_frames)
        for sf in w.selected_frames:
            ts_ms = sf["timestamp_ms"]
            if ts_ms not in seen_ms:
                targets.append((ts_ms, f"w{wid}_best_{int(ts_ms)}ms.jpg"))
                seen_ms.add(ts_ms)

    if not targets:
        return

    # timestamp 순 정렬 (seek 효율)
    targets.sort(key=lambda x: x[0])

    # FileSource로 seek + read
    from visualbase.sources.file import FileSource

    source = FileSource(str(video_path))
    source.open()

    saved = 0
    try:
        for ts_ms, filename in targets:
            t_ns = int(ts_ms * 1_000_000)
            if not source.seek(t_ns):
                logger.warning("Seek failed for %.1fms — skipping %s", ts_ms, filename)
                continue

            frame = source.read()
            if frame is None:
                logger.warning("Read failed after seek to %.1fms — skipping %s", ts_ms, filename)
                continue

            out_path = frames_dir / filename
            cv2.imwrite(str(out_path), frame.data)
            saved += 1
    finally:
        source.close()

    logger.info("Exported %d/%d peak/best frames to %s", saved, len(targets), frames_dir)
