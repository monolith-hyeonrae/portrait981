"""Collection result export: crops, clips, metadata.

Consolidates identity/export.py, identity/export_crops.py, and
adds clip extraction via visualbase Clipper + Trigger.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from momentscan.algorithm.collection.types import (
    CollectionRecord,
    CollectionResult,
    PersonCollection,
    SelectedFrame,
)

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95


# ── Metadata ──


def export_metadata(
    result: CollectionResult,
    output_dir: Path,
    *,
    highlights: Optional[List] = None,
) -> None:
    """Export collection metadata as collection.json.

    Args:
        result: CollectionEngine result.
        output_dir: Output root directory.
        highlights: Optional highlight windows from BatchHighlightEngine.
    """
    if not result.persons:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    data: Dict = {
        "frame_count": result.frame_count,
        "persons": {},
    }

    for pid, person in result.persons.items():
        all_frames = person.all_frames()
        data["persons"][str(pid)] = {
            "person_id": person.person_id,
            "prototype_frame_idx": person.prototype_frame_idx,
            "catalog_mode": person.catalog_mode,
            "grid_cells": len(person.grid),
            "total_frames": len(all_frames),
            "pose_coverage": person.pose_coverage,
            "category_coverage": person.category_coverage,
            "frames": [_frame_to_dict(f) for f in all_frames],
        }

    if highlights:
        data["highlights"] = [
            {
                "window_id": w.window_id,
                "start_ms": w.start_ms,
                "end_ms": w.end_ms,
                "peak_ms": w.peak_ms,
                "score": round(w.score, 4),
                "reason": w.reason,
                "selected_frames": w.selected_frames,
            }
            for w in highlights
        ]

    meta_path = output_dir / "collection.json"
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total = len([f for p in result.persons.values() for f in p.all_frames()])
    logger.info("Exported collection metadata: %s (%d frames)", meta_path, total)


def _frame_to_dict(frame: SelectedFrame) -> dict:
    """SelectedFrame -> JSON-serializable dict."""
    return {
        "frame_idx": frame.frame_idx,
        "timestamp_ms": frame.timestamp_ms,
        "set_type": frame.set_type,
        "pose_name": frame.pose_name,
        "pivot_name": frame.pivot_name,
        "cell_key": frame.cell_key,
        "quality_score": round(frame.quality_score, 4),
        "cell_score": round(frame.cell_score, 4),
        "catalog_sim": round(frame.catalog_sim, 4),
        "pose_fit": round(frame.pose_fit, 4),
        "stable_score": round(frame.stable_score, 4),
        "face_crop_box": frame.face_crop_box,
        "image_size": frame.image_size,
    }


# ── Crops ──


def export_crops(
    video_path: Path,
    result: CollectionResult,
    records: List[CollectionRecord],
    output_dir: Path,
) -> int:
    """Extract face crops for selected frames.

    Args:
        video_path: Source video path.
        result: CollectionEngine result.
        records: All CollectionRecords (for timestamp lookup).
        output_dir: Output root directory. Crops go to output_dir/crops/.

    Returns:
        Number of crops saved.
    """
    if not result.persons:
        return 0

    try:
        import cv2
    except ImportError:
        logger.warning("opencv not available — skipping crop export")
        return 0

    fidx_to_ts = {r.frame_idx: r.timestamp_ms for r in records}
    total_saved = 0

    for pid, person in result.persons.items():
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        all_frames = person.all_frames()
        if not all_frames:
            continue

        targets: List[Tuple[SelectedFrame, float]] = []
        for f in all_frames:
            ts = fidx_to_ts.get(f.frame_idx, f.timestamp_ms)
            targets.append((f, ts))
        targets.sort(key=lambda x: x[1])

        saved = _extract_crops_from_video(video_path, targets, crops_dir)
        total_saved += saved
        logger.info("Exported %d crops for person %d", saved, pid)

    return total_saved


def _extract_crops_from_video(
    video_path: Path,
    targets: List[Tuple[SelectedFrame, float]],
    crops_dir: Path,
) -> int:
    """Extract face crops from video frames."""
    import cv2
    from visualbase.sources.file import FileSource

    source = FileSource(str(video_path))
    source.open()

    saved = 0
    try:
        for selected_frame, ts_ms in targets:
            t_ns = int(ts_ms * 1_000_000)
            if not source.seek(t_ns):
                continue

            frame = source.read()
            if frame is None:
                continue

            img = frame.data
            f = selected_frame
            prefix = f"{f.cell_key.replace('|', '_')}_{f.frame_idx}"

            if f.face_crop_box is not None:
                head_img = _crop_image(img, f.face_crop_box)
                if head_img is not None:
                    path = crops_dir / f"{prefix}_head.jpg"
                    cv2.imwrite(
                        str(path), head_img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    saved += 1
    finally:
        source.close()

    return saved


def _crop_image(img, box: tuple) -> "Optional[Any]":
    """Crop image with (x1, y1, x2, y2) pixel coords, clamped to bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


# ── Clips ──


def export_clips(
    video_path: Path,
    result: CollectionResult,
    records: List[CollectionRecord],
    output_dir: Path,
    clip_pre_sec: float = 1.0,
    clip_post_sec: float = 1.5,
) -> int:
    """Extract 2-3s clips around selected frames.

    Uses visualbase Clipper + Trigger.point().

    Args:
        video_path: Source video path.
        result: CollectionEngine result.
        records: All CollectionRecords (for timestamp lookup).
        output_dir: Output root directory. Clips go to output_dir/clips/.
        clip_pre_sec: Seconds before event frame.
        clip_post_sec: Seconds after event frame.

    Returns:
        Number of clips saved.
    """
    if not result.persons:
        return 0

    try:
        from visualbase.packaging.clipper import Clipper
        from visualbase.packaging.trigger import Trigger
    except ImportError:
        logger.warning("visualbase.packaging not available — skipping clip export")
        return 0

    clips_dir = output_dir / "clips"
    clipper = Clipper(output_dir=clips_dir, codec="copy")
    saved = 0

    for pid, person in result.persons.items():
        for frame in person.all_frames():
            event_ns = int(frame.timestamp_ms * 1_000_000)
            label = frame.cell_key.replace("|", "_") if frame.cell_key else frame.set_type
            trigger = Trigger.point(
                event_time_ns=event_ns,
                pre_sec=clip_pre_sec,
                post_sec=clip_post_sec,
                label=label,
                score=frame.cell_score,
            )
            safe_key = frame.cell_key.replace("|", "_") if frame.cell_key else "grid"
            clip_result = clipper.extract(
                video_path, trigger,
                output_filename=f"{safe_key}_{frame.frame_idx}.mp4",
            )
            if clip_result.success:
                saved += 1

    if saved > 0:
        logger.info("Exported %d clips to %s", saved, clips_dir)

    return saved
