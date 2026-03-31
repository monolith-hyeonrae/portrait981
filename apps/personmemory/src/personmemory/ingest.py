"""Bridge: momentscan CollectionResult -> per-person MemoryBank.

Ingests selected frames and embeddings from the collection pipeline
into persistent memory banks for each detected person.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from personmemory.bank import MemoryBank
from personmemory.paths import get_bank_dir, get_bank_path
from personmemory.persistence import load_bank, save_bank

logger = logging.getLogger(__name__)

# Yaw bins (5 bins, degrees)
_YAW_BINS = [
    (-5, 5, "[-5,5]"),
    (-30, -5, "[-30,-5]"),
    (5, 30, "[5,30]"),
    (-70, -30, "[-70,-30]"),
    (30, 70, "[30,70]"),
]


def _yaw_to_bin(yaw: float) -> str:
    """Map head yaw (degrees) to a bin label."""
    for lo, hi, label in _YAW_BINS:
        if lo <= yaw <= hi:
            return label
    # Extreme yaw — clamp to outermost bin
    return "[-70,-30]" if yaw < -70 else "[30,70]"


def _expression_bin(record: Any) -> str:
    """Derive expression bin from a CollectionRecord."""
    # Catalog primary label takes precedence
    primary = getattr(record, "catalog_primary", "")
    if primary:
        lower = primary.lower()
        if "smile" in lower or "happy" in lower:
            return "smile"
        if "surprise" in lower:
            return "surprise"
        return "neutral"

    # Fallback: smile_intensity threshold
    smile = getattr(record, "smile_intensity", 0.0)
    if smile > 0.5:
        return "smile"
    if getattr(record, "em_surprise", 0.0) > 0.5:
        return "surprise"
    return "neutral"


@dataclass
class IngestStats:
    """Per-person ingest statistics."""

    person_id: int
    member_id: str
    nodes: int
    frames_total: int
    frames_ingested: int
    frames_skipped: int
    yaw_bins: List[str]
    bank_path: str


@dataclass
class IngestResult:
    """Result of ingest_collection()."""

    banks: Dict[int, MemoryBank] = field(default_factory=dict)
    stats: List[IngestStats] = field(default_factory=list)

    @property
    def total_persons(self) -> int:
        return len(self.banks)

    @property
    def total_nodes(self) -> int:
        return sum(len(b.nodes) for b in self.banks.values())

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.stats:
            return "No persons ingested"
        parts = []
        for s in self.stats:
            parts.append(
                f"person_{s.person_id}: {s.nodes} nodes, "
                f"{s.frames_ingested}/{s.frames_total} frames"
            )
        return f"Memory bank saved — {'; '.join(parts)}"


def save_selected_frames(
    video_path: Any,
    collection_result: Any,
    collection_records: List[Any],
    member_id: str,
) -> Dict[int, str]:
    """Extract selected frames from video and save to bank directory.

    Only person_id=0 is processed. Frames are saved to
    ``~/.portrait981/personmemory/{shard}/{member_id}/frames/``.

    Args:
        video_path: Path to source video file.
        collection_result: CollectionResult from CollectionEngine.
        collection_records: List[CollectionRecord] for metadata lookup.
        member_id: Unique member identifier.

    Returns:
        Dict mapping frame_idx to saved absolute file path.
        Empty dict if cv2/visualbase unavailable or no frames to save.
    """
    if not collection_result or not getattr(collection_result, "persons", None):
        return {}

    person = collection_result.persons.get(0)
    if person is None:
        return {}

    all_frames = person.all_frames()
    if not all_frames:
        return {}

    # Lazy import — graceful fallback if unavailable
    try:
        import cv2
        from visualbase.sources.file import FileSource
    except ImportError:
        logger.debug("cv2/visualbase not available — skipping frame save")
        return {}

    frames_dir = get_bank_dir(member_id) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build seek targets from SelectedFrames
    targets: list[tuple[float, int, Any]] = []  # (timestamp_ms, frame_idx, SelectedFrame)
    for sf in all_frames:
        targets.append((sf.timestamp_ms, sf.frame_idx, sf))

    # Sort by timestamp for efficient sequential seek
    targets.sort(key=lambda x: x[0])

    # Build frame_idx -> CollectionRecord lookup for face_bbox
    fidx_to_record: Dict[int, Any] = {}
    for r in collection_records:
        fidx_to_record[getattr(r, "frame_idx", -1)] = r

    video_stem = Path(str(video_path)).stem
    saved: Dict[int, str] = {}
    manifest: list[dict] = []

    # Load existing manifest for cumulative append
    manifest_path = frames_dir / "frames.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = []

    try:
        source = FileSource(str(video_path))
        source.open()
        try:
            for ts_ms, frame_idx, sf in targets:
                t_ns = int(ts_ms * 1_000_000)
                if not source.seek(t_ns):
                    logger.warning("Seek failed for %.1fms (frame %d)", ts_ms, frame_idx)
                    continue
                frame = source.read()
                if frame is None:
                    logger.warning("Read failed after seek to %.1fms (frame %d)", ts_ms, frame_idx)
                    continue
                cell_key = sf.cell_key.replace("|", "_")
                filename = f"{video_stem}_{cell_key}_{frame_idx}.jpg"
                out_path = frames_dir / filename
                cv2.imwrite(str(out_path), frame.data)
                saved[frame_idx] = str(out_path)

                # Build manifest entry
                record = fidx_to_record.get(frame_idx)
                face_bbox = getattr(record, "face_bbox", None) if record else None
                manifest.append({
                    "file": filename,
                    "frame_idx": frame_idx,
                    "timestamp_ms": ts_ms,
                    "pose_name": sf.pose_name,
                    "category": sf.pivot_name,
                    "cell_key": sf.cell_key,
                    "quality": sf.quality_score,
                    "cell_score": sf.cell_score,
                    "face_bbox": list(face_bbox) if face_bbox is not None else None,
                    "video": video_stem,
                })
        finally:
            source.close()
    except Exception as exc:
        logger.warning("Failed to extract frames from %s: %s", video_path, exc)

    # Save manifest
    if manifest:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    if saved:
        logger.info("Saved %d frames to %s", len(saved), frames_dir)

    return saved


def lookup_frames(
    member_id: str,
    *,
    pose: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 0,
) -> List[dict]:
    """Look up saved frames by pose/category from the frames manifest.

    Args:
        member_id: Member identifier.
        pose: Filter by pose_name (e.g. "left30", "frontal"). None = all.
        category: Filter by category (e.g. "warm_smile"). None = all.
        top_k: Return top-k by cell_score. 0 = all matching.

    Returns:
        List of manifest entries with "path" (absolute) added.
        Sorted by cell_score descending.
    """
    frames_dir = get_bank_dir(member_id) / "frames"
    manifest_path = frames_dir / "frames.json"
    if not manifest_path.exists():
        return []

    try:
        with open(manifest_path) as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    results = []
    for entry in entries:
        if pose is not None and entry.get("pose_name") != pose:
            continue
        if category is not None and entry.get("category") != category:
            continue
        # Add absolute path
        entry_with_path = dict(entry)
        entry_with_path["path"] = str(frames_dir / entry["file"])
        results.append(entry_with_path)

    results.sort(key=lambda e: e.get("cell_score", 0.0), reverse=True)
    if top_k > 0:
        results = results[:top_k]
    return results


def ingest_collection(
    collection_result: Any,
    collection_records: List[Any],
    frame_paths: Dict[int, str],
    member_id: str,
) -> IngestResult:
    """Ingest CollectionResult into a cumulative MemoryBank.

    Only the main person (person_id=0) is ingested. The bank is stored
    at ``~/.portrait981/personmemory/{shard}/{member_id}/memory_bank.json``
    and accumulates data across multiple video sessions.

    Args:
        collection_result: CollectionResult from CollectionEngine.
        collection_records: List[CollectionRecord] for e_id/quality lookup.
        frame_paths: Dict mapping frame_idx to saved image absolute path.
            Typically from ``save_selected_frames()``.
        member_id: Unique member identifier. Used for bank path sharding.

    Returns:
        IngestResult with banks dict and per-person stats.
    """
    if not collection_result or not getattr(collection_result, "persons", None):
        return IngestResult()

    # Build frame_idx -> CollectionRecord lookup
    fidx_to_record: Dict[int, Any] = {}
    for r in collection_records:
        fidx_to_record[r.frame_idx] = r

    result = IngestResult()

    # Only process main person (pid=0)
    person = collection_result.persons.get(0)
    if person is None:
        return IngestResult()

    pid = 0
    bank_path = get_bank_path(member_id)

    # Cumulative: load existing bank or create new
    if bank_path.exists():
        bank = load_bank(bank_path)
        logger.info("Loaded existing bank for %s (%d nodes)", member_id, len(bank.nodes))
    else:
        bank = MemoryBank(person_id=pid, q_update_min=0.0, q_new_min=0.0)

    all_frames = person.all_frames()
    ingested = 0
    skipped = 0

    for sf in all_frames:
        record = fidx_to_record.get(sf.frame_idx)
        if record is None:
            skipped += 1
            continue

        e_id = getattr(record, "e_id", None)
        if e_id is None:
            skipped += 1
            continue

        quality = sf.quality_score if sf.quality_score > 0 else 0.5

        # Build metadata bins
        meta = {
            "yaw": _yaw_to_bin(getattr(record, "head_yaw", 0.0)),
            "expression": _expression_bin(record),
        }

        image_path = frame_paths.get(sf.frame_idx, "")

        bank.update(e_id, quality, meta, image_path)
        ingested += 1

    if bank.nodes:
        save_bank(bank, bank_path)
        result.banks[pid] = bank

        yaw_bins = sorted({
            b for node in bank.nodes
            for b in node.meta_hist.yaw_bins
        })
        stats = IngestStats(
            person_id=pid,
            member_id=member_id,
            nodes=len(bank.nodes),
            frames_total=len(all_frames),
            frames_ingested=ingested,
            frames_skipped=skipped,
            yaw_bins=yaw_bins,
            bank_path=str(bank_path),
        )
        result.stats.append(stats)
        logger.info(
            "Memory bank %s: %d nodes, %d/%d frames → %s",
            member_id, len(bank.nodes), ingested, len(all_frames), bank_path,
        )
    elif skipped > 0:
        logger.warning(
            "Member %s: all %d frames skipped (no embedding or missing record)",
            member_id, skipped,
        )

    return result
