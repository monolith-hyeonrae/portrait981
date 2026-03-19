"""Merge anchor ZIP files into a dataset directory.

Combines label_tool export ZIPs (images/ + labels.csv) into a unified dataset,
deduplicating by filename.

Usage (CLI):
    annotator merge anchors_test2.zip anchors_test3.zip -o data/datasets/portrait-v1

Usage (API):
    from annotator.merge import merge_zips
    merge_zips(["anchors_test2.zip", "anchors_test3.zip"], output_dir="data/datasets/portrait-v1")
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections import Counter
from pathlib import Path

logger = logging.getLogger("annotator.merge")


def merge_zips(
    zip_paths: list[str | Path],
    output_dir: str | Path = "data/datasets/portrait-v1",
) -> Path:
    """Merge anchor ZIP files into a dataset directory.

    Each ZIP should contain ``images/*.jpg`` and ``labels.csv``.
    Deduplicates by filename. Returns the output directory path.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load existing labels.csv
    labels_path = output_dir / "labels.csv"
    existing_rows: list[dict] = []
    existing_files: set[str] = set()
    if labels_path.exists():
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_files.add(row["filename"])
        logger.info("Existing labels: %d rows", len(existing_rows))

    # Merge ZIPs
    new_images = 0
    new_rows: list[dict] = []
    for zp in zip_paths:
        zp = Path(zp)
        if not zp.exists():
            logger.warning("Not found: %s", zp)
            continue

        logger.info("Extracting: %s", zp.name)
        with zipfile.ZipFile(zp, "r") as zf:
            # Extract images
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                if len(parts) == 2 and parts[0] == "images":
                    filename = parts[1]
                    if filename in existing_files:
                        logger.debug("Skip duplicate: %s", filename)
                        continue
                    dest = images_dir / filename
                    dest.write_bytes(zf.read(info.filename))
                    existing_files.add(filename)
                    new_images += 1

            # Parse labels.csv from ZIP
            if "labels.csv" in zf.namelist():
                csv_bytes = zf.read("labels.csv").decode("utf-8")
                reader = csv.DictReader(io.StringIO(csv_bytes))
                for row in reader:
                    if row["filename"] not in {r["filename"] for r in existing_rows}:
                        new_rows.append(row)

            # Parse video_meta.json → append to videos.csv
            if "video_meta.json" in zf.namelist():
                import json as _json
                meta = _json.loads(zf.read("video_meta.json").decode("utf-8"))
                videos_path = output_dir / "videos.csv"
                existing_videos = set()
                if videos_path.exists():
                    with open(videos_path) as vf:
                        for vrow in csv.DictReader(vf):
                            existing_videos.add(vrow.get("workflow_id", ""))
                vid = meta.get("workflow_id", "")
                if vid and vid not in existing_videos:
                    v_fields = ["workflow_id", "scene", "gender", "ethnicity", "n_persons", "notes"]
                    write_header = not videos_path.exists()
                    with open(videos_path, "a", newline="") as vf:
                        writer = csv.DictWriter(vf, fieldnames=v_fields)
                        if write_header:
                            writer.writeheader()
                        n_p = "2" if meta.get("scene") == "duo" else "1"
                        writer.writerow({
                            "workflow_id": vid, "scene": meta.get("scene", ""),
                            "gender": meta.get("gender", ""), "ethnicity": meta.get("ethnicity", ""),
                            "n_persons": n_p, "notes": "",
                        })
                    logger.info("  Video meta added: %s (%s)", vid, meta.get("scene", ""))

    # Append new rows and save
    all_rows = existing_rows + new_rows
    fieldnames = ["filename", "workflow_id", "expression", "pose", "chemistry", "source"]
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    expr_c = Counter(r["expression"] for r in all_rows if r.get("expression"))
    pose_c = Counter(r["pose"] for r in all_rows if r.get("pose"))
    logger.info("Merged: +%d images, +%d labels", new_images, len(new_rows))
    logger.info("Total: %d images, %d labels", len(list(images_dir.glob("*"))), len(all_rows))
    logger.info("Expression: %s", dict(expr_c.most_common()))
    logger.info("Pose: %s", dict(pose_c.most_common()))

    return output_dir
