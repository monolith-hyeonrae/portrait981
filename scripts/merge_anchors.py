"""Anchor ZIP 파일들을 data/datasets/portrait-v1/에 병합.

label_tool에서 Export한 ZIP(images/ + labels.csv)을 기존 데이터셋에 병합.

Usage:
    python scripts/merge_anchors.py anchors_test2.zip anchors_test3.zip
    python scripts/merge_anchors.py ~/Downloads/anchors_*.zip -o data/datasets/portrait-v1
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("merge_anchors")


def main():
    parser = argparse.ArgumentParser(description="Merge anchor ZIPs into dataset")
    parser.add_argument("zips", nargs="+", help="anchor ZIP files from label_tool")
    parser.add_argument("--output", "-o", default="data/datasets/portrait-v1",
                        help="dataset directory (default: data/datasets/portrait-v1)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load existing labels.csv
    labels_path = output_dir / "labels.csv"
    existing_rows = []
    existing_files = set()
    if labels_path.exists():
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_files.add(row["filename"])
        logger.info("Existing labels: %d rows", len(existing_rows))

    # Merge ZIPs
    new_images = 0
    new_rows = []
    for zip_path in args.zips:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.warning("Not found: %s", zip_path)
            continue

        logger.info("Extracting: %s", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
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

    # Append new rows and save
    all_rows = existing_rows + new_rows
    fieldnames = ["filename", "member_id", "expression", "pose", "source"]
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    from collections import Counter
    expr_c = Counter(r["expression"] for r in all_rows if r.get("expression"))
    pose_c = Counter(r["pose"] for r in all_rows if r.get("pose"))
    logger.info("Merged: +%d images, +%d labels", new_images, len(new_rows))
    logger.info("Total: %d images, %d labels", len(list(images_dir.glob("*"))), len(all_rows))
    logger.info("Expression: %s", dict(expr_c.most_common()))
    logger.info("Pose: %s", dict(pose_c.most_common()))


if __name__ == "__main__":
    main()
