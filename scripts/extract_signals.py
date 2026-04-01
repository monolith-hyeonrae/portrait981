"""이미지 폴더에서 signal 추출 → parquet 저장.

ms.extract_signals()로 FlowGraph 정상 경로 사용 (65D).

Usage:
    uv run python scripts/extract_signals.py data/datasets/portrait-v1
    uv run python scripts/extract_signals.py data/datasets/portrait-v1 --limit 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_signals")

from visualbind.signals import SIGNAL_FIELDS, normalize_signal


def main():
    parser = argparse.ArgumentParser(description="Extract signals from labeled images")
    parser.add_argument("dataset", help="dataset directory")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    output_path = Path(args.output) if args.output else dataset_dir / "signals.parquet"

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)

    img_exts = {".jpg", ".jpeg", ".png", ".avif"}
    image_files = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts)
    if args.limit > 0:
        image_files = image_files[:args.limit]
    logger.info("Found %d images in %s", len(image_files), images_dir)

    import momentscan as ms

    import pandas as pd
    rows = []
    no_face = 0
    for i, img_path in enumerate(image_files):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info("Processing %d/%d: %s", i + 1, len(image_files), img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        result = ms.extract_signals(img)
        if not result.face_detected:
            no_face += 1
            continue
        row = {"filename": img_path.name}
        for f in SIGNAL_FIELDS:
            row[f] = normalize_signal(result.signals.get(f, 0.0), f)
        rows.append(row)

    if not rows:
        logger.error("No signals extracted")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d signals (%dD) to %s (no_face=%d)", len(df), len(SIGNAL_FIELDS), output_path, no_face)


if __name__ == "__main__":
    main()
