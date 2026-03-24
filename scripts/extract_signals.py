"""이미지 폴더에서 signal 추출 → parquet 저장.

SignalExtractor(momentscan DAG 기반)를 사용하여 signal을 추출.
momentscan의 face.quality(마스크 기반) + face.gate(다중 조건) 포함.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("extract_signals")

from visualbind.signals import SIGNAL_FIELDS, normalize_signal


def main():
    parser = argparse.ArgumentParser(description="Extract signals from labeled images")
    parser.add_argument("dataset", help="dataset directory (e.g. data/datasets/portrait-v1)")
    parser.add_argument("--output", "-o", default=None, help="output parquet path")
    parser.add_argument("--limit", type=int, default=0, help="limit number of images (0=all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    output_path = Path(args.output) if args.output else dataset_dir / "signals.parquet"

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)

    # Collect image files
    img_exts = {".jpg", ".jpeg", ".png", ".avif"}
    image_files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in img_exts
    )
    if args.limit > 0:
        image_files = image_files[:args.limit]

    logger.info("Found %d images in %s", len(image_files), images_dir)

    # Load SignalExtractor (momentscan DAG 기반)
    from momentscan.signals import SignalExtractor
    extractor = SignalExtractor()
    extractor.initialize()

    # Extract signals
    import pandas as pd

    rows = []
    no_face = 0
    for i, img_path in enumerate(image_files):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info("Processing %d/%d: %s", i + 1, len(image_files), img_path.name)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Cannot read: %s", img_path.name)
            continue

        result = extractor.extract(img_bgr, frame_id=i)
        if not result.face_detected:
            no_face += 1
            continue

        # Build normalized row
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

    # Summary stats
    for col in list(SIGNAL_FIELDS)[:5]:
        vals = df[col]
        logger.info("  %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f", col, vals.mean(), vals.std(), vals.min(), vals.max())
    logger.info("  ... (%d more fields)", len(SIGNAL_FIELDS) - 5)


if __name__ == "__main__":
    main()
