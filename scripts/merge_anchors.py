"""Anchor ZIP 파일들을 data/anchors/에 병합.

label_tool에서 Export한 ZIP 파일들을 카테고리별 폴더 구조로 병합한다.

Usage:
    python scripts/merge_anchors.py anchors_test2.zip anchors_test3.zip
    python scripts/merge_anchors.py ~/Downloads/anchors_*.zip --output data/anchors
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("merge_anchors")


def main():
    parser = argparse.ArgumentParser(description="Merge anchor ZIPs into folder structure")
    parser.add_argument("zips", nargs="+", help="anchor ZIP files from label_tool")
    parser.add_argument("--output", "-o", default="data/anchors", help="output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for zip_path in args.zips:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.warning("Not found: %s", zip_path)
            continue

        logger.info("Extracting: %s", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # e.g. "warm_smile/test2_0042.jpg" or "metadata.json"
                parts = Path(info.filename).parts
                if len(parts) == 2:
                    category, filename = parts
                    dest = output_dir / category / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(info.filename))
                    total_files += 1
                elif info.filename == "metadata.json":
                    # Skip metadata (or merge later)
                    pass

    # Summary
    categories = sorted(
        d.name for d in output_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    logger.info("Merged %d files into %s", total_files, output_dir)
    for cat in categories:
        count = len(list((output_dir / cat).glob("*.jpg")))
        logger.info("  %s: %d images", cat, count)


if __name__ == "__main__":
    main()
