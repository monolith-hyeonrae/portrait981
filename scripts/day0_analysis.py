"""Day 0: VisualBind Go/No-Go 분석.

테스트 비디오에서 momentscan v2를 실행하고,
65D signal을 추출하여 N_eff와 상관행렬을 분석한다.

Usage:
    uv run python scripts/day0_analysis.py ~/Videos/reaction_test/test_0.mp4
    uv run python scripts/day0_analysis.py ~/Videos/reaction_test/*.mp4
    uv run python scripts/day0_analysis.py ~/Videos/reaction_test/*.mp4 --fps 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("day0")


def main():
    parser = argparse.ArgumentParser(description="Day 0: VisualBind Go/No-Go")
    parser.add_argument("videos", nargs="+", help="mp4 video paths")
    parser.add_argument("--fps", type=int, default=5, help="frames per second (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None, help="parquet output path")
    args = parser.parse_args()

    from visualbind.signals import SIGNAL_FIELDS_EXTENDED, normalize_signal
    SIGNAL_FIELDS = SIGNAL_FIELDS_EXTENDED

    import momentscan as ms

    all_signals = []

    for video_path in args.videos:
        video = Path(video_path)
        if not video.exists():
            logger.warning("File not found: %s", video)
            continue

        logger.info("Processing: %s (fps=%d)", video.name, args.fps)
        try:
            results = ms.run(str(video), fps=args.fps)
            face_results = [r for r in results if r.face_detected]
            logger.info("  → %d frames (%d face detected)", len(results), len(face_results))
            for r in face_results:
                vec = np.array([
                    normalize_signal(r.signals.get(f, 0.0), f)
                    for f in SIGNAL_FIELDS
                ], dtype=np.float64)
                all_signals.append(vec)
        except Exception as e:
            logger.error("  → Failed: %s", e)
            continue

    if not all_signals:
        logger.error("No signals collected. Exiting.")
        sys.exit(1)

    vectors = np.stack(all_signals)
    logger.info("Signal matrix shape: %s", vectors.shape)

    # Save to parquet (optional)
    if args.output:
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {field: vectors[:, i] for i, field in enumerate(SIGNAL_FIELDS)}
        table = pa.table(columns)
        pq.write_table(table, args.output)
        logger.info("Saved to: %s", args.output)

    # Run visualbind Day 0 analysis
    from visualbind.analyzer import (
        compute_correlation_matrix,
        compute_neff,
        generate_report,
    )

    report = generate_report(vectors, SIGNAL_FIELDS)
    print("\n" + report)

    corr = compute_correlation_matrix(vectors)
    neff = compute_neff(corr)

    # Go/No-Go 판단
    print("\n### Go/No-Go 판단 ###")
    if neff >= 3:
        print(f"  N_eff = {neff:.2f} ≥ 3 → ✅ MVP 진행")
    elif neff >= 2:
        print(f"  N_eff = {neff:.2f} (2~3) → ⚠️ 합의 기준 상향 + 진행")
    else:
        print(f"  N_eff = {neff:.2f} < 2 → ❌ 근본적 재고 필요")


if __name__ == "__main__":
    main()
