"""Day 0: VisualBind Go/No-Go 분석.

테스트 비디오에서 momentscan을 실행하고,
21D signal을 추출하여 N_eff와 상관행렬을 분석한다.

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


def extract_signals_from_records(frame_records, signal_fields):
    """FrameRecord 리스트에서 signal vector 행렬을 추출."""
    from momentscan.algorithm.batch.catalog_scoring import extract_signal_vector

    vectors = []
    for record in frame_records:
        vec = extract_signal_vector(record, signal_fields=signal_fields)
        vectors.append(vec)

    if not vectors:
        return np.empty((0, len(signal_fields)))
    return np.stack(vectors)


def run_momentscan(video_path: str, fps: int = 5) -> list:
    """momentscan으로 비디오 처리 → FrameRecord 리스트 반환."""
    from momentscan.algorithm.batch.extract import extract_frame_record

    collected = []

    def on_frame(frame, results):
        record = extract_frame_record(frame, results)
        if record is not None:
            collected.append(record)
        return True

    import momentscan as ms
    ms.run(video_path, fps=fps, backend="simple", on_frame=on_frame)
    return collected


def main():
    parser = argparse.ArgumentParser(description="Day 0: VisualBind Go/No-Go")
    parser.add_argument("videos", nargs="+", help="mp4 video paths")
    parser.add_argument("--fps", type=int, default=5, help="frames per second (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None, help="parquet output path")
    args = parser.parse_args()

    # Signal fields (momentscan 21D: AU 10 + Emotion 4 + Pose 3 + CLIP 4)
    from momentscan.algorithm.batch.catalog_scoring import SIGNAL_FIELDS

    all_records = []

    for video_path in args.videos:
        video = Path(video_path)
        if not video.exists():
            logger.warning("File not found: %s", video)
            continue

        logger.info("Processing: %s (fps=%d)", video.name, args.fps)
        try:
            records = run_momentscan(str(video), fps=args.fps)
            logger.info("  → %d frame records", len(records))
            all_records.extend(records)
        except Exception as e:
            logger.error("  → Failed: %s", e)
            continue

    if not all_records:
        logger.error("No frame records collected. Exiting.")
        sys.exit(1)

    logger.info("Total frame records: %d", len(all_records))

    # Extract 21D signal vectors
    vectors = extract_signals_from_records(all_records, SIGNAL_FIELDS)
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

    # Top correlated pairs
    print("\n### 상관 상위 5쌍 ###")
    n_fields = len(SIGNAL_FIELDS)
    pairs = []
    for i in range(n_fields):
        for j in range(i + 1, n_fields):
            pairs.append((abs(corr[i, j]), SIGNAL_FIELDS[i], SIGNAL_FIELDS[j]))
    pairs.sort(reverse=True)
    for score, f1, f2 in pairs[:5]:
        print(f"  {f1} ↔ {f2}: {score:.3f}")

    # Lowest correlated pairs (most independent)
    print("\n### 가장 독립적인 5쌍 ###")
    for score, f1, f2 in pairs[-5:]:
        print(f"  {f1} ↔ {f2}: {score:.3f}")


if __name__ == "__main__":
    main()
