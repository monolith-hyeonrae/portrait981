"""전략별 프레임 분류 결과를 시각적으로 비교하는 HTML 리포트 생성.

비디오를 처리하여 각 프레임에 대해 Catalog/LR/XGBoost의 판단을 비교하고,
전략 간 불일치 프레임을 하이라이트한 HTML 리포트를 생성한다.

Usage:
    uv run python scripts/visual_compare.py ~/Videos/reaction_test/test_0.mp4
    uv run python scripts/visual_compare.py ~/Videos/reaction_test/test_0.mp4 --fps 2 --output report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("visual_compare")


def frame_to_base64(frame_bgr, max_width=320):
    """OpenCV frame → base64 encoded JPEG for HTML embedding."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Visual Strategy Comparison")
    parser.add_argument("video", help="mp4 video path")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--catalog", default="data/catalogs/portrait-v1")
    parser.add_argument("--output", "-o", default="visual_compare.html")
    parser.add_argument("--max-frames", type=int, default=200, help="max frames to show")
    args = parser.parse_args()

    from momentscan.algorithm.batch.extract import extract_frame_record
    from momentscan.algorithm.batch.catalog_scoring import (
        SIGNAL_FIELDS, extract_signal_vector,
    )

    # 1. Run momentscan and capture frames + records
    logger.info("Processing: %s (fps=%d)", args.video, args.fps)

    frames_data = []  # [(frame_bgr, record, signal_vec)]

    def on_frame(frame, results):
        record = extract_frame_record(frame, results)
        if record is not None:
            # Get the actual image
            frame_bgr = frame.image if hasattr(frame, 'image') else None
            if frame_bgr is None and hasattr(frame, 'data'):
                frame_bgr = frame.data
            if frame_bgr is not None:
                vec = extract_signal_vector(record, signal_fields=SIGNAL_FIELDS)
                frames_data.append((frame_bgr.copy(), record, vec))
        return True

    import momentscan as ms
    ms.run(args.video, fps=args.fps, backend="simple", on_frame=on_frame)
    logger.info("Collected %d frames with images", len(frames_data))

    if not frames_data:
        logger.error("No frames collected")
        return

    # 2. Build signal matrix
    vectors = np.array([fd[2] for fd in frames_data])

    # 3. Load catalog profiles and score with CatalogStrategy
    from visualbind.profile import load_profiles
    from visualbind.strategies.catalog import CatalogStrategy

    profiles = load_profiles(Path(args.catalog))
    cat_strat = CatalogStrategy(profiles=profiles)
    cat_names = [p.name for p in profiles]

    catalog_results = []
    for vec in vectors:
        scores = cat_strat.predict(vec)
        best = max(scores, key=scores.get)
        catalog_results.append((best, scores[best]))

    # 4. Train XGBoost on catalog assignments
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    y_catalog = [r[0] for r in catalog_results]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_catalog)

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0,
                            use_label_encoder=False, eval_metric="mlogloss")
        xgb.fit(vectors, y_encoded)
        xgb_proba = xgb.predict_proba(vectors)
        xgb_preds = le.inverse_transform(xgb.predict(vectors))
        xgb_available = True
        logger.info("XGBoost trained")
    except ImportError:
        xgb_available = False
        xgb_preds = y_catalog
        logger.warning("XGBoost not available, skipping")

    # LR
    lr = LogisticRegression(max_iter=1000)
    lr.fit(vectors, y_encoded)
    lr_proba = lr.predict_proba(vectors)
    lr_preds = le.inverse_transform(lr.predict(vectors))
    logger.info("LogisticRegression trained")

    # 5. Find disagreements
    disagree_indices = []
    for i in range(len(vectors)):
        if y_catalog[i] != lr_preds[i] or (xgb_available and y_catalog[i] != xgb_preds[i]):
            disagree_indices.append(i)
    logger.info("Disagreement frames: %d / %d (%.1f%%)",
                len(disagree_indices), len(vectors),
                100 * len(disagree_indices) / len(vectors))

    # 6. Generate HTML
    cat_colors = {
        "warm_smile": "#4CAF50",
        "cool_gaze": "#2196F3",
        "cool_expression": "#2196F3",
        "lateral": "#FF9800",
        "playful_face": "#E91E63",
        "wild_energy": "#9C27B0",
    }

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VisualBind Strategy Comparison</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
h1, h2, h3 { color: #e94560; }
.summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }
.frame-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 10px; }
.frame-card { background: #16213e; border-radius: 8px; padding: 10px; }
.frame-card.disagree { border: 2px solid #e94560; }
.frame-card img { width: 100%; border-radius: 4px; }
.label { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
.scores { font-size: 11px; color: #aaa; margin-top: 5px; }
table { border-collapse: collapse; margin: 10px 0; }
td, th { padding: 6px 12px; border: 1px solid #333; }
th { background: #16213e; }
.section { margin: 30px 0; }
</style></head><body>
"""]

    html_parts.append(f"<h1>VisualBind Strategy Comparison</h1>")
    html_parts.append(f"<div class='summary'>")
    html_parts.append(f"<p>Video: {Path(args.video).name} | Frames: {len(vectors)} | FPS: {args.fps}</p>")

    # Category distribution
    from collections import Counter
    cat_dist = Counter(y_catalog)
    html_parts.append(f"<p>Catalog distribution: {dict(cat_dist)}</p>")
    html_parts.append(f"<p>Disagreements: {len(disagree_indices)} ({100*len(disagree_indices)/len(vectors):.1f}%)</p>")
    html_parts.append(f"</div>")

    # Section 1: Disagreement frames
    html_parts.append(f"<div class='section'><h2>Disagreement Frames (Catalog ≠ LR/XGBoost)</h2>")
    html_parts.append(f"<div class='frame-grid'>")

    shown = 0
    for idx in disagree_indices:
        if shown >= args.max_frames // 2:
            break
        frame_bgr, record, vec = frames_data[idx]
        b64 = frame_to_base64(frame_bgr)

        cat_label = y_catalog[idx]
        lr_label = lr_preds[idx]
        xgb_label = xgb_preds[idx] if xgb_available else "N/A"

        cat_color = cat_colors.get(cat_label, "#666")
        lr_color = cat_colors.get(lr_label, "#666")
        xgb_color = cat_colors.get(xgb_label, "#666")

        html_parts.append(f"""
        <div class='frame-card disagree'>
            <img src='data:image/jpeg;base64,{b64}'>
            <div>
                <span class='label' style='background:{cat_color}'>Catalog: {cat_label}</span>
                <span class='label' style='background:{lr_color}'>LR: {lr_label}</span>
                <span class='label' style='background:{xgb_color}'>XGB: {xgb_label}</span>
            </div>
            <div class='scores'>Frame #{idx} |
                happy={vec[SIGNAL_FIELDS.index('em_happy')]:.2f}
                neutral={vec[SIGNAL_FIELDS.index('em_neutral')]:.2f}
            </div>
        </div>""")
        shown += 1

    html_parts.append(f"</div></div>")

    # Section 2: Per-category samples
    html_parts.append(f"<div class='section'><h2>Per-Category Samples (Catalog Assignment)</h2>")
    for cat_name in cat_names:
        indices = [i for i, c in enumerate(y_catalog) if c == cat_name]
        if not indices:
            continue
        html_parts.append(f"<h3 style='color:{cat_colors.get(cat_name, '#fff')}'>{cat_name} ({len(indices)} frames)</h3>")
        html_parts.append(f"<div class='frame-grid'>")
        # Show top-scored frames
        scored = [(catalog_results[i][1], i) for i in indices]
        scored.sort(reverse=True)
        for score, idx in scored[:8]:
            frame_bgr, record, vec = frames_data[idx]
            b64 = frame_to_base64(frame_bgr)
            is_disagree = idx in disagree_indices
            cls = "frame-card disagree" if is_disagree else "frame-card"

            xgb_label = xgb_preds[idx] if xgb_available else "N/A"
            html_parts.append(f"""
            <div class='{cls}'>
                <img src='data:image/jpeg;base64,{b64}'>
                <div>
                    <span class='label' style='background:{cat_colors.get(cat_name, "#666")}'>sim={score:.3f}</span>
                    <span class='label' style='background:{cat_colors.get(lr_preds[idx], "#666")}'>LR: {lr_preds[idx]}</span>
                    <span class='label' style='background:{cat_colors.get(xgb_label, "#666")}'>XGB: {xgb_label}</span>
                </div>
            </div>""")
        html_parts.append(f"</div>")

    html_parts.append(f"</div></body></html>")

    # Write HTML
    output_path = Path(args.output)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Report saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
