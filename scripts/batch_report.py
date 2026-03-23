"""일일 배치 리포트 — 여러 비디오의 XGBoost 예측을 요약.

매일 운영 데이터 N건을 배치로 처리하고, 모델 성능 추적 + 리뷰 대상 식별.

Usage:
    uv run python scripts/batch_report.py ~/Videos/batch_20260323/
    uv run python scripts/batch_report.py ~/Videos/batch_20260323/ --dataset data/datasets/portrait-v1
    uv run python scripts/batch_report.py ~/Videos/batch_20260323/ --output daily_report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_report")


def extract_shoot_thumbnails(video_dir: Path, predictions: dict, fps: int = 2, max_w: int = 200) -> dict:
    """Extract thumbnails for SHOOT frames from videos. Returns {filename: b64}."""
    thumbnails = {}
    for wf_id, preds in predictions.items():
        shoot_indices = {}
        for p in preds:
            if p["expression"] != "cut":
                # Parse frame index from filename: {stem}_{idx:04d}.jpg
                parts = p["filename"].rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[1].split(".")[0])
                        shoot_indices[idx] = p["filename"]
                    except ValueError:
                        pass

        if not shoot_indices:
            continue

        # Find video file
        video_path = video_dir / f"{wf_id}.mp4"
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps))
        frame_idx = 0
        extracted_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                if extracted_idx in shoot_indices:
                    h, w = frame.shape[:2]
                    if w > max_w:
                        s = max_w / w
                        frame = cv2.resize(frame, (max_w, int(h * s)))
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    thumbnails[shoot_indices[extracted_idx]] = base64.b64encode(buf).decode()
                extracted_idx += 1
            frame_idx += 1

        cap.release()
        logger.info("Extracted %d thumbnails from %s", len([k for k in thumbnails if k.startswith(wf_id)]), wf_id)

    return thumbnails


def load_predictions(pred_path: Path, model: str | None = None) -> dict[str, list[dict]]:
    """Load predictions.csv, group by workflow_id. Optionally filter by model."""
    if not pred_path.exists():
        return {}
    by_workflow = {}
    with open(pred_path, newline="") as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            wf = r["workflow_id"]
            by_workflow.setdefault(wf, []).append(r)
    return by_workflow


def load_labels(labels_path: Path) -> dict[str, dict]:
    """Load labels.csv, return {filename: row}."""
    if not labels_path.exists():
        return {}
    with open(labels_path, newline="") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Daily batch prediction report")
    parser.add_argument("videos", help="video directory for batch processing")
    parser.add_argument("--dataset", "-d", default="data/datasets/portrait-v1")
    parser.add_argument("--model", default="models/bind_v4.pkl")
    parser.add_argument("--pose-model", default="models/pose_v2.pkl")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--skip-predict", action="store_true", help="use existing predictions.csv")
    args = parser.parse_args()

    video_dir = Path(args.videos)
    dataset_dir = Path(args.dataset)
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = Path(args.output) if args.output else Path(f"batch_report_{today}.html")

    # Step 1: Run predictions (unless --skip-predict)
    if not args.skip_predict:
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            print(f"No mp4 files in {video_dir}")
            sys.exit(1)
        logger.info("Running predictions on %d videos...", len(videos))
        from predict_report import main as predict_main
        predict_main([str(video_dir), "--dataset", str(dataset_dir),
                      "--model", args.model, "--pose-model", args.pose_model,
                      "--fps", str(args.fps), "--no-html"])

    # Step 2: Load predictions + labels
    pred_path = dataset_dir / "predictions.csv"
    labels_path = dataset_dir / "labels.csv"

    model_meta_path = Path(args.model).with_suffix(".json")
    model_version = "unknown"
    if model_meta_path.exists():
        with open(model_meta_path) as f:
            model_version = json.load(f).get("version", "unknown")

    predictions = load_predictions(pred_path, model=model_version)
    labels = load_labels(labels_path)

    # Extract thumbnails for SHOOT frames
    logger.info("Extracting SHOOT thumbnails...")
    thumbnails = extract_shoot_thumbnails(video_dir, predictions, fps=args.fps)

    if not predictions:
        print("No predictions found")
        sys.exit(1)

    # Step 3: Analyze
    COLORS = {
        'cheese': '#4CAF50', 'goofy': '#E91E63', 'chill': '#2196F3',
        'edge': '#FF5722', 'hype': '#9C27B0', 'cut': '#d32f2f',
        'occluded': '#795548',
    }

    # Per-video stats
    video_stats = []
    total_frames = 0
    total_shoot = 0
    total_expr = Counter()
    total_confidence = []
    low_confidence = []  # frames with confidence < 0.6
    mismatches = []

    for wf_id, preds in sorted(predictions.items()):
        n = len(preds)
        expr_counts = Counter(p["expression"] for p in preds)
        shoot = sum(1 for p in preds if p["expression"] != "cut")
        confs = [float(p["confidence"]) for p in preds]
        avg_conf = sum(confs) / len(confs) if confs else 0

        total_frames += n
        total_shoot += shoot
        total_expr += expr_counts
        total_confidence.extend(confs)

        # Low confidence frames
        for p in preds:
            conf = float(p["confidence"])
            if conf < 0.6 and p["expression"] != "cut":
                low_confidence.append({**p, "workflow_id": wf_id})

        # Mismatches with manual labels
        for p in preds:
            lbl = labels.get(p["filename"])
            if lbl and lbl.get("expression") and lbl["expression"] != p["expression"]:
                mismatches.append({
                    "filename": p["filename"],
                    "workflow_id": wf_id,
                    "manual": lbl["expression"],
                    "predicted": p["expression"],
                    "confidence": p["confidence"],
                })

        video_stats.append({
            "workflow_id": wf_id,
            "total": n,
            "shoot": shoot,
            "shoot_pct": shoot / n * 100 if n else 0,
            "avg_conf": avg_conf,
            "expr": dict(expr_counts),
        })

    # Step 4: Generate HTML report
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Batch Report — {today}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; margin: 20px; max-width: 1200px; }}
h1 {{ color: #e94560; }}
h2 {{ margin-top: 24px; color: #444; }}
.summary {{ background: #fff; padding: 16px; border-radius: 8px; border: 1px solid #e0e0e0; margin: 12px 0; }}
table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
th, td {{ padding: 6px 12px; border: 1px solid #e0e0e0; text-align: right; }}
th {{ background: #f5f5f5; text-align: left; }}
.bar {{ display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: #eee; }}
.bar-seg {{ height: 100%; min-width: 2px; }}
.tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; color: #fff; margin: 1px; }}
.warn {{ color: #FF9800; }}
.good {{ color: #4CAF50; }}
</style>
</head><body>
<h1>Batch Report — {today}</h1>
<div class="summary">
    <b>Model:</b> {model_version} | <b>Videos:</b> {len(video_stats)} |
    <b>Total frames:</b> {total_frames} | <b>SHOOT:</b> {total_shoot} ({total_shoot/max(total_frames,1)*100:.0f}%) |
    <b>Avg confidence:</b> {sum(total_confidence)/max(len(total_confidence),1):.1%} |
    <b>Low confidence:</b> <span class="warn">{len(low_confidence)}</span> |
    <b>Mismatches:</b> <span class="warn">{len(mismatches)}</span>
</div>
"""

    # Overall distribution bar
    html += '<div class="summary"><b>Expression Distribution</b><div class="bar" style="margin-top:8px">'
    for expr in ['cheese', 'goofy', 'chill', 'edge', 'hype', 'cut']:
        n = total_expr.get(expr, 0)
        pct = n / max(total_frames, 1) * 100
        if n > 0:
            color = COLORS.get(expr, '#999')
            html += f'<div class="bar-seg" style="width:{max(pct,1):.1f}%;background:{color}" title="{expr}: {n}"></div>'
    html += '</div>'
    for expr in ['cheese', 'goofy', 'chill', 'edge', 'hype', 'cut']:
        n = total_expr.get(expr, 0)
        if n > 0:
            html += f'<span class="tag" style="background:{COLORS.get(expr,"#999")}">{expr} {n}</span> '
    html += '</div>'

    # Per-video table
    html += '<h2>Per-Video Summary</h2>'
    html += '<table><tr><th>workflow_id</th><th>Frames</th><th>SHOOT</th><th>%</th><th>Avg Conf</th><th>Distribution</th></tr>'
    for v in video_stats:
        bar = ''
        for expr in ['cheese', 'goofy', 'chill', 'edge', 'hype']:
            n = v["expr"].get(expr, 0)
            if n > 0:
                pct = n / v["total"] * 100
                bar += f'<div class="bar-seg" style="width:{max(pct,1):.1f}%;background:{COLORS.get(expr,"#999")}"></div>'
        html += f'<tr><td style="text-align:left">{v["workflow_id"]}</td><td>{v["total"]}</td><td>{v["shoot"]}</td>'
        html += f'<td>{v["shoot_pct"]:.0f}%</td><td>{v["avg_conf"]:.1%}</td>'
        html += f'<td><div class="bar">{bar}</div></td></tr>'
    html += '</table>'

    # SHOOT gallery per video
    html += '<h2>SHOOT Frames</h2>'
    for v in video_stats:
        wf = v["workflow_id"]
        shoot_preds = [p for p in predictions.get(wf, []) if p["expression"] != "cut"]
        if not shoot_preds:
            continue
        html += f'<div class="summary"><b>{wf}</b> — {len(shoot_preds)} SHOOT'
        html += '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">'
        for p in sorted(shoot_preds, key=lambda x: -float(x["confidence"])):
            b64 = thumbnails.get(p["filename"])
            if not b64:
                continue
            color = COLORS.get(p["expression"], "#999")
            conf = float(p["confidence"])
            html += f'''<div style="text-align:center;width:120px">
                <img src="data:image/jpeg;base64,{b64}" style="width:120px;border-radius:4px;border:2px solid {color}">
                <div style="font-size:10px;margin-top:2px">
                    <span class="tag" style="background:{color}">{p["expression"]}</span>
                    <span style="color:#888">{conf:.0%}</span>
                </div>
            </div>'''
        html += '</div></div>'

    # Low confidence frames
    if low_confidence:
        html += f'<h2>Low Confidence SHOOT ({len(low_confidence)})</h2>'
        html += '<div class="summary"><p style="font-size:12px;color:#888">Confidence &lt; 60% — 리뷰 대상</p>'
        html += '<table><tr><th>Filename</th><th>Workflow</th><th>Expression</th><th>Confidence</th></tr>'
        for lc in sorted(low_confidence, key=lambda x: float(x["confidence"]))[:50]:
            color = COLORS.get(lc["expression"], "#999")
            html += f'<tr><td style="text-align:left">{lc["filename"]}</td><td>{lc["workflow_id"]}</td>'
            html += f'<td><span class="tag" style="background:{color}">{lc["expression"]}</span></td>'
            html += f'<td class="warn">{float(lc["confidence"]):.1%}</td></tr>'
        html += '</table></div>'

    # Mismatches
    if mismatches:
        html += f'<h2>Mismatches with Manual Labels ({len(mismatches)})</h2>'
        html += '<div class="summary"><p style="font-size:12px;color:#888">수동 라벨 ≠ 모델 예측 — 모델 개선 포인트</p>'
        html += '<table><tr><th>Filename</th><th>Manual</th><th>Predicted</th><th>Confidence</th></tr>'
        for m in mismatches:
            mc = COLORS.get(m["manual"], "#999")
            pc = COLORS.get(m["predicted"], "#999")
            html += f'<tr><td style="text-align:left">{m["filename"]}</td>'
            html += f'<td><span class="tag" style="background:{mc}">{m["manual"]}</span></td>'
            html += f'<td><span class="tag" style="background:{pc}">{m["predicted"]}</span></td>'
            html += f'<td>{float(m["confidence"]):.1%}</td></tr>'
        html += '</table></div>'

    # Daily action items
    html += '<h2>Action Items</h2><div class="summary">'
    if low_confidence:
        html += f'<p>🔍 <b>{len(low_confidence)}건</b> low confidence 프레임 리뷰 필요</p>'
    if mismatches:
        html += f'<p>⚠️ <b>{len(mismatches)}건</b> mismatch — 수동 라벨 또는 모델 점검</p>'
    shoot_rate = total_shoot / max(total_frames, 1) * 100
    if shoot_rate > 50:
        html += f'<p class="warn">📊 SHOOT 비율 {shoot_rate:.0f}% — 너무 높음, gate threshold 조정 검토</p>'
    elif shoot_rate < 5:
        html += f'<p class="warn">📊 SHOOT 비율 {shoot_rate:.0f}% — 너무 낮음, 모델 또는 데이터 검토</p>'
    else:
        html += f'<p class="good">📊 SHOOT 비율 {shoot_rate:.0f}% — 적정 범위</p>'
    html += '</div>'

    html += '</body></html>'

    output_path.write_text(html, encoding="utf-8")
    logger.info("Batch report saved: %s", output_path)
    logger.info("Summary: %d videos, %d frames, %d SHOOT (%.0f%%), %d low-conf, %d mismatches",
                len(video_stats), total_frames, total_shoot, shoot_rate, len(low_confidence), len(mismatches))


if __name__ == "__main__":
    main()
