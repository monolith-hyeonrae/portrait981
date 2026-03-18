"""데이터셋 리뷰 HTML 생성 — 카테고리별 이미지 갤러리.

labels.csv를 읽어서 expression/pose 축별로 이미지를 그룹핑하여 보여준다.

Usage:
    python scripts/review_dataset.py data/datasets/portrait-v1 --output review.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("review")


def img_to_b64(img_path: Path, max_width=280) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def main():
    parser = argparse.ArgumentParser(description="Dataset Review HTML")
    parser.add_argument("dataset", help="dataset directory (e.g. data/datasets/portrait-v1)")
    parser.add_argument("--output", "-o", default="review.html")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"

    if not labels_path.exists():
        logger.error("labels.csv not found in %s", dataset_dir)
        return

    # Load labels
    rows = []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("Loaded %d labels", len(rows))

    # Group by expression
    by_expr = {}
    by_pose = {}
    unlabeled_expr = []
    for row in rows:
        expr = row.get("expression", "")
        pose = row.get("pose", "")
        if expr:
            by_expr.setdefault(expr, []).append(row)
        else:
            unlabeled_expr.append(row)
        if pose:
            by_pose.setdefault(pose, []).append(row)

    colors = {
        "cheese": "#4CAF50", "chill": "#2196F3", "edge": "#FF5722",
        "hype": "#9C27B0", "pass": "#d32f2f",
        "front": "#00BCD4", "angle": "#FF9800", "side": "#795548",
    }

    html = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dataset Review</title>
<style>
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }
h1 { color: #e94560; }
h2 { margin-top: 30px; }
.summary { background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.card { background: #16213e; border-radius: 6px; padding: 6px; text-align: center; }
.card img { width: 100%; border-radius: 4px; }
.card .name { font-size: 11px; color: #888; margin-top: 4px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
.card .tags { margin-top: 4px; }
.tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 1px; }
.section { margin: 20px 0; }
</style></head><body>
<h1>Dataset Review</h1>"""]

    # Summary
    html.append(f"""<div class="summary">
    <b>Total:</b> {len(rows)} images |
    <b>Expression:</b> {', '.join(f'{k}={len(v)}' for k, v in sorted(by_expr.items()))} |
    <b>Pose:</b> {', '.join(f'{k}={len(v)}' for k, v in sorted(by_pose.items()))} |
    <b>Unlabeled expression:</b> {len(unlabeled_expr)}
</div>""")

    # Expression sections
    for expr in ["cheese", "chill", "edge", "hype", "pass"]:
        items = by_expr.get(expr, [])
        if not items:
            continue
        color = colors.get(expr, "#666")
        html.append(f'<div class="section"><h2 style="color:{color}">{expr} ({len(items)})</h2>')
        html.append('<div class="grid">')
        for row in items:
            img_path = images_dir / row["filename"]
            if not img_path.exists():
                continue
            b64 = img_to_b64(img_path)
            if not b64:
                continue
            pose = row.get("pose", "")
            member = row.get("member_id", "")
            tags = ""
            if pose:
                pc = colors.get(pose, "#666")
                tags += f'<span class="tag" style="background:{pc}">{pose}</span>'
            if member:
                tags += f'<span class="tag" style="background:#333">{member}</span>'
            html.append(f"""<div class="card">
                <img src="data:image/jpeg;base64,{b64}">
                <div class="name">{row['filename']}</div>
                <div class="tags">{tags}</div>
            </div>""")
        html.append('</div></div>')

    # Pose sections
    for pose in ["front", "angle", "side"]:
        items = by_pose.get(pose, [])
        if not items:
            continue
        color = colors.get(pose, "#666")
        html.append(f'<div class="section"><h2 style="color:{color}">Pose: {pose} ({len(items)})</h2>')
        html.append('<div class="grid">')
        for row in items:
            img_path = images_dir / row["filename"]
            if not img_path.exists():
                continue
            b64 = img_to_b64(img_path)
            if not b64:
                continue
            expr = row.get("expression", "")
            ec = colors.get(expr, "#666")
            html.append(f"""<div class="card">
                <img src="data:image/jpeg;base64,{b64}">
                <div class="name">{row['filename']}</div>
                <div class="tags"><span class="tag" style="background:{ec}">{expr}</span></div>
            </div>""")
        html.append('</div></div>')

    # Unlabeled
    if unlabeled_expr:
        html.append(f'<div class="section"><h2 style="color:#888">Unlabeled Expression ({len(unlabeled_expr)})</h2>')
        html.append('<div class="grid">')
        for row in unlabeled_expr[:50]:
            img_path = images_dir / row["filename"]
            if not img_path.exists():
                continue
            b64 = img_to_b64(img_path)
            if not b64:
                continue
            html.append(f"""<div class="card">
                <img src="data:image/jpeg;base64,{b64}">
                <div class="name">{row['filename']}</div>
            </div>""")
        html.append('</div></div>')

    html.append("</body></html>")

    out = Path(args.output)
    out.write_text("\n".join(html), encoding="utf-8")
    logger.info("Review: %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
