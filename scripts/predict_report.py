"""비디오에 XGBoost 모델을 적용하여 추론 리포트 생성.

비디오 → 프레임 추출 → 45D signal → XGBoost 예측 → HTML 리포트

Usage:
    uv run python scripts/predict_report.py ~/Videos/reaction_test/test_3.mp4
    uv run python scripts/predict_report.py ~/Videos/reaction_test/test_3.mp4 --model models/bind_v3.pkl
    uv run python scripts/predict_report.py ~/Videos/reaction_test/test_3.mp4 --fps 3 --output report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_report")

from visualbind.signals import SIGNAL_FIELDS, normalize_signal


def extract_frames(video_path: Path, fps: int, max_frames: int):
    """Extract frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0 and len(frames) < max_frames:
            frames.append((idx, frame.copy()))
        idx += 1
    cap.release()
    return frames


def frame_to_b64(img, max_w=320):
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (max_w, int(h * s)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def main():
    parser = argparse.ArgumentParser(description="XGBoost prediction report for video")
    parser.add_argument("video", help="mp4 video path")
    parser.add_argument("--model", default="models/bind_v3.pkl", help="expression XGBoost model")
    parser.add_argument("--pose-model", default="models/pose_v1.pkl", help="pose XGBoost model")
    parser.add_argument("--meta", default=None, help="model meta JSON path")
    parser.add_argument("--fps", type=int, default=2, help="frames per second")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--output", "-o", default=None, help="output HTML path")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    meta_path = Path(args.meta) if args.meta else model_path.with_suffix(".json")
    output_path = Path(args.output) if args.output else Path(f"report_{video_path.stem}.html")

    # Load models
    import joblib
    logger.info("Loading expression model: %s", model_path)
    clf = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    feature_names = meta["feature_names"]
    logger.info("Expression: %d classes %s", len(classes), classes)

    pose_path = Path(args.pose_model)
    pose_meta_path = pose_path.with_suffix(".json")
    clf_pose = None
    pose_classes = []
    if pose_path.exists():
        clf_pose = joblib.load(pose_path)
        with open(pose_meta_path) as f:
            pose_meta = json.load(f)
        pose_classes = pose_meta["classes"]
        logger.info("Pose: %d classes %s", len(pose_classes), pose_classes)
    else:
        logger.warning("Pose model not found: %s", pose_path)

    # Extract frames
    logger.info("Extracting frames from %s (fps=%d)", video_path, args.fps)
    frames = extract_frames(video_path, args.fps, args.max_frames)
    logger.info("Extracted %d frames", len(frames))

    # Load analyzers
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_signals import load_analyzers, extract_signals_from_image
    logger.info("Loading analyzers...")
    analyzers = load_analyzers()

    # Extract signals + predict
    results = []
    for i, (orig_idx, img) in enumerate(frames):
        if (i + 1) % 50 == 0:
            logger.info("Processing %d/%d", i + 1, len(frames))

        signals = extract_signals_from_image(img, analyzers, frame_id=i)
        if not signals:
            results.append({"idx": i, "orig_idx": orig_idx, "pred": "no_face", "conf": 0, "b64": frame_to_b64(img)})
            continue

        vec = np.array([normalize_signal(signals.get(f, 0.0), f) for f in feature_names]).reshape(1, -1)
        proba = clf.predict_proba(vec)[0]
        pred_idx = np.argmax(proba)
        pred = classes[pred_idx]
        conf = float(proba[pred_idx])

        # Pose prediction
        pose_pred = ""
        pose_conf = 0.0
        if clf_pose is not None:
            pose_proba = clf_pose.predict_proba(vec)[0]
            pose_idx = np.argmax(pose_proba)
            pose_pred = pose_classes[pose_idx]
            pose_conf = float(pose_proba[pose_idx])

        results.append({
            "idx": i,
            "orig_idx": orig_idx,
            "pred": pred,
            "conf": conf,
            "pose": pose_pred,
            "pose_conf": pose_conf,
            "proba": {c: float(p) for c, p in zip(classes, proba)},
            "b64": frame_to_b64(img),
        })

    # Stats
    from collections import Counter
    pred_counts = Counter(r["pred"] for r in results)
    shoot_count = sum(1 for r in results if r["pred"] != "cut" and r["pred"] != "no_face")
    logger.info("Predictions: %s", dict(pred_counts))
    logger.info("SHOOT: %d / %d frames", shoot_count, len(results))

    # Generate HTML report
    html = generate_report_html(video_path.name, results, classes, meta)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


COLORS = {
    'cheese': '#4CAF50', 'goofy': '#E91E63', 'chill': '#2196F3',
    'edge': '#FF5722', 'hype': '#9C27B0', 'cut': '#d32f2f',
    'occluded': '#795548', 'no_face': '#999',
    'front': '#00BCD4', 'angle': '#FF9800', 'side': '#795548',
}


def generate_report_html(video_name, results, classes, meta):
    from collections import Counter
    counts = Counter(r["pred"] for r in results)
    shoot_results = [r for r in results if r["pred"] not in ("cut", "no_face")]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Prediction Report — {video_name}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; margin: 20px; }}
h1 {{ color: #e94560; }}
h2 {{ margin-top: 24px; color: #444; }}
.summary {{ background: #fff; padding: 12px; border-radius: 8px; margin: 10px 0; border: 1px solid #e0e0e0; font-size: 13px; }}
.timeline {{ display: flex; height: 20px; margin: 10px 0; border-radius: 4px; overflow: hidden; }}
.tl-seg {{ flex: 1; min-width: 1px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; margin: 10px 0; }}
.card {{ background: #fff; border-radius: 6px; padding: 6px; text-align: center; border: 1px solid #e0e0e0; }}
.card img {{ width: 100%; border-radius: 4px; }}
.card .info {{ font-size: 11px; margin-top: 4px; }}
.tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; color: #fff; margin: 1px; }}
.bar {{ display: flex; gap: 4px; margin: 8px 0; align-items: center; }}
.bar-seg {{ height: 16px; border-radius: 3px; display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: #fff; font-weight: 600; min-width: 30px; }}
.conf-bar {{ height: 4px; border-radius: 2px; margin-top: 4px; }}
.timeline {{ cursor: pointer; }}
.tl-seg:hover {{ opacity: 0.7; }}
.tl-seg.active {{ outline: 2px solid #e94560; z-index: 1; }}
.preview {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px;
    margin: 12px 0; display: none; align-items: flex-start; gap: 16px; }}
.preview.open {{ display: flex; }}
.preview img {{ max-width: 480px; max-height: 360px; border-radius: 6px; object-fit: contain; }}
.preview .pv-info {{ font-size: 12px; line-height: 1.8; }}
.preview .pv-nav {{ display: flex; gap: 8px; margin-top: 8px; }}
.preview .pv-nav button {{ padding: 6px 16px; border: none; border-radius: 4px; background: #e8e8e8;
    cursor: pointer; font-size: 13px; }}
.preview .pv-nav button:hover {{ background: #ddd; }}
</style>
</head><body>
<h1>Prediction Report</h1>
<div class="summary">
    <b>{video_name}</b> | {len(results)} frames | model: {meta.get('version','?')} ({meta.get('cv_accuracy',0):.1%} CV) |
    <b>SHOOT: {len(shoot_results)}</b> / {len(results)} ({len(shoot_results)/max(len(results),1):.0%})
</div>
"""

    # Distribution bar
    total = len(results)
    html += '<div class="bar">'
    for cls in classes + ["no_face"]:
        n = counts.get(cls, 0)
        if n == 0:
            continue
        pct = n / total * 100
        color = COLORS.get(cls, '#999')
        html += f'<div class="bar-seg" style="background:{color};width:{max(pct, 3):.1f}%">{cls} {n}</div>'
    html += '</div>'

    # Timeline — expression
    html += '<div style="font-size:10px;color:#999;margin-top:8px">Expression</div>'
    html += '<div class="timeline" id="tlExpr">'
    for r in results:
        color = COLORS.get(r["pred"], '#999')
        html += f'<div class="tl-seg" data-idx="{r["idx"]}" style="background:{color}" title="#{r["idx"]} {r["pred"]} {r["conf"]:.0%}" onclick="showPreview({r["idx"]})"></div>'
    html += '</div>'

    # Timeline — pose
    html += '<div style="font-size:10px;color:#999;margin-top:2px">Pose</div>'
    html += '<div class="timeline" style="height:10px" id="tlPose">'
    for r in results:
        pose_color = COLORS.get(r.get("pose",""), '#eee')
        html += f'<div class="tl-seg" data-idx="{r["idx"]}" style="background:{pose_color}" title="#{r["idx"]} {r.get("pose","?")}" onclick="showPreview({r["idx"]})"></div>'
    html += '</div>'

    # Preview panel
    html += """
<div class="preview" id="preview">
    <img id="pvImg" src="">
    <div class="pv-info">
        <div id="pvDetail"></div>
        <div class="pv-nav">
            <button onclick="pvNav(-1)">← Prev (H)</button>
            <button onclick="pvNav(1)">Next (L) →</button>
            <button onclick="closePreview()" style="margin-left:auto;background:#eee">Close (Esc)</button>
        </div>
    </div>
</div>
"""

    # Frame data as JSON (without b64 to keep small, b64 loaded separately)
    frame_meta = json.dumps([
        {"idx": r["idx"], "pred": r["pred"], "conf": round(r["conf"], 3),
         "pose": r.get("pose",""), "pose_conf": round(r.get("pose_conf",0), 3)}
        for r in results
    ])
    # b64 images stored in separate array
    b64_list = json.dumps([r["b64"] for r in results])

    html += f"""
<script>
const FRAMES = {frame_meta};
const B64 = {b64_list};
const COLORS = {json.dumps(COLORS)};
let pvIdx = -1;

function showPreview(idx) {{
    pvIdx = idx;
    const f = FRAMES[idx];
    document.getElementById('pvImg').src = 'data:image/jpeg;base64,' + B64[idx];
    const exprColor = COLORS[f.pred] || '#999';
    const poseColor = COLORS[f.pose] || '#999';
    document.getElementById('pvDetail').innerHTML =
        `<div style="font-size:16px;margin-bottom:4px"><span class="tag" style="background:${{exprColor}}">${{f.pred}}</span> ` +
        `<span class="tag" style="background:${{poseColor}}">${{f.pose||'?'}}</span></div>` +
        `<div>Frame #${{f.idx}} / ${{FRAMES.length}}</div>` +
        `<div>Expression: ${{f.pred}} (${{(f.conf*100).toFixed(0)}}%)</div>` +
        `<div>Pose: ${{f.pose||'?'}} (${{(f.pose_conf*100).toFixed(0)}}%)</div>`;
    document.getElementById('preview').classList.add('open');
    // Highlight in timeline
    document.querySelectorAll('.tl-seg').forEach(s => s.classList.remove('active'));
    document.querySelectorAll(`.tl-seg[data-idx="${{idx}}"]`).forEach(s => s.classList.add('active'));
}}

function closePreview() {{
    document.getElementById('preview').classList.remove('open');
    document.querySelectorAll('.tl-seg').forEach(s => s.classList.remove('active'));
    pvIdx = -1;
}}

function pvNav(delta) {{
    if (pvIdx < 0) return;
    const next = pvIdx + delta;
    if (next >= 0 && next < FRAMES.length) showPreview(next);
}}

document.addEventListener('keydown', e => {{
    if (pvIdx < 0) return;
    if (e.key === 'Escape') {{ closePreview(); e.preventDefault(); }}
    else if (e.key === 'ArrowLeft' || e.key === 'h') {{ pvNav(-1); e.preventDefault(); }}
    else if (e.key === 'ArrowRight' || e.key === 'l') {{ pvNav(1); e.preventDefault(); }}
}});
</script>
"""

    # SHOOT frames grouped by expression
    for cls in classes:
        if cls == "cut":
            continue
        group = [r for r in results if r["pred"] == cls]
        if not group:
            continue
        color = COLORS.get(cls, '#999')
        html += f'<h2 style="color:{color}">{cls} ({len(group)})</h2>'
        html += '<div class="grid">'
        for r in sorted(group, key=lambda x: -x["conf"]):
            pose_color = COLORS.get(r.get("pose",""), "#999")
            pose_tag = f'<span class="tag" style="background:{pose_color}">{r.get("pose","?")}</span>' if r.get("pose") else ""
            html += f'''<div class="card">
                <img src="data:image/jpeg;base64,{r["b64"]}">
                <div class="info">
                    <span class="tag" style="background:{color}">{cls}</span>
                    {pose_tag}
                    <span style="color:#888">#{r["idx"]} ({r["conf"]:.0%})</span>
                </div>
                <div class="conf-bar" style="background:linear-gradient(to right, {color} {r["conf"]*100:.0f}%, #eee {r["conf"]*100:.0f}%)"></div>
            </div>'''
        html += '</div>'

    # CUT samples (show a few)
    cut_frames = [r for r in results if r["pred"] == "cut"]
    if cut_frames:
        html += f'<h2 style="color:#d32f2f">cut ({len(cut_frames)}) — showing first 20</h2>'
        html += '<div class="grid">'
        for r in cut_frames[:20]:
            html += f'''<div class="card" style="opacity:0.6">
                <img src="data:image/jpeg;base64,{r["b64"]}">
                <div class="info"><span class="tag" style="background:#d32f2f">cut</span> #{r["idx"]} ({r["conf"]:.0%})</div>
            </div>'''
        html += '</div>'

    html += '</body></html>'
    return html


if __name__ == "__main__":
    main()
