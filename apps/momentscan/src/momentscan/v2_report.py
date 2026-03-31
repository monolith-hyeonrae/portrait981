"""MomentscanV2 HTML report — single-file interactive analysis report.

Plotly 기반 인터랙티브 차트. 단일 HTML 파일로 자체 완결.

Usage:
    from momentscan.v2_report import export_v2_report

    results = app.run("video.mp4")
    selected = app.select_frames(results)
    export_v2_report(results, selected, "report.html", video_name="test_0")
"""

from __future__ import annotations

import base64
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("momentscan.v2_report")

# Expression colors (match v2_debug)
EXPR_COLORS = {
    "cheese": "#ffc800",
    "chill": "#00b4d8",
    "edge": "#b400b4",
    "goofy": "#00dc96",
    "hype": "#ff6000",
    "cut": "#c83232",
}

THUMB_WIDTH = 240
JPEG_QUALITY = 70


def export_v2_report(
    results: list,
    selected: list,
    output_path: str | Path,
    video_name: str = "",
    summary=None,
) -> Path:
    """Generate v2 HTML report.

    Args:
        results: list[FrameResult] from MomentscanV2.run()
        selected: list[FrameResult] from select_frames()
        output_path: output HTML path
        video_name: video identifier for title
        summary: SignalSummary from MomentscanV2.summary()
    """
    output_path = Path(output_path)

    total = len(results)
    face = sum(1 for r in results if r.face_detected)
    gate_fail = sum(1 for r in results if r.face_detected and not r.gate_passed)
    shoot = sum(1 for r in results if r.is_shoot)

    # Build data
    frames = []
    for r in results:
        entry = {
            "idx": r.frame_idx,
            "ts": r.timestamp_ms,
            "face": r.face_detected,
            "gate": r.gate_passed,
            "shoot": r.is_shoot,
            "quality": r.judgment.quality if hasattr(r, "judgment") else "",
            "quality_conf": r.judgment.quality_conf if hasattr(r, "judgment") else 0.0,
            "expr": r.expression,
            "expr_conf": r.expression_conf,
            "pose": r.pose,
            "pose_conf": r.pose_conf,
            "z_score": r.z_score,
            "gate_reasons": r.judgment.gate_reasons if hasattr(r, "judgment") else [],
            "expr_scores": r.judgment.expression_scores if hasattr(r, "judgment") and r.gate_passed else {},
            "signals": {k: round(v, 4) for k, v in r.signals.items()},
        }
        frames.append(entry)

    # Thumbnails for selected frames
    thumbs_html = _build_thumbnails(selected)

    # Build plotly charts
    expr_chart = _build_expression_chart(frames)
    gate_chart = _build_gate_chart(frames)
    zscore_chart = _build_zscore_chart(frames)
    au_chart = _build_au_heatmap(frames)
    coverage_html = _build_coverage_table(selected)
    dist_html = _build_distribution_section(summary) if summary else ""
    signal_radar = _build_signal_radar(summary) if summary else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MomentscanV2 Report — {video_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0; margin: 20px; }}
h1 {{ color: #00b4d8; }}
h2 {{ color: #a0a0c0; border-bottom: 1px solid #333; padding-bottom: 4px; }}
.stats {{ display: flex; gap: 30px; margin: 15px 0; }}
.stat {{ background: #16213e; padding: 12px 20px; border-radius: 8px; }}
.stat-value {{ font-size: 24px; font-weight: bold; }}
.stat-label {{ font-size: 12px; color: #888; }}
.green {{ color: #00c853; }}
.red {{ color: #ff1744; }}
.yellow {{ color: #ffd600; }}
.chart {{ margin: 20px 0; }}
.thumbs {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }}
.thumb {{ background: #16213e; border-radius: 8px; padding: 8px; text-align: center; }}
.thumb img {{ border-radius: 4px; max-width: {THUMB_WIDTH}px; }}
.thumb-label {{ font-size: 11px; color: #888; margin-top: 4px; }}
.coverage {{ border-collapse: collapse; margin: 10px 0; }}
.coverage td, .coverage th {{ padding: 8px 16px; border: 1px solid #333; text-align: center; }}
.coverage th {{ background: #16213e; }}
.coverage .filled {{ background: #1a3a2a; color: #00c853; }}
.coverage .empty {{ background: #2a1a1a; color: #555; }}
</style>
</head>
<body>
<h1>MomentscanV2 — {video_name}</h1>

<div class="stats">
  <div class="stat"><div class="stat-value">{total}</div><div class="stat-label">Total Frames</div></div>
  <div class="stat"><div class="stat-value">{face}</div><div class="stat-label">Face Detected</div></div>
  <div class="stat"><div class="stat-value red">{gate_fail}</div><div class="stat-label">Gate Fail</div></div>
  <div class="stat"><div class="stat-value green">{shoot}</div><div class="stat-label">SHOOT</div></div>
  <div class="stat"><div class="stat-value yellow">{len(selected)}</div><div class="stat-label">Selected</div></div>
</div>

<h2>Selected Frames</h2>
<div class="thumbs">{thumbs_html}</div>

<h2>Expression × Pose Coverage</h2>
{coverage_html}

{dist_html}

<h2>Expression Confidence Timeline</h2>
<div class="chart" id="expr_chart"></div>

<h2>Gate Severity Timeline</h2>
<div class="chart" id="gate_chart"></div>

<h2>Quality × Expression × Z-Score</h2>
<div class="chart" id="zscore_chart"></div>

<h2>Mean Signal Profile</h2>
<div class="chart" id="radar_chart"></div>

<h2>AU Activation Heatmap</h2>
<div class="chart" id="au_chart"></div>

<script>
{expr_chart}
{gate_chart}
{zscore_chart}
{au_chart}
{signal_radar}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s", output_path)
    return output_path


def _build_expression_chart(frames: list) -> str:
    """Plotly expression confidence timeline."""
    categories = set()
    for f in frames:
        categories.update(f.get("expr_scores", {}).keys())
    categories = sorted(categories)

    traces = []
    for cat in categories:
        x = []
        y = []
        for f in frames:
            if f["face"]:
                x.append(f["idx"])
                scores = f.get("expr_scores", {})
                y.append(scores.get(cat, 0.0) if f["gate"] else None)
        color = EXPR_COLORS.get(cat, "#888")
        traces.append(f"""{{
            x: {x}, y: {y}, name: '{cat}', type: 'scatter', mode: 'lines',
            line: {{color: '{color}', width: 1.5}},
        }}""")

    return f"""Plotly.newPlot('expr_chart', [{','.join(traces)}], {{
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: {{color: '#e0e0e0'}},
        xaxis: {{title: 'Frame', gridcolor: '#333'}},
        yaxis: {{title: 'Confidence', range: [0, 1], gridcolor: '#333'}},
        legend: {{orientation: 'h', y: -0.15}},
        margin: {{t: 10}},
        height: 300,
    }});"""


def _build_gate_chart(frames: list) -> str:
    """Plotly gate severity timeline."""
    x = []
    severity = []
    colors = []

    for f in frames:
        if not f["face"]:
            continue
        x.append(f["idx"])
        sev = _compute_gate_severity_from_signals(f["signals"])
        severity.append(sev)
        if f["shoot"]:
            colors.append("#00c853")
        elif not f["gate"]:
            colors.append("#ff1744")
        else:
            colors.append("#ffd600")

    return f"""Plotly.newPlot('gate_chart', [{{
        x: {x}, y: {[round(s, 3) for s in severity]},
        type: 'bar', marker: {{color: {colors}}},
        hovertext: {[f['gate_reasons'] for f in frames if f['face']]},
    }}], {{
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: {{color: '#e0e0e0'}},
        xaxis: {{title: 'Frame', gridcolor: '#333'}},
        yaxis: {{title: 'Gate Severity', range: [0, 1], gridcolor: '#333'}},
        margin: {{t: 10}},
        height: 200,
    }});"""


def _build_zscore_chart(frames: list) -> str:
    """Plotly q×e×z timeline — quality_conf, expression_conf, z_score 3축."""
    x = []
    q_vals, e_vals, z_vals, qez_vals = [], [], [], []

    for f in frames:
        if not f["face"]:
            continue
        x.append(f["idx"])
        q = f.get("quality_conf", 0.0)
        e = f.get("expr_conf", 0.0)
        z = f.get("z_score", 0.0)
        q_vals.append(round(q, 3))
        e_vals.append(round(e, 3))
        z_vals.append(round(z, 3))
        qez = q * e * max(z, 0.1)
        qez_vals.append(round(qez, 3))

    return f"""Plotly.newPlot('zscore_chart', [
        {{x: {x}, y: {q_vals}, name: 'quality', type: 'scatter', mode: 'lines',
          line: {{color: '#4fc3f7', width: 1}}}},
        {{x: {x}, y: {e_vals}, name: 'expression', type: 'scatter', mode: 'lines',
          line: {{color: '#ab47bc', width: 1}}}},
        {{x: {x}, y: {z_vals}, name: 'z_score', type: 'scatter', mode: 'lines',
          line: {{color: '#ff7043', width: 1}}}},
        {{x: {x}, y: {qez_vals}, name: 'q×e×z', type: 'scatter', mode: 'lines',
          line: {{color: '#ffd600', width: 2}}}},
    ], {{
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: {{color: '#e0e0e0'}},
        xaxis: {{title: 'Frame', gridcolor: '#333'}},
        yaxis: {{title: 'Score', gridcolor: '#333'}},
        legend: {{x: 0, y: 1.1, orientation: 'h'}},
        margin: {{t: 10}},
        height: 250,
    }});"""


def _build_au_heatmap(frames: list) -> str:
    """Plotly AU heatmap."""
    au_names = [
        "au1_inner_brow", "au2_outer_brow", "au4_brow_lowerer", "au5_upper_lid",
        "au6_cheek_raiser", "au9_nose_wrinkler", "au12_lip_corner", "au15_lip_depressor",
        "au17_chin_raiser", "au20_lip_stretcher", "au25_lips_part", "au26_jaw_drop",
    ]
    au_short = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]

    x = [f["idx"] for f in frames if f["face"]]
    z = []
    for au in au_names:
        row = [f["signals"].get(au, 0.0) for f in frames if f["face"]]
        z.append(row)

    return f"""Plotly.newPlot('au_chart', [{{
        x: {x}, y: {au_short}, z: {z},
        type: 'heatmap',
        colorscale: [[0,'#000'],[0.3,'#004d00'],[0.6,'#cccc00'],[1,'#cc0000']],
        zmin: 0, zmax: 1,
    }}], {{
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: {{color: '#e0e0e0'}},
        xaxis: {{title: 'Frame', gridcolor: '#333'}},
        margin: {{t: 10, l: 60}},
        height: 250,
    }});"""


def _build_coverage_table(selected: list) -> str:
    """HTML table for expression × pose coverage."""
    expr_cats = ["cheese", "chill", "edge", "goofy", "hype"]
    pose_cats = ["front", "angle", "side"]

    grid = {}
    for r in selected:
        key = (r.expression, r.pose)
        if r.expression_conf > grid.get(key, 0.0):
            grid[key] = r.expression_conf

    rows = ""
    for expr in expr_cats:
        cells = f'<td style="color:{EXPR_COLORS.get(expr, "#888")}">{expr}</td>'
        for pose in pose_cats:
            conf = grid.get((expr, pose), 0.0)
            if conf > 0:
                cells += f'<td class="filled">{conf:.0%}</td>'
            else:
                cells += '<td class="empty">-</td>'
        rows += f"<tr>{cells}</tr>\n"

    return f"""<table class="coverage">
<tr><th></th>{''.join(f'<th>{p}</th>' for p in pose_cats)}</tr>
{rows}</table>"""


def _build_thumbnails(selected: list) -> str:
    """Base64 thumbnails for selected frames."""
    parts = []
    for r in selected:
        img = getattr(r, "image", None)
        if img is None:
            parts.append(f'<div class="thumb"><div class="thumb-label">#{r.frame_idx} {r.expression} ({r.expression_conf:.0%}) {r.pose}</div></div>')
            continue

        # Resize
        h, w = img.shape[:2]
        scale = THUMB_WIDTH / w
        thumb = cv2.resize(img, (THUMB_WIDTH, int(h * scale)))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buf).decode()

        parts.append(f'''<div class="thumb">
<img src="data:image/jpeg;base64,{b64}">
<div class="thumb-label">#{r.frame_idx} {r.expression} ({r.expression_conf:.0%}) {r.pose}</div>
</div>''')

    return "\n".join(parts)


def _build_distribution_section(summary) -> str:
    """Person signal distribution — expression/pose pie + signal stats."""
    if summary is None or summary.n_shoot == 0:
        return ""

    # Expression distribution pie
    expr_labels = list(summary.expression_dist.keys())
    expr_values = [round(v * 100, 1) for v in summary.expression_dist.values()]
    expr_colors = [EXPR_COLORS.get(k, "#888") for k in expr_labels]

    # Pose distribution pie
    pose_labels = list(summary.pose_dist.keys())
    pose_values = [round(v * 100, 1) for v in summary.pose_dist.values()]

    # Signal variance (top-10 most variable dimensions)
    from visualbind.signals import SIGNAL_FIELDS
    fields = list(SIGNAL_FIELDS)
    variances = np.diag(summary.cov) if summary.cov.ndim == 2 else np.zeros(len(fields))
    sorted_idx = np.argsort(variances)[::-1][:10]
    var_html = ""
    for i in sorted_idx:
        if variances[i] < 1e-6:
            continue
        name = fields[i]
        std = variances[i] ** 0.5
        mean = summary.mean[i]
        bar_w = min(100, int(std / (abs(mean) + 1e-6) * 100))
        var_html += f'<div style="margin:2px 0;"><span style="display:inline-block;width:160px;font-size:11px;color:#aaa;">{name}</span>'
        var_html += f'<span style="display:inline-block;width:{bar_w}px;height:8px;background:#00b4d8;border-radius:2px;"></span>'
        var_html += f'<span style="font-size:10px;color:#666;margin-left:6px;">μ={mean:.3f} σ={std:.3f}</span></div>'

    # Face embedding info
    n_embeds = len(summary.face_embeddings)
    embed_info = f"<p style='font-size:12px;color:#888;'>Face embeddings collected: {n_embeds}</p>" if n_embeds > 0 else ""

    return f"""
<h2>Person Signal Distribution</h2>
<div style="display:flex;gap:30px;flex-wrap:wrap;">
  <div>
    <h3 style="color:#a0a0c0;font-size:14px;">Expression Distribution ({summary.n_shoot} SHOOT frames)</h3>
    <div style="display:flex;gap:10px;align-items:center;">
      {''.join(f'<div style="text-align:center;"><div style="width:60px;height:60px;border-radius:50%;background:{expr_colors[i]};display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:bold;">{expr_values[i]:.0f}%</div><div style="font-size:10px;color:#aaa;margin-top:2px;">{expr_labels[i]}</div></div>' for i in range(len(expr_labels)))}
    </div>
  </div>
  <div>
    <h3 style="color:#a0a0c0;font-size:14px;">Pose Distribution</h3>
    <div style="display:flex;gap:10px;align-items:center;">
      {''.join(f'<div style="text-align:center;"><div style="width:60px;height:60px;border-radius:50%;background:#334;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:bold;color:#00b4d8;">{pose_values[i]:.0f}%</div><div style="font-size:10px;color:#aaa;margin-top:2px;">{pose_labels[i]}</div></div>' for i in range(len(pose_labels)))}
    </div>
  </div>
</div>

<h3 style="color:#a0a0c0;font-size:14px;margin-top:15px;">Signal Variance (이 사람의 가장 큰 변화 축)</h3>
<div style="background:#16213e;padding:12px;border-radius:8px;max-width:500px;">
{var_html}
</div>
{embed_info}
"""


def _build_signal_radar(summary) -> str:
    """Plotly radar chart for mean signal profile."""
    if summary is None or summary.n_shoot == 0:
        return ""

    # Select meaningful signal groups for radar
    groups = {
        "AU6 cheek": "au6_cheek_raiser",
        "AU12 lip": "au12_lip_corner",
        "AU25 lips": "au25_lips_part",
        "AU4 brow": "au4_brow_lowerer",
        "Happy": "em_happy",
        "Neutral": "em_neutral",
        "Surprise": "em_surprise",
        "Yaw": "head_yaw_dev",
        "Blur": "face_blur",
        "Exposure": "face_exposure",
        "Seg Face": "seg_face",
        "Eye Vis": "eye_visible_ratio",
    }

    from visualbind.signals import SIGNAL_FIELDS, SIGNAL_RANGES
    fields = list(SIGNAL_FIELDS)

    labels = []
    values = []
    for display, sig in groups.items():
        if sig not in fields:
            continue
        idx = fields.index(sig)
        raw = summary.mean[idx]
        # Normalize to 0-1 for radar
        lo, hi = SIGNAL_RANGES.get(sig, (0, 1))
        norm = (raw - lo) / (hi - lo + 1e-8)
        norm = max(0.0, min(1.0, norm))
        labels.append(display)
        values.append(round(norm, 3))

    if not labels:
        return ""

    # Close the radar
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    return f"""Plotly.newPlot('radar_chart', [{{
        type: 'scatterpolar', r: {values_closed}, theta: {labels_closed},
        fill: 'toself', fillcolor: 'rgba(0,180,216,0.15)',
        line: {{color: '#00b4d8'}}, name: 'Mean Signal',
    }}], {{
        polar: {{
            radialaxis: {{visible: true, range: [0, 1], gridcolor: '#333', color: '#888'}},
            angularaxis: {{gridcolor: '#333', color: '#aaa'}},
            bgcolor: '#16213e',
        }},
        paper_bgcolor: '#1a1a2e', font: {{color: '#e0e0e0'}},
        margin: {{t: 10, b: 10}}, height: 350,
        showlegend: false,
    }});"""


def _compute_gate_severity_from_signals(signals: dict) -> float:
    """Same logic as v2_debug._compute_gate_severity but from flat dict."""
    severity = 0.0

    expo = signals.get("face_exposure", 100.0)
    if expo > 0:
        if expo < 50:
            severity += (50 - expo) / 50
        elif expo > 200:
            severity += (expo - 200) / 55

    contrast = signals.get("face_contrast", 0.5)
    if 0 < contrast < 0.10:
        severity += (0.10 - contrast) / 0.10

    severity += min(1.0, signals.get("clipped_ratio", 0) / 0.15)
    severity += min(1.0, signals.get("crushed_ratio", 0) / 0.15)

    blur = signals.get("face_blur", 50.0)
    if 0 < blur < 5:
        severity += (5 - blur) / 5

    conf = signals.get("face_confidence", 0.85)
    if 0 < conf < 0.7:
        severity += (0.7 - conf) / 0.7

    yaw = abs(signals.get("head_yaw_dev", 0))
    pitch = abs(signals.get("head_pitch", 0))
    roll = abs(signals.get("head_roll", 0))
    if yaw > 55: severity += (yaw - 55) / 35
    if pitch > 35: severity += (pitch - 35) / 55
    if roll > 35: severity += (roll - 35) / 55
    combined = math.sqrt(yaw**2 + pitch**2 + roll**2)
    if combined > 55: severity += (combined - 55) / 45

    seg = signals.get("seg_face", -1)
    if seg >= 0 and seg < 0.01: severity += 0.5
    au_keys = [k for k in signals if k.startswith("au")]
    if au_keys and sum(signals[k] for k in au_keys) < 0.05: severity += 0.3

    return min(1.0, severity)
