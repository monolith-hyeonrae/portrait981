"""Dataset review HTML generation with label editing capability.

Reads labels.csv and generates an interactive HTML gallery grouped by expression/pose,
with inline editing buttons for each image card.

Usage (CLI):
    annotator review data/datasets/portrait-v1 --output review.html

Usage (API):
    from annotator.review import generate_review_html
    generate_review_html("data/datasets/portrait-v1", output_path="review.html")
"""

from __future__ import annotations

import base64
import csv
import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger("annotator.review")


def _img_to_b64(img_path: Path, max_width: int = 280) -> str:
    """Read image file and return base64-encoded JPEG."""
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def generate_review_html(
    dataset_dir: str | Path,
    output_path: str | Path = "review.html",
) -> Path:
    """Generate interactive review HTML with label editing from a dataset directory.

    The dataset directory must contain ``labels.csv`` and an ``images/`` subdirectory.
    Returns the path to the generated HTML file.
    """
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found in {dataset_dir}")

    # Load labels
    rows: list[dict] = []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("Loaded %d labels", len(rows))

    # Group by expression and pose
    by_expr: dict[str, list[dict]] = {}
    by_pose: dict[str, list[dict]] = {}
    unlabeled_expr: list[dict] = []
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
    colors_json = json.dumps(colors)

    # Build rows data for JS editing
    rows_json = json.dumps(rows)

    # Pre-encode all images
    image_data: dict[str, str] = {}
    for row in rows:
        fname = row["filename"]
        if fname not in image_data:
            img_path = images_dir / fname
            if img_path.exists():
                image_data[fname] = _img_to_b64(img_path)

    image_data_json = json.dumps(image_data)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dataset Review</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
h1 {{ color: #e94560; }}
h2 {{ margin-top: 30px; }}
.toolbar {{ background: #0f0f23; padding: 12px 20px; border-radius: 8px; margin: 10px 0;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.toolbar button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.btn-save {{ background: #4CAF50; color: #fff; }}
.btn-save:disabled {{ opacity: 0.5; cursor: default; }}
.changes-count {{ font-size: 14px; color: #FF9800; }}
.summary {{ background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }}
.card {{ background: #16213e; border-radius: 6px; padding: 8px; text-align: center; }}
.card img {{ width: 100%; border-radius: 4px; }}
.card .name {{ font-size: 11px; color: #888; margin-top: 4px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }}
.card .tags {{ margin-top: 4px; }}
.tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 1px; }}
.section {{ margin: 20px 0; }}
.edit-btns {{ display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 6px; }}
.edit-btn {{ padding: 2px 6px; border: 1px solid #555; border-radius: 3px; background: #222;
    color: #aaa; cursor: pointer; font-size: 10px; transition: all .1s; }}
.edit-btn:hover {{ background: #444; color: #fff; }}
.edit-btn.active {{ color: #fff; font-weight: bold; }}
.modified {{ box-shadow: 0 0 0 2px #FF9800; }}
</style>
</head><body>
<h1>Dataset Review</h1>

<div class="toolbar">
    <button class="btn-save" id="saveBtn" onclick="downloadCSV()" disabled>Download Modified CSV</button>
    <span class="changes-count" id="changesCount"></span>
</div>

<div class="summary">
    <b>Total:</b> {len(rows)} images |
    <b>Expression:</b> {', '.join(f'{k}={len(v)}' for k, v in sorted(by_expr.items()))} |
    <b>Pose:</b> {', '.join(f'{k}={len(v)}' for k, v in sorted(by_pose.items()))} |
    <b>Unlabeled expression:</b> {len(unlabeled_expr)}
</div>

<div id="content"></div>

<script>
const ROWS = {rows_json};
const IMAGES = {image_data_json};
const COLORS = {colors_json};
const EXPRESSIONS = ['cheese', 'chill', 'edge', 'hype', 'pass'];
const POSES = ['front', 'angle', 'side'];
let changes = {{}};
let changeCount = 0;

function getColor(cat) {{ return COLORS[cat] || '#666'; }}

function setExpr(idx, expr) {{
    const row = ROWS[idx];
    const old = row.expression;
    if (old === expr) return;
    row.expression = expr;
    if (!changes[idx]) changes[idx] = {{}};
    changes[idx].expression = expr;
    changeCount++;
    updateUI(idx);
    updateSaveBtn();
}}

function setPose(idx, pose) {{
    const row = ROWS[idx];
    const old = row.pose;
    if (old === pose) return;
    row.pose = pose;
    if (!changes[idx]) changes[idx] = {{}};
    changes[idx].pose = pose;
    changeCount++;
    updateUI(idx);
    updateSaveBtn();
}}

function updateUI(idx) {{
    const card = document.getElementById('card-' + idx);
    if (!card) return;
    card.classList.add('modified');
    const row = ROWS[idx];
    // Update expression buttons
    card.querySelectorAll('.expr-btn').forEach(btn => {{
        const val = btn.dataset.val;
        if (val === row.expression) {{
            btn.classList.add('active');
            btn.style.background = getColor(val);
        }} else {{
            btn.classList.remove('active');
            btn.style.background = '#222';
        }}
    }});
    // Update pose buttons
    card.querySelectorAll('.pose-btn').forEach(btn => {{
        const val = btn.dataset.val;
        if (val === row.pose) {{
            btn.classList.add('active');
            btn.style.background = getColor(val);
        }} else {{
            btn.classList.remove('active');
            btn.style.background = '#222';
        }}
    }});
}}

function updateSaveBtn() {{
    const n = Object.keys(changes).length;
    document.getElementById('saveBtn').disabled = n === 0;
    document.getElementById('changesCount').textContent = n > 0 ? n + ' cards modified' : '';
}}

function downloadCSV() {{
    const header = 'filename,member_id,expression,pose,source';
    const lines = [header];
    for (const row of ROWS) {{
        lines.push([row.filename, row.member_id || '', row.expression || '', row.pose || '', row.source || ''].join(','));
    }}
    const blob = new Blob([lines.join('\\n') + '\\n'], {{ type: 'text/csv' }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'labels_modified.csv';
    a.click();
}}

function renderCard(idx, row) {{
    const fname = row.filename;
    const b64 = IMAGES[fname];
    if (!b64) return '';

    const expr = row.expression || '';
    const pose = row.pose || '';
    const member = row.member_id || '';

    let tags = '';
    if (expr) tags += `<span class="tag" style="background:${{getColor(expr)}}">${{expr}}</span>`;
    if (pose) tags += `<span class="tag" style="background:${{getColor(pose)}}">${{pose}}</span>`;
    if (member) tags += `<span class="tag" style="background:#333">${{member}}</span>`;

    let exprBtns = '';
    for (const e of EXPRESSIONS) {{
        const active = expr === e;
        const bg = active ? `background:${{getColor(e)}}` : 'background:#222';
        exprBtns += `<button class="edit-btn expr-btn${{active ? ' active' : ''}}" data-val="${{e}}" style="${{bg}}" onclick="setExpr(${{idx}},'${{e}}')">${{e}}</button>`;
    }}

    let poseBtns = '';
    for (const p of POSES) {{
        const active = pose === p;
        const bg = active ? `background:${{getColor(p)}}` : 'background:#222';
        poseBtns += `<button class="edit-btn pose-btn${{active ? ' active' : ''}}" data-val="${{p}}" style="${{bg}}" onclick="setPose(${{idx}},'${{p}}')">${{p}}</button>`;
    }}

    return `<div class="card" id="card-${{idx}}">
        <img src="data:image/jpeg;base64,${{b64}}">
        <div class="name">${{fname}}</div>
        <div class="tags">${{tags}}</div>
        <div class="edit-btns">${{exprBtns}}</div>
        <div class="edit-btns" style="margin-top:3px">${{poseBtns}}</div>
    </div>`;
}}

function renderAll() {{
    const content = document.getElementById('content');
    let html = '';

    // Group by expression
    const byExpr = {{}};
    const unlabeled = [];
    ROWS.forEach((row, idx) => {{
        const e = row.expression || '';
        if (e) {{
            if (!byExpr[e]) byExpr[e] = [];
            byExpr[e].push({{ row, idx }});
        }} else {{
            unlabeled.push({{ row, idx }});
        }}
    }});

    for (const expr of EXPRESSIONS) {{
        const items = byExpr[expr] || [];
        if (items.length === 0) continue;
        const color = getColor(expr);
        html += `<div class="section"><h2 style="color:${{color}}">${{expr}} (${{items.length}})</h2><div class="grid">`;
        for (const {{ row, idx }} of items) {{
            html += renderCard(idx, row);
        }}
        html += '</div></div>';
    }}

    // Pose sections
    const byPose = {{}};
    ROWS.forEach((row, idx) => {{
        const p = row.pose || '';
        if (p) {{
            if (!byPose[p]) byPose[p] = [];
            byPose[p].push({{ row, idx }});
        }}
    }});

    for (const pose of POSES) {{
        const items = byPose[pose] || [];
        if (items.length === 0) continue;
        const color = getColor(pose);
        html += `<div class="section"><h2 style="color:${{color}}">Pose: ${{pose}} (${{items.length}})</h2><div class="grid">`;
        for (const {{ row, idx }} of items) {{
            html += renderCard(idx, row);
        }}
        html += '</div></div>';
    }}

    // Unlabeled
    if (unlabeled.length > 0) {{
        html += `<div class="section"><h2 style="color:#888">Unlabeled Expression (${{unlabeled.length}})</h2><div class="grid">`;
        for (const {{ row, idx }} of unlabeled.slice(0, 50)) {{
            html += renderCard(idx, row);
        }}
        html += '</div></div>';
    }}

    content.innerHTML = html;
}}

renderAll();
</script>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Review: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path
