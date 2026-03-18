"""Anchor Set labeling tool -- interactive HTML interface for ground-truth annotation.

2-stage labeling: expression (1=cheese, 2=chill, 3=edge, 4=hype, 5=pass) + pose (q=front, w=angle, e=side).
JSZip export with images/ + labels.csv structure.

Usage (CLI):
    annotator label video.mp4 --fps 2 --output labels.html --max-frames 500

Usage (API):
    from annotator.label import generate_label_html
    generate_label_html("video.mp4", fps=2, max_frames=500, output_path="labels.html")
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger("annotator.label")


def frame_to_base64(frame_bgr, max_width: int = 640) -> str:
    """OpenCV frame -> base64 JPEG."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii")


def _generate_html(frames_info: list[dict], categories: list[str], video_name: str) -> str:
    """Build self-contained interactive HTML labeling tool."""
    frames_json = json.dumps([
        {k: v for k, v in f.items() if k != "b64"} for f in frames_info
    ])

    cat_colors = {
        "cheese": "#4CAF50",
        "chill": "#2196F3",
        "edge": "#FF5722",
        "hype": "#9C27B0",
        "front": "#00BCD4",
        "angle": "#FF9800",
        "side": "#795548",
    }
    cat_list_json = json.dumps(categories)
    cat_colors_json = json.dumps(cat_colors)

    img_entries = ",".join(f'"{f["index"]}":"{f["b64"]}"' for f in frames_info)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<title>Anchor Set Label Tool</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, sans-serif; margin: 0; background: #1a1a2e; color: #eee;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
.toolbar {{ background: #0f0f23; padding: 10px 20px; border-bottom: 2px solid #e94560;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; flex-shrink: 0; }}
.toolbar h1 {{ margin: 0; font-size: 18px; color: #e94560; }}
.progress {{ font-size: 14px; color: #aaa; }}
.progress b {{ color: #4CAF50; font-size: 18px; }}
.toolbar button {{ padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.btn-export {{ background: #4CAF50; color: #fff; }}
.btn-reset {{ background: #666; color: #fff; }}
.filter-group {{ display: flex; gap: 6px; }}
.filter-btn {{ background: #333; color: #ccc; padding: 4px 10px; border: 1px solid #555;
    border-radius: 3px; cursor: pointer; font-size: 12px; }}
.filter-btn.active {{ background: #e94560; color: #fff; border-color: #e94560; }}
.main {{ flex: 1; display: flex; overflow: hidden; }}
.focus-panel {{ flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 20px; min-width: 0; }}
.focus-img {{ max-width: 100%; max-height: 55vh; border-radius: 8px; object-fit: contain; }}
.focus-meta {{ margin: 12px 0 6px; font-size: 14px; color: #aaa; }}
.focus-preds {{ margin: 6px 0; }}
.pred {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 13px; margin: 2px; }}
.focus-label {{ font-size: 16px; color: #4CAF50; margin: 8px 0; font-weight: bold; }}
.buttons {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 10px; }}
.cat-btn {{ padding: 8px 18px; border: 2px solid #444; border-radius: 6px; background: #222;
    color: #ccc; cursor: pointer; font-size: 14px; transition: all .15s; }}
.cat-btn:hover {{ background: #444; color: #fff; transform: scale(1.05); }}
.cat-btn.selected {{ color: #fff; font-weight: bold; }}
.cat-btn.reject {{ border-color: #d32f2f; }}
.cat-btn.skip {{ border-color: #555; }}
.cat-btn.none {{ border-color: #555; }}
.nav {{ display: flex; gap: 12px; margin-top: 14px; align-items: center; }}
.nav button {{ padding: 10px 24px; border: none; border-radius: 6px; background: #333;
    color: #eee; cursor: pointer; font-size: 16px; }}
.nav button:hover {{ background: #555; }}
.nav button:disabled {{ opacity: 0.3; cursor: default; }}
.nav .pos {{ font-size: 14px; color: #aaa; min-width: 80px; text-align: center; }}
.sidebar {{ width: 180px; overflow-y: auto; background: #0f0f23; border-left: 1px solid #333;
    flex-shrink: 0; }}
.thumb {{ width: 100%; padding: 4px; cursor: pointer; opacity: 0.5; transition: opacity .15s;
    border-left: 3px solid transparent; }}
.thumb:hover {{ opacity: 0.8; }}
.thumb.active {{ opacity: 1; border-left-color: #e94560; }}
.thumb.labeled {{ border-left-color: #4CAF50; }}
.thumb img {{ width: 100%; border-radius: 3px; display: block; }}
.shortcut-hint {{ font-size: 11px; color: #555; text-align: center; margin-top: 8px; }}
</style>
</head><body>

<div class="toolbar">
    <h1>Label Tool</h1>
    <div class="progress">
        <b id="count">0</b> / <span id="total">{len(frames_info)}</span> labeled
        &nbsp;| {video_name}
    </div>
    <div class="filter-group">
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="unlabeled">Unlabeled</button>
        <button class="filter-btn" data-filter="labeled">Labeled</button>
        <button class="filter-btn" data-filter="disagree">Disagree</button>
    </div>
    <button class="btn-export" onclick="exportLabels()">Export ZIP</button>
    <button class="btn-reset" onclick="resetLabels()">Reset All</button>
</div>

<div class="main">
    <div class="focus-panel" id="focus"></div>
    <div class="sidebar" id="sidebar"></div>
</div>

<script>
const FRAMES = {frames_json};
const IMAGES = {{{img_entries}}};
const CATEGORIES = {cat_list_json};
const CAT_COLORS = {cat_colors_json};
const STORAGE_KEY = "label_tool_{video_name.replace('.', '_')}";

let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'  );
let currentFilter = 'all';
let filteredList = [];
let currentPos = 0;

function getColor(cat) {{ return CAT_COLORS[cat] || '#666'; }}

function buildFilteredList() {{
    filteredList = FRAMES.filter(f => {{
        if (currentFilter === 'unlabeled') return labels[f.index] === undefined;
        if (currentFilter === 'labeled') return labels[f.index] !== undefined;
        if (currentFilter === 'disagree') return f.is_disagree;
        return true;
    }});
    filteredList.sort((a, b) => {{
        const la = labels[a.index] !== undefined ? 1 : 0;
        const lb = labels[b.index] !== undefined ? 1 : 0;
        if (la !== lb) return la - lb;
        if (a.is_disagree !== b.is_disagree) return a.is_disagree ? -1 : 1;
        return a.index - b.index;
    }});
}}

function renderSidebar() {{
    const sb = document.getElementById('sidebar');
    sb.innerHTML = '';
    filteredList.forEach((f, pos) => {{
        const div = document.createElement('div');
        const isLabeled = labels[f.index] !== undefined;
        div.className = 'thumb' + (pos === currentPos ? ' active' : '') + (isLabeled ? ' labeled' : '');
        div.innerHTML = `<img src="data:image/jpeg;base64,${{IMAGES[f.index]}}" loading="lazy">`;
        div.onclick = () => {{ currentPos = pos; renderFocus(); renderSidebar(); }};
        sb.appendChild(div);
        if (pos === currentPos) div.scrollIntoView({{ block: 'nearest' }});
    }});
}}

function renderFocus() {{
    const panel = document.getElementById('focus');
    if (filteredList.length === 0) {{
        panel.innerHTML = '<p style="color:#888">No frames to show</p>';
        updateCount();
        return;
    }}
    if (currentPos >= filteredList.length) currentPos = filteredList.length - 1;
    if (currentPos < 0) currentPos = 0;

    const f = filteredList[currentPos];
    const label = labels[f.index];
    const pose = poses[f.index];
    const exprText = label ? label : '';
    const poseText = pose ? pose : '';
    const labelHtml = (label || pose)
        ? `<div class="focus-label">${{exprText}}${{exprText && poseText ? ' + ' : ''}}${{poseText}}</div>`
        : `<div class="focus-label" style="color:#888">Unlabeled</div>`;

    let btnsHtml = '<div class="buttons">';
    const EXPRESSIONS = ['cheese', 'chill', 'edge', 'hype'];
    for (const cat of EXPRESSIONS) {{
        const sel = label === cat;
        const bg = sel ? `background:${{getColor(cat)}};color:#fff;` : '';
        btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setLabel(${{f.index}},'${{cat}}')">${{cat}}</button>`;
    }}
    btnsHtml += `<button class="cat-btn${{label === 'pass' ? ' selected' : ''}}" style="${{label === 'pass' ? 'background:#d32f2f;color:#fff;' : ''}}" onclick="setLabel(${{f.index}},'pass')">pass</button>`;
    btnsHtml += `<button class="cat-btn${{label === 'skip' ? ' selected' : ''}}" style="${{label === 'skip' ? 'background:#888;color:#fff;' : ''}}" onclick="setLabel(${{f.index}},'skip')">skip</button>`;
    btnsHtml += '</div>';

    if (label && label !== 'pass' && label !== 'skip') {{
        btnsHtml += '<div class="buttons" style="margin-top:6px">';
        const POSES = ['front', 'angle', 'side'];
        for (const p of POSES) {{
            const sel = pose === p;
            const bg = sel ? `background:${{getColor(p)}};color:#fff;` : '';
            btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setPose(${{f.index}},'${{p}}')">${{p}}</button>`;
        }}
        btnsHtml += '</div>';
    }}

    panel.innerHTML = `
        <img class="focus-img" src="data:image/jpeg;base64,${{IMAGES[f.index]}}">
        <div class="focus-meta">Frame #${{f.index}} &nbsp; ${{currentPos + 1}} / ${{filteredList.length}}
            ${{f.is_disagree ? '&nbsp; <span style="color:#e94560">DISAGREE</span>' : ''}}</div>
        <div class="focus-preds">
            <span class="pred" style="background:${{getColor(f.catalog)}}">Cat: ${{f.catalog}}</span>
            <span class="pred" style="background:${{getColor(f.lr)}}">LR: ${{f.lr}}</span>
            <span class="pred" style="background:${{getColor(f.xgb)}}">XGB: ${{f.xgb}}</span>
        </div>
        ${{labelHtml}}
        ${{btnsHtml}}
        <div class="nav">
            <button onclick="go(-1)" ${{currentPos <= 0 ? 'disabled' : ''}}>&larr; Prev</button>
            <div class="pos">${{currentPos + 1}} / ${{filteredList.length}}</div>
            <button onclick="go(1)" ${{currentPos >= filteredList.length - 1 ? 'disabled' : ''}}>Next &rarr;</button>
        </div>
        <div class="shortcut-hint">Keyboard: &larr;&rarr; navigate &nbsp;|&nbsp; 1=cheese 2=chill 3=edge 4=hype 5=pass s=skip &nbsp;|&nbsp; q=front w=angle e=side</div>
    `;
    updateCount();
}}

function go(delta) {{
    currentPos = Math.max(0, Math.min(filteredList.length - 1, currentPos + delta));
    renderFocus();
    renderSidebar();
}}

let poses = JSON.parse(localStorage.getItem(STORAGE_KEY + '_pose') || '{{}}');

function setLabel(index, label) {{
    if (labels[index] === label) {{
        delete labels[index];
    }} else {{
        labels[index] = label;
    }}
    localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    renderFocus();
    renderSidebar();
}}

function setPose(index, pose) {{
    if (poses[index] === pose) {{
        delete poses[index];
    }} else {{
        poses[index] = pose;
    }}
    localStorage.setItem(STORAGE_KEY + '_pose', JSON.stringify(poses));
    renderFocus();
}}

function updateCount() {{
    document.getElementById('count').textContent = Object.keys(labels).length;
}}

async function exportLabels() {{
    const labeled = FRAMES.filter(f => labels[f.index] && labels[f.index] !== 'skip');
    if (labeled.length === 0) {{
        alert('No labeled frames to export.');
        return;
    }}

    const btn = document.querySelector('.btn-export');
    btn.textContent = 'Exporting...';
    btn.disabled = true;

    try {{
        const zip = new JSZip();
        const videoBase = "{video_name}".replace(/\\.[^.]+$/, '');

        const csvRows = ['filename,member_id,expression,pose,source'];
        for (const f of labeled) {{
            const expr = labels[f.index];
            const pose = poses[f.index] || '';
            const fname = `${{videoBase}}_${{String(f.index).padStart(4, '0')}}.jpg`;
            const b64 = IMAGES[f.index];
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            zip.file(`images/${{fname}}`, bytes);
            csvRows.push(`${{fname}},${{videoBase}},${{expr}},${{pose}},operational`);
        }}
        zip.file('labels.csv', csvRows.join('\\n') + '\\n');

        const blob = await zip.generateAsync({{ type: 'blob' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `anchors_${{videoBase}}.zip`;
        a.click();
    }} catch (e) {{
        alert('Export failed: ' + e.message);
    }} finally {{
        btn.textContent = 'Export ZIP';
        btn.disabled = false;
    }}
}}

function resetLabels() {{
    if (!confirm('Reset all labels?')) return;
    labels = {{}};
    poses = {{}};
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY + '_pose');
    buildFilteredList();
    currentPos = 0;
    renderFocus();
    renderSidebar();
}}

const EXPR_KEYS = {{'1':'cheese', '2':'chill', '3':'edge', '4':'hype', '5':'pass'}};
const POSE_KEYS = {{'q':'front', 'w':'angle', 'e':'side'}};

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
    else if (e.key === 's') {{ if (filteredList[currentPos]) setLabel(filteredList[currentPos].index, 'skip'); }}
    else if (EXPR_KEYS[e.key]) {{
        if (filteredList[currentPos]) setLabel(filteredList[currentPos].index, EXPR_KEYS[e.key]);
    }}
    else if (POSE_KEYS[e.key]) {{
        if (filteredList[currentPos]) setPose(filteredList[currentPos].index, POSE_KEYS[e.key]);
    }}
}});

document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        buildFilteredList();
        currentPos = 0;
        renderFocus();
        renderSidebar();
    }});
}});

buildFilteredList();
renderFocus();
renderSidebar();
</script>
</body></html>"""


def generate_label_html(
    video_path: str | Path,
    fps: int = 2,
    max_frames: int = 500,
    output_path: str | Path = "labels.html",
) -> Path:
    """Generate interactive HTML labeling tool from a video file.

    Requires the ``video`` extra (momentscan) for frame extraction.
    Returns the path to the generated HTML file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    try:
        from momentscan.algorithm.batch.extract import extract_frame_record
    except ImportError:
        raise ImportError(
            "momentscan is required for video frame extraction. "
            "Install with: pip install annotator[video]"
        )

    import momentscan as ms

    logger.info("Processing: %s (fps=%d)", video_path, fps)
    frames_data: list[tuple] = []

    def on_frame(frame, results):
        record = extract_frame_record(frame, results)
        if record is not None:
            frame_bgr = getattr(frame, "data", None)
            if frame_bgr is not None:
                frames_data.append((frame_bgr.copy(), record))
        return True

    ms.run(str(video_path), fps=fps, backend="simple", on_frame=on_frame)
    logger.info("Collected %d frames", len(frames_data))

    if not frames_data:
        logger.error("No frames collected")
        raise RuntimeError("No frames collected from video")

    frames_info = []
    for idx, (frame_bgr, _record) in enumerate(frames_data):
        if idx >= max_frames:
            break
        frames_info.append({
            "index": idx,
            "b64": frame_to_base64(frame_bgr),
            "catalog": "",
            "lr": "",
            "xgb": "",
            "is_disagree": False,
        })

    video_name = video_path.name
    cat_names = ["cheese", "chill", "edge", "hype"]
    html = _generate_html(frames_info, cat_names, video_name)

    output_path.write_text(html, encoding="utf-8")
    logger.info("Label tool saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    logger.info("Open in browser to start labeling. Labels persist in localStorage.")
    return output_path
