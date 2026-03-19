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
        "cheese": "#4CAF50", "chill": "#2196F3", "edge": "#FF5722", "hype": "#9C27B0",
        "cut": "#d32f2f",
        "front": "#00BCD4", "angle": "#FF9800", "side": "#795548",
        "solo": "#607D8B", "duo": "#E91E63",
        "sync": "#FFD700", "interact": "#00E676",
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
.bucket-bar {{ padding: 6px 20px; background: #0a0a1a; border-bottom: 1px solid #333; flex-shrink: 0; }}
.bucket-matrix {{ border-collapse: collapse; font-size: 11px; }}
.bucket-matrix th {{ padding: 3px 8px; color: #888; font-weight: normal; }}
.bucket-matrix td {{ padding: 3px 8px; text-align: center; font-weight: bold; min-width: 40px; }}
.bucket-matrix .row-total {{ border-left: 1px solid #333; color: #aaa; }}
.bucket-matrix .col-total {{ border-top: 1px solid #333; color: #aaa; }}
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

<div class="bucket-bar" id="bucketBar"></div>
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
        const lbl = labels[f.index];
        const isLabeled = lbl !== undefined;
        const borderColor = isLabeled ? (getColor(lbl) || '#4CAF50') : 'transparent';
        div.className = 'thumb' + (pos === currentPos ? ' active' : '');
        div.style.borderLeftColor = borderColor;
        div.style.borderLeftWidth = isLabeled ? '4px' : '3px';
        div.style.opacity = isLabeled ? '0.7' : (pos === currentPos ? '1' : '0.5');
        const tag = isLabeled ? `<div style="font-size:9px;color:${{borderColor}};text-align:center;margin-top:1px">${{lbl}}</div>` : '';
        div.innerHTML = `<img src="data:image/jpeg;base64,${{IMAGES[f.index]}}" loading="lazy">${{tag}}`;
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
    const scene = scenes[f.index];
    const chem = chemistries[f.index];
    const isAccepted = label && label !== 'cut' && label !== '__shoot__';
    const parts = [scene, label, pose, chem].filter(x => x && x !== '__shoot__');
    const labelHtml = parts.length > 0
        ? `<div class="focus-label">${{parts.join(' + ')}}</div>`
        : `<div class="focus-label" style="color:#888">Unlabeled</div>`;

    const K = '<span style="font-size:10px;opacity:0.5;margin-left:4px">';
    const FOCUS = 'border:3px solid #e94560;';
    let btnsHtml = '';

    // Determine current step: shoot → scene → chemistry(duo) → expression → pose
    const step = !label ? 0
        : label === 'cut' ? -1
        : !scene ? 1
        : (scene === 'duo' && !chem) ? 2
        : (label === '__shoot__' || !isAccepted) ? 3
        : !pose ? 4
        : 5;  // complete

    if (step === -1) {{
        // cut — 변경 가능
        btnsHtml += '<div class="buttons">';
        btnsHtml += `<button class="cat-btn" style="background:#4CAF50;color:#fff;padding:10px 24px" onclick="setLabel(${{f.index}},'__shoot__')">→ SHOOT ${{K}}1</span></button>`;
        btnsHtml += `<button class="cat-btn selected" style="background:#d32f2f;color:#fff">cut ✂️</button>`;
        btnsHtml += '</div>';
    }} else {{
        // Step 0: SHOOT / CUT
        btnsHtml += `<div class="buttons" style="${{step === 0 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
        btnsHtml += `<button class="cat-btn" style="${{label && label !== 'cut' ? 'background:#4CAF50;color:#fff;' : ''}}padding:${{step === 0 ? '12px 32px;font-size:16px' : '8px 16px'}}" onclick="setLabel(${{f.index}},'__shoot__')">SHOOT 📸 ${{K}}1</span></button>`;
        btnsHtml += `<button class="cat-btn" style="background:#333;color:#aaa;padding:${{step === 0 ? '12px 32px;font-size:16px' : '8px 16px'}}" onclick="setLabel(${{f.index}},'cut')">CUT ✂️ ${{K}}2</span></button>`;
        btnsHtml += '</div>';

        if (step >= 1) {{
            // Step 1: SCENE (solo/duo)
            btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 1 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
            const SCENE_MAP = [['solo','1'],['duo','2']];
            for (const [s, key] of SCENE_MAP) {{
                const sel = scene === s;
                const bg = sel ? `background:${{getColor(s)}};color:#fff;` : '';
                btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setScene(${{f.index}},'${{s}}')">${{s}} ${{K}}${{key}}</span></button>`;
            }}
            btnsHtml += '</div>';
        }}

        if (step === 2) {{
            // Step 2: CHEMISTRY (duo only)
            btnsHtml += `<div class="buttons" style="margin-top:6px;${{FOCUS}}padding:4px;border-radius:8px;">`;
            const CHEM_MAP = [['sync','1'],['interact','2']];
            for (const [c, key] of CHEM_MAP) {{
                const sel = chem === c;
                const bg = sel ? `background:${{getColor(c)}};color:#fff;` : '';
                btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setChemistry(${{f.index}},'${{c}}')">${{c}} ${{K}}${{key}}</span></button>`;
            }}
            btnsHtml += '</div>';
        }} else if (step > 2 && scene === 'duo' && chem) {{
            // Show chemistry as selected (not focused)
            btnsHtml += `<div class="buttons" style="margin-top:6px">`;
            const CHEM_MAP = [['sync','1'],['interact','2']];
            for (const [c, key] of CHEM_MAP) {{
                const sel = chem === c;
                const bg = sel ? `background:${{getColor(c)}};color:#fff;` : '';
                btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setChemistry(${{f.index}},'${{c}}')">${{c}} ${{K}}${{key}}</span></button>`;
            }}
            btnsHtml += '</div>';
        }}

        if (step >= 3) {{
            // Step 3: EXPRESSION
            btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 3 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
            const EXPR_MAP = [['cheese', '1'],['chill', '2'],['edge', '3'],['hype', '4']];
            for (const [cat, key] of EXPR_MAP) {{
                const sel = label === cat;
                const bg = sel ? `background:${{getColor(cat)}};color:#fff;` : '';
                btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setLabel(${{f.index}},'${{cat}}')">${{cat}} ${{K}}${{key}}</span></button>`;
            }}
            btnsHtml += '</div>';
        }}

        if (step >= 4) {{
            // Step 4: POSE
            btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 4 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
            const POSE_MAP = [['front','1'],['angle','2'],['side','3']];
            for (const [p, key] of POSE_MAP) {{
                const sel = pose === p;
                const bg = sel ? `background:${{getColor(p)}};color:#fff;` : '';
                btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setPose(${{f.index}},'${{p}}')">${{p}} ${{K}}${{key}}</span></button>`;
            }}
            btnsHtml += '</div>';
        }}

        if (step === 5) {{
            // Complete — 자동 다음 프레임 (0.3초 후)
            btnsHtml += '<div style="color:#4CAF50;font-size:13px;margin-top:8px;text-align:center">✓ Complete</div>';
        }}
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
            <button onclick="go(-1)" ${{currentPos <= 0 ? 'disabled' : ''}}>Prev ${{K}}K</span></button>
            <div class="pos">${{currentPos + 1}} / ${{filteredList.length}}</div>
            <button onclick="go(1)" ${{currentPos >= filteredList.length - 1 ? 'disabled' : ''}}>Next ${{K}}J</span></button>
        </div>
        <div class="shortcut-hint">1-4 = 현재 포커스 step 선택 | J next K prev | shoot→scene→chemistry→expression→pose</div>
    `;
    updateCount();
}}

function go(delta) {{
    currentPos = Math.max(0, Math.min(filteredList.length - 1, currentPos + delta));
    renderFocus();
    renderSidebar();
}}

let poses = JSON.parse(localStorage.getItem(STORAGE_KEY + '_pose') || '{{}}');
let scenes = JSON.parse(localStorage.getItem(STORAGE_KEY + '_scene') || '{{}}');
let chemistries = JSON.parse(localStorage.getItem(STORAGE_KEY + '_chem') || '{{}}');

function setLabel(index, label) {{
    if (labels[index] === label) {{
        delete labels[index];
    }} else {{
        labels[index] = label;
    }}
    localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    // CUT → 자동 다음 프레임
    if (label === 'cut') {{
        setTimeout(() => {{ go(1); }}, 300);
    }}
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
    checkAutoAdvance(index);
    renderFocus();
    renderSidebar();
}}

function setScene(index, scene) {{
    if (scenes[index] === scene) {{
        delete scenes[index];
    }} else {{
        scenes[index] = scene;
        // solo면 chemistry 자동 클리어
        if (scene === 'solo') delete chemistries[index];
    }}
    localStorage.setItem(STORAGE_KEY + '_scene', JSON.stringify(scenes));
    localStorage.setItem(STORAGE_KEY + '_chem', JSON.stringify(chemistries));
    renderFocus();
    renderSidebar();
}}

function setChemistry(index, chem) {{
    if (chemistries[index] === chem) {{
        delete chemistries[index];
    }} else {{
        chemistries[index] = chem;
    }}
    localStorage.setItem(STORAGE_KEY + '_chem', JSON.stringify(chemistries));
    checkAutoAdvance(index);
    renderFocus();
    renderSidebar();
}}

function checkAutoAdvance(index) {{
    // 모든 라벨이 완성되었으면 0.4초 후 다음 프레임으로 자동 이동
    const lbl = labels[index];
    const p = poses[index];
    const s = scenes[index];
    const c = chemistries[index];
    const isComplete = lbl && lbl !== '__shoot__' && lbl !== 'cut' && p && s && (s !== 'duo' || c);
    if (isComplete) {{
        setTimeout(() => {{ go(1); }}, 400);
    }}
}}

function focusBucket(axis, value) {{
    // Find first unlabeled frame for this axis
    const axisData = axis === 'expression' ? labels : axis === 'pose' ? poses : axis === 'scene' ? scenes : chemistries;
    const target = filteredList.find((f, pos) => {{
        if (value === '(none)' || value === '') {{
            return axisData[f.index] === undefined;
        }}
        return axisData[f.index] === value;
    }});
    if (target) {{
        const pos = filteredList.indexOf(target);
        if (pos >= 0) {{ currentPos = pos; renderFocus(); renderSidebar(); }}
    }}
}}

function updateCount() {{
    const total = Object.keys(labels).length;
    document.getElementById('count').textContent = total;

    // Count all axes
    const EXPRS = ['cheese','chill','edge','hype','cut'];
    const POSES = ['front','angle','side',''];
    const SCENES = ['solo','duo',''];
    const CHEMS = ['sync','interact',''];

    const exprCounts = {{}};
    const poseCounts = {{}};
    const sceneCounts = {{}};
    const chemCounts = {{}};
    EXPRS.forEach(e => exprCounts[e] = 0);
    POSES.forEach(p => poseCounts[p] = 0);
    SCENES.forEach(s => sceneCounts[s] = 0);
    CHEMS.forEach(c => chemCounts[c] = 0);

    const exprPose = {{}};
    EXPRS.forEach(e => POSES.forEach(p => exprPose[e+'|'+p] = 0));

    for (const [idx, lbl] of Object.entries(labels)) {{
        if (!lbl || lbl === '__shoot__') continue;
        const p = poses[idx] || '';
        const s = scenes[idx] || '';
        const c = chemistries[idx] || '';
        if (exprCounts[lbl] !== undefined) exprCounts[lbl]++;
        poseCounts[p] = (poseCounts[p] || 0) + 1;
        sceneCounts[s] = (sceneCounts[s] || 0) + 1;
        chemCounts[c] = (chemCounts[c] || 0) + 1;
        const key = lbl + '|' + p;
        if (exprPose[key] !== undefined) exprPose[key]++;
    }}

    const unlabeled = FRAMES.length - Object.keys(labels).length;
    const bar = document.getElementById('bucketBar');

    // Expression × Pose matrix
    let html = '<table class="bucket-matrix"><tr><th></th>';
    ['front','angle','side','(none)'].forEach((p, i) => {{
        const pk = POSES[i];
        const c = pk ? getColor(pk) : '#666';
        html += `<th style="color:${{c}};cursor:pointer" onclick="focusBucket('pose','${{pk}}')">${{p}}</th>`;
    }});
    html += '<th class="row-total">total</th></tr>';

    EXPRS.forEach(e => {{
        const ec = getColor(e);
        html += `<tr><th style="color:${{ec}};cursor:pointer" onclick="focusBucket('expression','${{e}}')">${{e}}</th>`;
        POSES.forEach(p => {{
            const v = exprPose[e+'|'+p];
            const style = v > 0 ? `color:${{ec}};font-size:13px` : 'color:#333';
            html += `<td style="${{style}}">${{v}}</td>`;
        }});
        const t = exprCounts[e];
        html += `<td class="row-total" style="${{t > 0 ? 'color:'+ec : 'color:#333'}}">${{t}}</td></tr>`;
    }});

    // Totals row
    const grand = Object.values(exprCounts).reduce((a,b) => a+b, 0);
    html += '<tr><th class="col-total">total</th>';
    POSES.forEach(p => html += `<td class="col-total">${{poseCounts[p] || 0}}</td>`);
    html += `<td class="col-total" style="color:#4CAF50;font-weight:bold">${{grand}}</td></tr>`;

    // Unlabeled row (clickable)
    html += `<tr><th style="color:#888;cursor:pointer" onclick="focusBucket('expression','')">(unlabeled)</th>`;
    html += `<td colspan="4" style="color:#e94560;cursor:pointer;text-align:left;padding-left:12px" onclick="focusBucket('expression','')">${{unlabeled}} frames</td></tr>`;

    html += '</table>';

    // Scene + Chemistry summary (inline, right side)
    html += '&nbsp;&nbsp;<span style="font-size:11px;color:#888">';
    html += `scene: <span style="color:${{getColor('solo')}};cursor:pointer" onclick="focusBucket('scene','solo')">solo ${{sceneCounts['solo'] || 0}}</span>`;
    html += ` <span style="color:${{getColor('duo')}};cursor:pointer" onclick="focusBucket('scene','duo')">duo ${{sceneCounts['duo'] || 0}}</span>`;
    html += ` <span style="color:#555;cursor:pointer" onclick="focusBucket('scene','')">(none) ${{sceneCounts[''] || 0}}</span>`;
    html += ` &nbsp;chem: <span style="color:${{getColor('sync')}};cursor:pointer" onclick="focusBucket('chemistry','sync')">sync ${{chemCounts['sync'] || 0}}</span>`;
    html += ` <span style="color:${{getColor('interact')}};cursor:pointer" onclick="focusBucket('chemistry','interact')">interact ${{chemCounts['interact'] || 0}}</span>`;
    html += '</span>';

    bar.innerHTML = html;
}}

async function exportLabels() {{
    const labeled = FRAMES.filter(f => labels[f.index] && labels[f.index] !== '__shoot__');
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

        const csvRows = ['filename,member_id,expression,pose,scene,chemistry,source'];
        for (const f of labeled) {{
            const expr = labels[f.index];
            const pose = poses[f.index] || '';
            const scene = scenes[f.index] || '';
            const chem = chemistries[f.index] || '';
            const fname = `${{videoBase}}_${{String(f.index).padStart(4, '0')}}.jpg`;
            const b64 = IMAGES[f.index];
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            zip.file(`images/${{fname}}`, bytes);
            csvRows.push(`${{fname}},${{videoBase}},${{expr}},${{pose}},${{scene}},${{chem}},operational`);
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
    scenes = {{}};
    chemistries = {{}};
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY + '_pose');
    localStorage.removeItem(STORAGE_KEY + '_scene');
    localStorage.removeItem(STORAGE_KEY + '_chem');
    buildFilteredList();
    currentPos = 0;
    renderFocus();
    renderSidebar();
}}

// Step-based numeric input: 1,2,3,4 applies to the currently focused step
const STEP_OPTIONS = {{
    '-1': [['__shoot__','setLabel']],                                       // cut → undo
    0: [['__shoot__','setLabel'], ['cut','setLabel']],                     // shoot/cut
    1: [['solo','setScene'], ['duo','setScene']],                          // scene
    2: [['sync','setChemistry'], ['interact','setChemistry']],             // chemistry
    3: [['cheese','setLabel'], ['chill','setLabel'], ['edge','setLabel'], ['hype','setLabel']],  // expression
    4: [['front','setPose'], ['angle','setPose'], ['side','setPose']],     // pose
}};

function getStep(idx) {{
    const lbl = labels[idx];
    const s = scenes[idx];
    const c = chemistries[idx];
    const p = poses[idx];
    const isAccepted = lbl && lbl !== 'cut' && lbl !== '__shoot__';
    if (!lbl) return 0;
    if (lbl === 'cut') return -1;
    if (!s) return 1;
    if (s === 'duo' && !c) return 2;
    if (lbl === '__shoot__' || !isAccepted) return 3;
    if (!p) return 4;
    return 5;
}}

document.addEventListener('keydown', e => {{
    const f = filteredList[currentPos];
    if (!f) return;
    const idx = f.index;
    const step = getStep(idx);

    if (e.key === 'j') go(1);
    else if (e.key === 'k') go(-1);
    else {{
        const n = parseInt(e.key);
        if (n >= 1 && step >= 0 && STEP_OPTIONS[step]) {{
            const opts = STEP_OPTIONS[step];
            if (n <= opts.length) {{
                const [value, fn] = opts[n - 1];
                if (fn === 'setLabel') setLabel(idx, value);
                else if (fn === 'setScene') setScene(idx, value);
                else if (fn === 'setChemistry') setChemistry(idx, value);
                else if (fn === 'setPose') setPose(idx, value);
            }}
        }}
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
