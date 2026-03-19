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
.main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
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
.strip {{ display: flex; overflow-x: auto; background: #0f0f23; border-top: 1px solid #333;
    flex-shrink: 0; padding: 4px; gap: 3px; }}
.thumb {{ flex-shrink: 0; cursor: pointer; opacity: 0.5; transition: opacity .15s;
    border-bottom: 3px solid transparent; position: relative; }}
.thumb:hover {{ opacity: 0.8; }}
.thumb.active {{ opacity: 1; border-bottom-color: #e94560; }}
.thumb img {{ height: 56px; border-radius: 3px; display: block; }}
.shortcut-hint {{ font-size: 11px; color: #555; text-align: center; margin-top: 8px; }}
.bucket-bar {{ padding: 6px 20px; background: #0a0a1a; border-bottom: 1px solid #333; flex-shrink: 0; }}
.timeline-bar {{
    display: flex; height: 16px; background: #111;
    border-bottom: 1px solid #333; flex-shrink: 0;
    cursor: pointer;
}}
.timeline-bar .tl-frame {{
    flex: 1; min-width: 1px;
    transition: opacity 0.1s;
}}
.bucket-matrix {{ border-collapse: collapse; font-size: 11px; }}
.bucket-matrix th {{ padding: 3px 8px; color: #888; font-weight: normal; }}
.bucket-matrix td {{ padding: 3px 8px; text-align: center; font-weight: bold; min-width: 40px; }}
.bucket-matrix .row-total {{ border-left: 1px solid #333; color: #aaa; }}
.bucket-matrix .col-total {{ border-top: 1px solid #333; color: #aaa; }}
.modal-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.6); z-index: 9999; display: flex; align-items: center;
    justify-content: center; }}
.modal-box {{ background: #1a1a2e; border: 2px solid #e94560; border-radius: 12px;
    padding: 32px 40px; min-width: 360px; max-width: 480px; }}
.modal-box h2 {{ margin: 0 0 20px; color: #e94560; font-size: 20px; }}
.modal-row {{ margin: 12px 0; }}
.modal-row label {{ display: block; font-size: 13px; color: #aaa; margin-bottom: 6px; }}
.modal-opts {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.modal-opt {{ padding: 6px 16px; border: 2px solid #444; border-radius: 6px; background: #222;
    color: #ccc; cursor: pointer; font-size: 13px; transition: all .15s; }}
.modal-opt:hover {{ background: #444; color: #fff; }}
.modal-opt.selected {{ background: #e94560; color: #fff; border-color: #e94560; }}
.modal-start {{ margin-top: 24px; padding: 10px 32px; border: none; border-radius: 6px;
    background: #4CAF50; color: #fff; font-size: 15px; cursor: pointer; width: 100%; }}
.modal-start:hover {{ background: #45a049; }}
.modal-start:disabled {{ opacity: 0.4; cursor: default; }}
.video-meta-indicator {{ font-size: 12px; color: #aaa; background: #222; padding: 3px 10px;
    border-radius: 3px; border: 1px solid #444; }}
.btn-meta {{ background: #444; color: #ccc; font-size: 12px; padding: 4px 10px; }}
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
    <span class="video-meta-indicator" id="metaIndicator"></span>
    <button class="btn-meta" onclick="showMetaModal()">Edit Meta</button>
    <button class="btn-export" onclick="exportLabels()">Export ZIP</button>
    <button class="btn-reset" onclick="resetLabels()">Reset All</button>
</div>

<div class="bucket-bar" id="bucketBar"></div>
<div class="timeline-bar" id="timelineBar"></div>
<div class="main">
    <div class="focus-panel" id="focus"></div>
    <div class="strip" id="strip"></div>
</div>

<div class="modal-overlay" id="metaModal" style="display:none">
    <div class="modal-box">
        <h2>Video Setup: {video_name}</h2>
        <div class="modal-row">
            <label>Scene</label>
            <div class="modal-opts" id="metaScene">
                <div class="modal-opt" data-value="solo" onclick="selectMetaOpt('scene','solo')">
                    <span style="opacity:0.5;font-size:11px">1 </span>solo</div>
                <div class="modal-opt" data-value="duo" onclick="selectMetaOpt('scene','duo')">
                    <span style="opacity:0.5;font-size:11px">2 </span>duo</div>
            </div>
        </div>
        <div class="modal-row">
            <label>Main Gender</label>
            <div class="modal-opts" id="metaMain_gender">
                <div class="modal-opt" data-value="male" onclick="selectMetaOpt('main_gender','male')">
                    <span style="opacity:0.5;font-size:11px">1 </span>male</div>
                <div class="modal-opt" data-value="female" onclick="selectMetaOpt('main_gender','female')">
                    <span style="opacity:0.5;font-size:11px">2 </span>female</div>
            </div>
        </div>
        <div class="modal-row">
            <label>Main Ethnicity</label>
            <div class="modal-opts" id="metaMain_ethnicity">
                <div class="modal-opt" data-value="asian" onclick="selectMetaOpt('main_ethnicity','asian')">
                    <span style="opacity:0.5;font-size:11px">1 </span>asian</div>
                <div class="modal-opt" data-value="western" onclick="selectMetaOpt('main_ethnicity','western')">
                    <span style="opacity:0.5;font-size:11px">2 </span>western</div>
                <div class="modal-opt" data-value="other" onclick="selectMetaOpt('main_ethnicity','other')">
                    <span style="opacity:0.5;font-size:11px">3 </span>other</div>
            </div>
        </div>
        <div class="modal-row" id="passengerGenderRow" style="display:none">
            <label>Passenger Gender</label>
            <div class="modal-opts" id="metaPassenger_gender">
                <div class="modal-opt" data-value="male" onclick="selectMetaOpt('passenger_gender','male')">
                    <span style="opacity:0.5;font-size:11px">1 </span>male</div>
                <div class="modal-opt" data-value="female" onclick="selectMetaOpt('passenger_gender','female')">
                    <span style="opacity:0.5;font-size:11px">2 </span>female</div>
            </div>
        </div>
        <div class="modal-row" id="passengerEthnicityRow" style="display:none">
            <label>Passenger Ethnicity</label>
            <div class="modal-opts" id="metaPassenger_ethnicity">
                <div class="modal-opt" data-value="asian" onclick="selectMetaOpt('passenger_ethnicity','asian')">
                    <span style="opacity:0.5;font-size:11px">1 </span>asian</div>
                <div class="modal-opt" data-value="western" onclick="selectMetaOpt('passenger_ethnicity','western')">
                    <span style="opacity:0.5;font-size:11px">2 </span>western</div>
                <div class="modal-opt" data-value="other" onclick="selectMetaOpt('passenger_ethnicity','other')">
                    <span style="opacity:0.5;font-size:11px">3 </span>other</div>
            </div>
        </div>
        <button class="modal-start" id="metaStartBtn" onclick="saveMetaAndStart()" disabled>Start Labeling</button>
    </div>
</div>

<script>
const FRAMES = {frames_json};
const IMAGES = {{{img_entries}}};
const CATEGORIES = {cat_list_json};
const CAT_COLORS = {cat_colors_json};
const STORAGE_KEY = "label_tool_{video_name.replace('.', '_')}";
const META_KEY = STORAGE_KEY + '_video_meta';

let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'  );
let currentFilter = 'all';
let filteredList = [];
let currentPos = 0;
let manualStep = null;  // null = auto, number = manual override

let videoMeta = JSON.parse(localStorage.getItem(META_KEY) || 'null');
let metaDraft = {{ scene: null, main_gender: null, main_ethnicity: null, passenger_gender: null, passenger_ethnicity: null }};
let modalMode = 'field';  // 'field' or 'value'
let modalField = 0;       // 0=scene, 1=main_gender, 2=main_ethnicity, 3=passenger_gender, 4=passenger_ethnicity
const MODAL_FIELDS_SOLO = ['scene', 'main_gender', 'main_ethnicity'];
const MODAL_FIELDS_DUO = ['scene', 'main_gender', 'main_ethnicity', 'passenger_gender', 'passenger_ethnicity'];
function getModalFields() {{ return metaDraft.scene === 'duo' ? MODAL_FIELDS_DUO : MODAL_FIELDS_SOLO; }}
const MODAL_VALUES = {{
    scene: ['solo', 'duo'],
    main_gender: ['male', 'female'],
    main_ethnicity: ['asian', 'western', 'other'],
    passenger_gender: ['male', 'female'],
    passenger_ethnicity: ['asian', 'western', 'other'],
}};

function getColor(cat) {{ return CAT_COLORS[cat] || '#666'; }}

function updateMetaIndicator() {{
    const el = document.getElementById('metaIndicator');
    if (videoMeta) {{
        if (videoMeta.scene === 'duo') {{
            el.textContent = 'duo | M:' + videoMeta.main_gender + '/' + videoMeta.main_ethnicity
                + ' P:' + videoMeta.passenger_gender + '/' + videoMeta.passenger_ethnicity;
        }} else {{
            el.textContent = 'solo | ' + videoMeta.main_gender + ' | ' + videoMeta.main_ethnicity;
        }}
    }} else {{
        el.textContent = '(no video meta)';
        el.style.color = '#e94560';
    }}
}}

function selectMetaOpt(field, value) {{
    metaDraft[field] = value;
    if (field === 'scene') {{
        const pGR = document.getElementById('passengerGenderRow');
        const pER = document.getElementById('passengerEthnicityRow');
        if (value === 'duo') {{
            pGR.style.display = '';
            pER.style.display = '';
        }} else {{
            pGR.style.display = 'none';
            pER.style.display = 'none';
            metaDraft.passenger_gender = null;
            metaDraft.passenger_ethnicity = null;
        }}
    }}
    renderMetaSelections();
}}

function getMetaContainerId(field) {{
    // main_gender -> Main_gender, passenger_ethnicity -> Passenger_ethnicity
    return 'meta' + field.charAt(0).toUpperCase() + field.slice(1);
}}

function renderMetaSelections() {{
    const allFields = MODAL_FIELDS_DUO;
    for (const field of allFields) {{
        const container = document.getElementById(getMetaContainerId(field));
        if (!container) continue;
        container.querySelectorAll('.modal-opt').forEach(el => {{
            el.classList.toggle('selected', el.dataset.value === metaDraft[field]);
        }});
    }}
    // Highlight focused field
    const fields = getModalFields();
    allFields.forEach((field) => {{
        const container = document.getElementById(getMetaContainerId(field));
        if (!container) return;
        const idx = fields.indexOf(field);
        container.parentElement.style.background = (idx >= 0 && modalField === idx)
            ? 'rgba(233,69,96,0.1)' : '';
        container.parentElement.style.borderRadius = '6px';
    }});
    const btn = document.getElementById('metaStartBtn');
    const isDuo = metaDraft.scene === 'duo';
    const baseOk = metaDraft.scene && metaDraft.main_gender && metaDraft.main_ethnicity;
    const passengerOk = !isDuo || (metaDraft.passenger_gender && metaDraft.passenger_ethnicity);
    btn.disabled = !(baseOk && passengerOk);
}}

function showMetaModal() {{
    metaDraft = videoMeta
        ? {{ scene: videoMeta.scene, main_gender: videoMeta.main_gender, main_ethnicity: videoMeta.main_ethnicity,
             passenger_gender: videoMeta.passenger_gender || null, passenger_ethnicity: videoMeta.passenger_ethnicity || null }}
        : {{ scene: null, main_gender: null, main_ethnicity: null, passenger_gender: null, passenger_ethnicity: null }};
    modalMode = 'field';
    modalField = 0;
    // Show/hide passenger rows
    const isDuo = metaDraft.scene === 'duo';
    document.getElementById('passengerGenderRow').style.display = isDuo ? '' : 'none';
    document.getElementById('passengerEthnicityRow').style.display = isDuo ? '' : 'none';
    document.getElementById('metaModal').style.display = 'flex';
    renderMetaSelections();
}}

function hideMetaModal() {{
    document.getElementById('metaModal').style.display = 'none';
}}

function saveMetaAndStart() {{
    const isDuo = metaDraft.scene === 'duo';
    if (!metaDraft.scene || !metaDraft.main_gender || !metaDraft.main_ethnicity) return;
    if (isDuo && (!metaDraft.passenger_gender || !metaDraft.passenger_ethnicity)) return;
    const videoBase = "{video_name}".replace(/\\.[^.]+$/, '');
    videoMeta = {{
        video_id: videoBase,
        scene: metaDraft.scene,
        main_gender: metaDraft.main_gender,
        main_ethnicity: metaDraft.main_ethnicity,
    }};
    if (isDuo) {{
        videoMeta.passenger_gender = metaDraft.passenger_gender;
        videoMeta.passenger_ethnicity = metaDraft.passenger_ethnicity;
    }}
    localStorage.setItem(META_KEY, JSON.stringify(videoMeta));
    hideMetaModal();
    updateMetaIndicator();
    buildFilteredList();
    renderFocus();
    renderStrip();
}}

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

function renderStrip() {{
    const sb = document.getElementById('strip');
    sb.innerHTML = '';
    filteredList.forEach((f, pos) => {{
        const div = document.createElement('div');
        const lbl = labels[f.index];
        const isLabeled = lbl !== undefined;
        const borderColor = isLabeled ? (getColor(lbl) || '#4CAF50') : 'transparent';
        div.className = 'thumb' + (pos === currentPos ? ' active' : '');
        div.style.borderBottomColor = borderColor;
        div.style.opacity = isLabeled ? '0.7' : (pos === currentPos ? '1' : '0.5');
        div.innerHTML = `<img src="data:image/jpeg;base64,${{IMAGES[f.index]}}" loading="lazy">`;
        div.onclick = () => {{ currentPos = pos; renderFocus(); renderStrip(); }};
        sb.appendChild(div);
        if (pos === currentPos) {{
            requestAnimationFrame(() => {{
                const strip = document.getElementById('strip');
                const left = div.offsetLeft - strip.clientWidth / 2 + div.offsetWidth / 2;
                strip.scrollLeft = left;
            }});
        }}
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
    const scene = videoMeta ? videoMeta.scene : null;
    const chem = chemistries[f.index];
    const isDuo = scene === 'duo';
    const isAccepted = label && label !== 'cut' && label !== '__shoot__';
    const parts = [label, pose, chem].filter(x => x && x !== '__shoot__');
    const labelHtml = parts.length > 0
        ? `<div class="focus-label">${{parts.join(' + ')}}</div>`
        : `<div class="focus-label" style="color:#888">Unlabeled</div>`;

    const K = '<span style="font-size:10px;opacity:0.5;margin-left:4px">';
    const FOCUS = 'border:3px solid #e94560;';
    const DESC = {{
        '__shoot__': '이 프레임을 촬영합니다',
        'cut': '이 장면은 찍으면 안 됩니다',
        'sync': '함께 웃기, 동시 반응 — 둘의 타이밍이 맞는 순간',
        'interact': '서로 쳐다보기, 하이파이브 — 둘이 교감하는 순간',
        'cheese': '얼굴이 주인공 — 프로필 사진, 인물 초상화용 이쁜 미소',
        'chill': '쿨하고 여유로운 — 편안하고 힘 빠진 자연스러운 표정',
        'edge': '날카롭고 강렬한 — 인상 찡그리는데 간지나는 제임스딘',
        'hype': '순간이 주인공 — 역동적 액션, 에너지 폭발 카타르시스',
        'front': '정면 — 카메라를 바라보는 앵글',
        'angle': '3/4 앵글 — 약간 돌린 자연스러운 각도',
        'side': '측면 — 프로필 샷',
    }};
    const descHtml = (val) => val && DESC[val] ? `<div style="font-size:11px;color:#888;margin-top:2px;text-align:center">${{DESC[val]}}</div>` : '';
    let btnsHtml = '';

    // Determine current step: shoot → chemistry(duo only) → expression → pose
    const autoStep = getStep(f.index);
    const step = (manualStep !== null && manualStep >= -1) ? manualStep : autoStep;

    // Step 0: SHOOT / CUT (항상 표시)
    const isShot = label && label !== 'cut';
    const isCut = label === 'cut';
    btnsHtml += `<div class="buttons" style="${{step === 0 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
    btnsHtml += `<button class="cat-btn${{isShot ? ' selected' : ''}}" style="${{isShot ? 'background:#4CAF50;color:#fff;' : ''}}padding:8px 18px" onclick="setLabel(${{f.index}},'__shoot__')">SHOOT 📸 ${{K}}Q</span></button>`;
    btnsHtml += `<button class="cat-btn${{isCut ? ' selected' : ''}}" style="${{isCut ? 'background:#d32f2f;color:#fff;' : 'background:#333;color:#aaa;'}}padding:8px 18px" onclick="setLabel(${{f.index}},'cut')">CUT ✂️ ${{K}}W</span></button>`;
    btnsHtml += '</div>';

    // Step 1: CHEMISTRY (duo only, 항상 표시)
    if (isDuo) {{
        btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 1 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
        for (const [c, key] of [['sync','Q'],['interact','W']]) {{
            const sel = chem === c;
            const bg = sel ? `background:${{getColor(c)}};color:#fff;` : '';
            btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setChemistry(${{f.index}},'${{c}}')">${{c}} ${{K}}${{key}}</span></button>`;
        }}
        btnsHtml += '</div>';
        btnsHtml += descHtml(chem);
    }}

    // Step 2: EXPRESSION (항상 표시)
    btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 2 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
    for (const [cat, key] of [['cheese','Q'],['chill','W'],['edge','E'],['hype','R']]) {{
        const sel = label === cat;
        const bg = sel ? `background:${{getColor(cat)}};color:#fff;` : '';
        btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setLabel(${{f.index}},'${{cat}}')">${{cat}} ${{K}}${{key}}</span></button>`;
    }}
    btnsHtml += '</div>';
    btnsHtml += descHtml(isAccepted ? label : null);

    // Step 3: POSE (항상 표시)
    btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 3 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
    for (const [p, key] of [['front','Q'],['angle','W'],['side','E']]) {{
        const sel = pose === p;
        const bg = sel ? `background:${{getColor(p)}};color:#fff;` : '';
        btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setPose(${{f.index}},'${{p}}')">${{p}} ${{K}}${{key}}</span></button>`;
    }}
    btnsHtml += '</div>';
    btnsHtml += descHtml(pose);

    if (step === 4) {{
        btnsHtml += '<div style="color:#4CAF50;font-size:13px;margin-top:8px;text-align:center">✓ Complete</div>';
    }}

    const stepHint = isDuo ? 'shoot→chemistry→expression→pose' : 'shoot→expression→pose';

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
            <button onclick="go(-1)" ${{currentPos <= 0 ? 'disabled' : ''}}>Prev ${{K}}H</span></button>
            <div class="pos">${{currentPos + 1}} / ${{filteredList.length}}</div>
            <button onclick="go(1)" ${{currentPos >= filteredList.length - 1 ? 'disabled' : ''}}>Next ${{K}}L</span></button>
        </div>
        <div class="shortcut-hint">H prev L next | J step↓ K step↑ | Q W E R = 선택 | ${{stepHint}}</div>
    `;
    updateCount();
    renderTimeline();
}}

function go(delta) {{
    currentPos = Math.max(0, Math.min(filteredList.length - 1, currentPos + delta));
    manualStep = null;  // 프레임 이동 시 자동 step으로 복귀
    renderFocus();
    renderStrip();
}}

let poses = JSON.parse(localStorage.getItem(STORAGE_KEY + '_pose') || '{{}}');
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
    renderStrip();
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
    renderStrip();
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
    renderStrip();
}}

function checkAutoAdvance(index) {{
    // 모든 라벨이 완성되었으면 0.4초 후 다음 프레임으로 자동 이동
    const lbl = labels[index];
    const p = poses[index];
    const c = chemistries[index];
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const isComplete = lbl && lbl !== '__shoot__' && lbl !== 'cut' && p && (!isDuo || c);
    if (isComplete) {{
        setTimeout(() => {{ go(1); }}, 400);
    }}
}}

function focusBucket(axis, value) {{
    const axisData = axis === 'expression' ? labels : axis === 'pose' ? poses : chemistries;
    const target = filteredList.find((f, pos) => {{
        if (value === '(none)' || value === '') {{
            return axisData[f.index] === undefined;
        }}
        return axisData[f.index] === value;
    }});
    if (target) {{
        const pos = filteredList.indexOf(target);
        if (pos >= 0) {{ currentPos = pos; renderFocus(); renderStrip(); }}
    }}
}}

function updateCount() {{
    const total = Object.keys(labels).length;
    document.getElementById('count').textContent = total;

    const isDuo = videoMeta && videoMeta.scene === 'duo';

    // Count all axes
    const EXPRS = ['cheese','chill','edge','hype','cut'];
    const POSES = ['front','angle','side',''];
    const CHEMS = ['sync','interact',''];

    const exprCounts = {{}};
    const poseCounts = {{}};
    const chemCounts = {{}};
    EXPRS.forEach(e => exprCounts[e] = 0);
    POSES.forEach(p => poseCounts[p] = 0);
    CHEMS.forEach(c => chemCounts[c] = 0);

    const exprPose = {{}};
    EXPRS.forEach(e => POSES.forEach(p => exprPose[e+'|'+p] = 0));

    for (const [idx, lbl] of Object.entries(labels)) {{
        if (!lbl || lbl === '__shoot__') continue;
        const p = poses[idx] || '';
        const c = chemistries[idx] || '';
        if (exprCounts[lbl] !== undefined) exprCounts[lbl]++;
        poseCounts[p] = (poseCounts[p] || 0) + 1;
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

    // Chemistry summary (inline, right side) — only for duo
    if (isDuo) {{
        html += '&nbsp;&nbsp;<span style="font-size:11px;color:#888">';
        html += `chem: <span style="color:${{getColor('sync')}};cursor:pointer" onclick="focusBucket('chemistry','sync')">sync ${{chemCounts['sync'] || 0}}</span>`;
        html += ` <span style="color:${{getColor('interact')}};cursor:pointer" onclick="focusBucket('chemistry','interact')">interact ${{chemCounts['interact'] || 0}}</span>`;
        html += '</span>';
    }}

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

        const csvRows = ['filename,video_id,expression,pose,chemistry,source'];
        for (const f of labeled) {{
            const expr = labels[f.index];
            const pose = poses[f.index] || '';
            const chem = chemistries[f.index] || '';
            const fname = `${{videoBase}}_${{String(f.index).padStart(4, '0')}}.jpg`;
            const b64 = IMAGES[f.index];
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            zip.file(`images/${{fname}}`, bytes);
            csvRows.push(`${{fname}},${{videoBase}},${{expr}},${{pose}},${{chem}},operational`);
        }}
        zip.file('labels.csv', csvRows.join('\\n') + '\\n');

        // Export video metadata
        if (videoMeta) {{
            zip.file('video_meta.json', JSON.stringify(videoMeta, null, 2) + '\\n');
        }}

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
    chemistries = {{}};
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY + '_pose');
    localStorage.removeItem(STORAGE_KEY + '_chem');
    buildFilteredList();
    currentPos = 0;
    renderFocus();
    renderStrip();
}}

// Step-based options: solo vs duo
const STEP_OPTIONS_SOLO = {{
    '-1': [['__shoot__','setLabel']],
    '0': [['__shoot__','setLabel'], ['cut','setLabel']],
    '2': [['cheese','setLabel'], ['chill','setLabel'], ['edge','setLabel'], ['hype','setLabel']],
    '3': [['front','setPose'], ['angle','setPose'], ['side','setPose']],
}};
const STEP_OPTIONS_DUO = {{
    '-1': [['__shoot__','setLabel']],
    '0': [['__shoot__','setLabel'], ['cut','setLabel']],
    '1': [['sync','setChemistry'], ['interact','setChemistry']],
    '2': [['cheese','setLabel'], ['chill','setLabel'], ['edge','setLabel'], ['hype','setLabel']],
    '3': [['front','setPose'], ['angle','setPose'], ['side','setPose']],
}};

function getStep(idx) {{
    const lbl = labels[idx];
    const c = chemistries[idx];
    const p = poses[idx];
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const isAccepted = lbl && lbl !== 'cut' && lbl !== '__shoot__';
    if (!lbl) return 0;                          // shoot/cut
    if (lbl === 'cut') return -1;                // cut done
    if (isDuo && !c) return 1;                   // chemistry (duo only)
    if (lbl === '__shoot__' || !isAccepted) return 2;  // expression
    if (!p) return 3;                            // pose
    return 4;                                    // complete
}}

function isModalOpen() {{
    return document.getElementById('metaModal').style.display !== 'none';
}}

document.addEventListener('keydown', e => {{
    // Handle modal keyboard input
    if (isModalOpen()) {{
        const n = parseInt(e.key);
        const fields = getModalFields();
        if (e.key === 'Enter') {{
            const isDuo = metaDraft.scene === 'duo';
            const baseOk = metaDraft.scene && metaDraft.main_gender && metaDraft.main_ethnicity;
            const passengerOk = !isDuo || (metaDraft.passenger_gender && metaDraft.passenger_ethnicity);
            if (baseOk && passengerOk) {{
                saveMetaAndStart();
            }}
            return;
        }}
        if (e.key === 'Escape') {{
            if (videoMeta) hideMetaModal();
            return;
        }}
        if (e.key === 'Tab') {{
            e.preventDefault();
            modalField = (modalField + 1) % fields.length;
            renderMetaSelections();
            return;
        }}
        if (n >= 1) {{
            const field = fields[modalField];
            const values = MODAL_VALUES[field];
            if (n <= values.length) {{
                selectMetaOpt(field, values[n - 1]);
                // Re-get fields (scene change may add passenger fields)
                const updatedFields = getModalFields();
                // Auto-advance to next unfilled field
                for (let i = 0; i < updatedFields.length; i++) {{
                    const nextField = (modalField + 1 + i) % updatedFields.length;
                    if (!metaDraft[updatedFields[nextField]]) {{
                        modalField = nextField;
                        renderMetaSelections();
                        return;
                    }}
                }}
                // All filled, stay
                renderMetaSelections();
            }}
        }}
        return;
    }}

    const f = filteredList[currentPos];
    if (!f) return;
    const idx = f.index;
    const autoStep = getStep(idx);
    const step = (manualStep !== null) ? manualStep : autoStep;
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const STEP_OPTIONS = isDuo ? STEP_OPTIONS_DUO : STEP_OPTIONS_SOLO;

    // h/l = prev/next frame
    if (e.key === 'h') {{ go(-1); return; }}
    if (e.key === 'l') {{ go(1); return; }}

    // j/k = step focus down/up
    const isDuo2 = videoMeta && videoMeta.scene === 'duo';
    const maxStep = isDuo2 ? 3 : 3;
    const minStep = -1;
    if (e.key === 'j') {{
        if (manualStep === null) manualStep = step;
        manualStep = Math.min(maxStep, manualStep + 1);
        renderFocus();
        return;
    }}
    if (e.key === 'k') {{
        if (manualStep === null) manualStep = step;
        manualStep = Math.max(minStep, manualStep - 1);
        renderFocus();
        return;
    }}

    // q/w/e/r = 현재 포커스 step에서 1/2/3/4번 선택
    const QWER = {{'q': 1, 'w': 2, 'e': 3, 'r': 4}};
    const n = QWER[e.key];
    const activeStep = String(step);  // STEP_OPTIONS keys are strings
    if (n && STEP_OPTIONS[activeStep]) {{
        const opts = STEP_OPTIONS[activeStep];
        if (n <= opts.length) {{
            const [value, fn] = opts[n - 1];
            if (fn === 'setLabel') setLabel(idx, value);
            else if (fn === 'setChemistry') setChemistry(idx, value);
            else if (fn === 'setPose') setPose(idx, value);
            manualStep = null;  // 선택 후 자동 step으로
        }}
    }}
}});

const TL_COLORS = {{
    '': '#333',
    '__shoot__': '#666',
    'cut': '#d32f2f',
    'cheese': '#4CAF50',
    'chill': '#2196F3',
    'edge': '#FF5722',
    'hype': '#9C27B0',
}};

function renderTimeline() {{
    const bar = document.getElementById('timelineBar');
    bar.innerHTML = '';
    const currentFrame = filteredList[currentPos];
    const currentIdx = currentFrame ? currentFrame.index : -1;
    for (const f of FRAMES) {{
        const el = document.createElement('div');
        el.className = 'tl-frame';
        const lbl = labels[f.index] || '';
        el.style.background = TL_COLORS[lbl] || TL_COLORS[''];
        if (f.index === currentIdx) {{
            el.style.outline = '1px solid #fff';
            el.style.zIndex = '1';
            el.style.opacity = '1';
        }}
        el.onclick = () => {{
            // Find this frame in filteredList
            const pos = filteredList.findIndex(ff => ff.index === f.index);
            if (pos >= 0) {{
                currentPos = pos;
                manualStep = null;
                renderFocus();
                renderStrip();
            }}
        }};
        bar.appendChild(el);
    }}
}}

document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        buildFilteredList();
        currentPos = 0;
        renderFocus();
        renderStrip();
    }});
}});

// Init: always render frames first, then show modal overlay if needed
buildFilteredList();
renderFocus();
renderStrip();
updateMetaIndicator();
if (!videoMeta) {{
    showMetaModal();
}}
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

    import numpy as np

    logger.info("Extracting frames: %s (fps=%d)", video_path, fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    frames_data: list[tuple] = []
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0 and len(frames_data) < max_frames:
            frames_data.append(frame_bgr.copy())
        frame_idx += 1
    cap.release()

    logger.info("Extracted %d frames (from %d total, interval=%d)",
                len(frames_data), total_frames, frame_interval)

    if not frames_data:
        logger.error("No frames collected")
        raise RuntimeError("No frames collected from video")

    frames_info = []
    for idx, frame_bgr in enumerate(frames_data):
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
