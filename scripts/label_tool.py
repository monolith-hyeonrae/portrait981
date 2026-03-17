"""Anchor Set labeling tool — interactive HTML interface for ground-truth annotation.

Extends visual_compare.py into a browser-based labeling tool where users assign
correct categories to frames, with stratified sampling that prioritizes disagreement.

Usage:
    uv run python scripts/label_tool.py ~/Videos/reaction_test/test_0.mp4
    uv run python scripts/label_tool.py ~/Videos/reaction_test/test_0.mp4 --fps 2 --output labels.html --max-frames 500
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("label_tool")


def frame_to_base64(frame_bgr, max_width=280):
    """OpenCV frame -> base64 JPEG."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buf).decode("ascii")


def sample_frames(n_total, disagree_indices, catalog_confidences, max_frames):
    """Stratified sampling: 50% disagreement, 25% high-conf, 25% low-conf."""
    disagree_set = set(disagree_indices)
    agree_indices = [i for i in range(n_total) if i not in disagree_set]

    # Sort by confidence
    agree_by_conf = sorted(agree_indices, key=lambda i: catalog_confidences[i])
    mid = len(agree_by_conf) // 2
    low_conf = agree_by_conf[:mid]
    high_conf = agree_by_conf[mid:]

    budget_disagree = max_frames // 2
    budget_high = max_frames // 4
    budget_low = max_frames - budget_disagree - budget_high

    rng = np.random.default_rng(42)
    selected = set()

    def pick(pool, n):
        pool = list(pool)
        if len(pool) <= n:
            return pool
        return list(rng.choice(pool, size=n, replace=False))

    selected.update(pick(disagree_indices, budget_disagree))
    selected.update(pick(low_conf, budget_low))
    selected.update(pick(high_conf, budget_high))
    return sorted(selected)


def generate_html(frames_info, categories, video_name):
    """Build self-contained interactive HTML labeling tool."""
    # frames_info: list of dicts with keys: index, b64, catalog, lr, xgb, is_disagree
    frames_json = json.dumps([
        {k: v for k, v in f.items() if k != "b64"} for f in frames_info
    ])

    cat_colors = {
        "warm_smile": "#4CAF50", "cool_gaze": "#2196F3",
        "cool_expression": "#2196F3", "lateral": "#FF9800",
        "playful_face": "#E91E63", "wild_energy": "#9C27B0",
    }
    cat_list_json = json.dumps(categories)
    cat_colors_json = json.dumps(cat_colors)

    # Build image data separately (large)
    img_entries = ",".join(f'"{f["index"]}":"{f["b64"]}"' for f in frames_info)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
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
    <button class="btn-export" onclick="exportLabels()">Export JSON</button>
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

let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
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
    // unlabeled + disagree first
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
    const labelHtml = label
        ? `<div class="focus-label">Label: ${{label}}</div>`
        : `<div class="focus-label" style="color:#888">Unlabeled</div>`;

    let btnsHtml = '<div class="buttons">';
    for (const cat of CATEGORIES) {{
        const sel = label === cat;
        const bg = sel ? `background:${{getColor(cat)}};color:#fff;` : '';
        btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setLabel(${{f.index}},'${{cat}}')">${{cat}}</button>`;
    }}
    for (const sp of ['reject', 'skip', 'none']) {{
        const sel = label === sp;
        const bg = sel ? (sp === 'reject' ? 'background:#d32f2f;color:#fff;' : 'background:#888;color:#fff;') : '';
        btnsHtml += `<button class="cat-btn ${{sp}}${{sel ? ' selected' : ''}}" style="${{bg}}" onclick="setLabel(${{f.index}},'${{sp}}')">${{sp}}</button>`;
    }}
    btnsHtml += '</div>';

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
        <div class="shortcut-hint">Keyboard: &larr; &rarr; navigate &nbsp;|&nbsp; 1-${{CATEGORIES.length}} category &nbsp;|&nbsp; r reject &nbsp;|&nbsp; s skip</div>
    `;
    updateCount();
}}

function go(delta) {{
    currentPos = Math.max(0, Math.min(filteredList.length - 1, currentPos + delta));
    renderFocus();
    renderSidebar();
}}

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

function updateCount() {{
    document.getElementById('count').textContent = Object.keys(labels).length;
}}

function exportLabels() {{
    const frames = FRAMES.map(f => ({{
        index: f.index,
        label: labels[f.index] || null,
        catalog: f.catalog,
        lr: f.lr,
        xgb: f.xgb,
        is_disagree: f.is_disagree,
    }})).filter(f => f.label !== null);

    const data = {{
        video: "{video_name}",
        total_frames: FRAMES.length,
        labeled_count: frames.length,
        frames: frames,
    }};

    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'anchor_labels.json';
    a.click();
}}

function resetLabels() {{
    if (!confirm('Reset all labels?')) return;
    labels = {{}};
    localStorage.removeItem(STORAGE_KEY);
    buildFilteredList();
    currentPos = 0;
    renderFocus();
    renderSidebar();
}}

// Keyboard shortcuts
document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
    else if (e.key === 'r') {{ if (filteredList[currentPos]) setLabel(filteredList[currentPos].index, 'reject'); }}
    else if (e.key === 's') {{ if (filteredList[currentPos]) setLabel(filteredList[currentPos].index, 'skip'); }}
    else {{
        const n = parseInt(e.key);
        if (n >= 1 && n <= CATEGORIES.length && filteredList[currentPos]) {{
            setLabel(filteredList[currentPos].index, CATEGORIES[n - 1]);
        }}
    }}
}});

// Filter buttons
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


def main():
    parser = argparse.ArgumentParser(description="Anchor Set Label Tool")
    parser.add_argument("video", help="mp4 video path")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--catalog", default="data/catalogs/portrait-v1")
    parser.add_argument("--output", "-o", default="labels.html")
    parser.add_argument("--max-frames", type=int, default=500)
    args = parser.parse_args()

    from momentscan.algorithm.batch.extract import extract_frame_record
    from momentscan.algorithm.batch.catalog_scoring import SIGNAL_FIELDS, extract_signal_vector

    # 1. Run momentscan
    logger.info("Processing: %s (fps=%d)", args.video, args.fps)
    frames_data = []

    def on_frame(frame, results):
        record = extract_frame_record(frame, results)
        if record is not None:
            frame_bgr = frame.image if hasattr(frame, "image") else None
            if frame_bgr is None and hasattr(frame, "data"):
                frame_bgr = frame.data
            if frame_bgr is not None:
                vec = extract_signal_vector(record, signal_fields=SIGNAL_FIELDS)
                frames_data.append((frame_bgr.copy(), record, vec))
        return True

    import momentscan as ms
    ms.run(args.video, fps=args.fps, backend="simple", on_frame=on_frame)
    logger.info("Collected %d frames", len(frames_data))

    if not frames_data:
        logger.error("No frames collected")
        return

    # 2. Score with all 3 strategies
    vectors = np.array([fd[2] for fd in frames_data])

    from visualbind.profile import load_profiles
    from visualbind.strategies.catalog import CatalogStrategy
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    profiles = load_profiles(Path(args.catalog))
    cat_strat = CatalogStrategy(profiles=profiles)
    cat_names = [p.name for p in profiles]

    catalog_results = []
    for vec in vectors:
        scores = cat_strat.predict(vec)
        best = max(scores, key=scores.get)
        catalog_results.append((best, scores[best]))

    y_catalog = [r[0] for r in catalog_results]
    confidences = [r[1] for r in catalog_results]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_catalog)

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0,
                            use_label_encoder=False, eval_metric="mlogloss")
        xgb.fit(vectors, y_encoded)
        xgb_preds = le.inverse_transform(xgb.predict(vectors))
        logger.info("XGBoost trained")
    except ImportError:
        xgb_preds = y_catalog
        logger.warning("XGBoost not available")

    # LR
    lr = LogisticRegression(max_iter=1000)
    lr.fit(vectors, y_encoded)
    lr_preds = le.inverse_transform(lr.predict(vectors))
    logger.info("LogisticRegression trained")

    # 3. Find disagreements
    disagree_indices = [
        i for i in range(len(vectors))
        if y_catalog[i] != lr_preds[i] or y_catalog[i] != xgb_preds[i]
    ]
    logger.info("Disagreements: %d / %d (%.1f%%)",
                len(disagree_indices), len(vectors),
                100 * len(disagree_indices) / len(vectors))

    # 4. Stratified sampling
    selected = sample_frames(len(vectors), disagree_indices, confidences, args.max_frames)
    logger.info("Sampled %d frames (from %d total)", len(selected), len(vectors))

    # 5. Build frame info
    disagree_set = set(disagree_indices)
    frames_info = []
    for idx in selected:
        frames_info.append({
            "index": idx,
            "b64": frame_to_base64(frames_data[idx][0]),
            "catalog": str(y_catalog[idx]),
            "lr": str(lr_preds[idx]),
            "xgb": str(xgb_preds[idx]),
            "is_disagree": idx in disagree_set,
        })

    # 6. Generate HTML
    video_name = Path(args.video).name
    html = generate_html(frames_info, cat_names, video_name)

    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Label tool saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    logger.info("Open in browser to start labeling. Labels persist in localStorage.")


if __name__ == "__main__":
    main()
