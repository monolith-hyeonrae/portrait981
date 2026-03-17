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
body {{ font-family: -apple-system, sans-serif; margin: 0; background: #1a1a2e; color: #eee; }}
.toolbar {{ position: sticky; top: 0; z-index: 100; background: #0f0f23; padding: 12px 20px;
    border-bottom: 2px solid #e94560; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.toolbar h1 {{ margin: 0; font-size: 18px; color: #e94560; }}
.progress {{ font-size: 14px; color: #aaa; }}
.progress b {{ color: #4CAF50; font-size: 18px; }}
.toolbar button {{ padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.btn-export {{ background: #4CAF50; color: #fff; }}
.btn-export:hover {{ background: #388E3C; }}
.btn-reset {{ background: #666; color: #fff; }}
.filter-group {{ display: flex; gap: 6px; }}
.filter-btn {{ background: #333; color: #ccc; padding: 4px 10px; border: 1px solid #555;
    border-radius: 3px; cursor: pointer; font-size: 12px; }}
.filter-btn.active {{ background: #e94560; color: #fff; border-color: #e94560; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 10px; padding: 16px; }}
.card {{ background: #16213e; border-radius: 8px; padding: 10px; transition: opacity .2s; }}
.card.disagree {{ border-left: 3px solid #e94560; }}
.card.labeled {{ opacity: 0.7; }}
.card img {{ width: 100%; border-radius: 4px; }}
.preds {{ margin: 6px 0; }}
.pred {{ display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 11px; margin: 1px; }}
.meta {{ font-size: 11px; color: #888; margin-bottom: 6px; }}
.buttons {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.cat-btn {{ padding: 4px 10px; border: 1px solid #444; border-radius: 4px; background: #222;
    color: #ccc; cursor: pointer; font-size: 12px; transition: all .15s; }}
.cat-btn:hover {{ background: #444; color: #fff; }}
.cat-btn.selected {{ color: #fff; font-weight: bold; border-width: 2px; }}
.cat-btn.reject {{ background: #d32f2f; }}
.cat-btn.skip {{ background: #333; }}
.cat-btn.none {{ background: #333; }}
.current-label {{ font-size: 11px; color: #4CAF50; margin-top: 4px; }}
</style>
</head><body>

<div class="toolbar">
    <h1>Anchor Set Label Tool</h1>
    <div class="progress">
        <b id="count">0</b> / <span id="total">{len(frames_info)}</span> labeled
        &nbsp;| Video: {video_name}
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

<div class="grid" id="grid"></div>

<script>
const FRAMES = {frames_json};
const IMAGES = {{{img_entries}}};
const CATEGORIES = {cat_list_json};
const CAT_COLORS = {cat_colors_json};
const STORAGE_KEY = "label_tool_{video_name.replace('.', '_')}";

let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
let currentFilter = 'all';

function getColor(cat) {{ return CAT_COLORS[cat] || '#666'; }}

function renderCards() {{
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    // Sort: unlabeled first, then labeled
    const sorted = [...FRAMES].sort((a, b) => {{
        const la = labels[a.index] !== undefined ? 1 : 0;
        const lb = labels[b.index] !== undefined ? 1 : 0;
        if (la !== lb) return la - lb;
        // Disagree first within group
        if (a.is_disagree !== b.is_disagree) return a.is_disagree ? -1 : 1;
        return a.index - b.index;
    }});

    let shown = 0;
    for (const f of sorted) {{
        const isLabeled = labels[f.index] !== undefined;
        if (currentFilter === 'unlabeled' && isLabeled) continue;
        if (currentFilter === 'labeled' && !isLabeled) continue;
        if (currentFilter === 'disagree' && !f.is_disagree) continue;

        const card = document.createElement('div');
        card.className = 'card' + (f.is_disagree ? ' disagree' : '') + (isLabeled ? ' labeled' : '');
        card.id = 'card-' + f.index;

        const label = labels[f.index];
        const labelHtml = label ? `<div class="current-label">Label: ${{label}}</div>` : '';

        card.innerHTML = `
            <img src="data:image/jpeg;base64,${{IMAGES[f.index]}}" loading="lazy">
            <div class="meta">Frame #${{f.index}}${{f.is_disagree ? ' | DISAGREE' : ''}}</div>
            <div class="preds">
                <span class="pred" style="background:${{getColor(f.catalog)}}">Cat: ${{f.catalog}}</span>
                <span class="pred" style="background:${{getColor(f.lr)}}">LR: ${{f.lr}}</span>
                <span class="pred" style="background:${{getColor(f.xgb)}}">XGB: ${{f.xgb}}</span>
            </div>
            <div class="buttons" id="btns-${{f.index}}"></div>
            ${{labelHtml}}
        `;
        grid.appendChild(card);

        const btns = card.querySelector('.buttons');
        for (const cat of CATEGORIES) {{
            const btn = document.createElement('button');
            btn.className = 'cat-btn' + (label === cat ? ' selected' : '');
            btn.textContent = cat;
            if (label === cat) btn.style.background = getColor(cat);
            btn.onclick = () => setLabel(f.index, cat);
            btns.appendChild(btn);
        }}
        for (const special of ['reject', 'skip', 'none']) {{
            const btn = document.createElement('button');
            btn.className = 'cat-btn ' + special + (label === special ? ' selected' : '');
            btn.textContent = special;
            if (label === special) btn.style.background = special === 'reject' ? '#d32f2f' : '#888';
            btn.onclick = () => setLabel(f.index, special);
            btns.appendChild(btn);
        }}
        shown++;
    }}
    updateCount();
}}

function setLabel(index, label) {{
    if (labels[index] === label) {{
        delete labels[index];
    }} else {{
        labels[index] = label;
    }}
    localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    renderCards();
}}

function updateCount() {{
    const count = Object.keys(labels).length;
    document.getElementById('count').textContent = count;
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
    if (!confirm('Reset all labels? This cannot be undone.')) return;
    labels = {{}};
    localStorage.removeItem(STORAGE_KEY);
    renderCards();
}}

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        renderCards();
    }});
}});

renderCards();
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
