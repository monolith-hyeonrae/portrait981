"""Dataset review HTML with label editing — gallery view by category.

Reads labels.csv + videos.csv and generates an interactive gallery
grouped by expression/pose/chemistry, with inline editing.

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


def _img_to_b64(img_path: Path, max_width: int = 240) -> str:
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
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"
    videos_path = dataset_dir / "videos.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found in {dataset_dir}")

    # Load labels
    rows: list[dict] = []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("Loaded %d labels", len(rows))

    # Load video metadata
    videos: dict[str, dict] = {}
    if videos_path.exists():
        with open(videos_path, newline="") as f:
            for row in csv.DictReader(f):
                videos[row["workflow_id"]] = row

    # Scan for new images not in labels.csv
    labeled_files = {row["filename"] for row in rows}
    new_count = 0
    if images_dir.exists():
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".avif") and img_path.name not in labeled_files:
                rows.append({
                    "filename": img_path.name,
                    "workflow_id": "",
                    "expression": "",
                    "pose": "",
                    "chemistry": "",
                    "source": "reference",
                })
                new_count += 1
    if new_count > 0:
        logger.info("Found %d new images not in labels.csv", new_count)

    # Pre-encode images
    image_data: dict[str, str] = {}
    for row in rows:
        fname = row["filename"]
        if fname not in image_data:
            img_path = images_dir / fname
            if img_path.exists():
                image_data[fname] = _img_to_b64(img_path)

    rows_json = json.dumps(rows)
    image_data_json = json.dumps(image_data)
    videos_json = json.dumps(videos)

    # Counts for summary
    from collections import Counter
    expr_c = Counter(r.get("expression", "") for r in rows if r.get("expression"))
    pose_c = Counter(r.get("pose", "") for r in rows if r.get("pose"))
    chem_c = Counter(r.get("chemistry", "") for r in rows if r.get("chemistry"))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dataset Review</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
h1 {{ color: #e94560; }}
h2 {{ margin-top: 30px; cursor: pointer; }}
h2:hover {{ opacity: 0.8; }}
.toolbar {{ background: #0f0f23; padding: 12px 20px; border-radius: 8px; margin: 10px 0;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.toolbar button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.btn-save {{ background: #4CAF50; color: #fff; }}
.btn-save:disabled {{ opacity: 0.5; }}
.changes-count {{ font-size: 14px; color: #FF9800; }}
.filter-group {{ display: flex; gap: 6px; }}
.filter-btn {{ background: #333; color: #ccc; padding: 4px 10px; border: 1px solid #555;
    border-radius: 3px; cursor: pointer; font-size: 12px; }}
.filter-btn.active {{ background: #e94560; color: #fff; border-color: #e94560; }}
.summary {{ background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }}
.card {{ background: #16213e; border-radius: 6px; padding: 6px; text-align: center; transition: box-shadow .2s; }}
.card img {{ width: 100%; border-radius: 4px; cursor: pointer; }}
.card .name {{ font-size: 10px; color: #666; margin-top: 3px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }}
.card .tags {{ margin-top: 4px; }}
.tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 1px; }}
.section {{ margin: 20px 0; }}
.edit-btns {{ display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 4px; }}
.edit-btn {{ padding: 2px 6px; border: 1px solid #444; border-radius: 3px; background: #222;
    color: #aaa; cursor: pointer; font-size: 10px; }}
.edit-btn:hover {{ background: #444; color: #fff; }}
.edit-btn.active {{ color: #fff; font-weight: bold; }}
.modified {{ box-shadow: 0 0 0 2px #FF9800; }}
.desc {{ font-size: 11px; color: #888; margin-top: 2px; }}
</style>
</head><body>
<h1>Dataset Review</h1>

<div class="toolbar">
    <div class="filter-group">
        <button class="filter-btn active" onclick="setView('expression')">Expression</button>
        <button class="filter-btn" onclick="setView('pose')">Pose</button>
        <button class="filter-btn" onclick="setView('chemistry')">Chemistry</button>
        <button class="filter-btn" onclick="setView('cut')">Cut</button>
        <button class="filter-btn" onclick="setView('all')">All</button>
    </div>
    <button class="btn-save" id="saveBtn" onclick="downloadCSV()" disabled>Download Modified CSV</button>
    <span class="changes-count" id="changesCount"></span>
</div>

<div class="summary">
    <b>Total:</b> {len(rows)} |
    <b>Expression:</b> {', '.join(f'{k}={v}' for k, v in expr_c.most_common())} |
    <b>Pose:</b> {', '.join(f'{k}={v}' for k, v in pose_c.most_common())} |
    <b>Chemistry:</b> {', '.join(f'{k}={v}' for k, v in chem_c.most_common()) or 'none'} |
    <b>Videos:</b> {len(videos)}
    {f'| <b style="color:#FF9800">New:</b> {new_count}' if new_count > 0 else ''}
</div>

<div id="videoMeta" style="margin:10px 0"></div>
<div id="content"></div>

<script>
const ROWS = {rows_json};
const IMAGES = {image_data_json};
const VIDEOS = {videos_json};
const COLORS = {{
    cheese: '#4CAF50', goofy: '#E91E63', chill: '#2196F3', edge: '#FF5722', hype: '#9C27B0',
    cut: '#d32f2f', occluded: '#795548', front: '#00BCD4', angle: '#FF9800', side: '#795548',
    sync: '#FFD700', interact: '#00E676', solo: '#607D8B', duo: '#E91E63',
}};
const DESC = {{
    cheese: '얼굴이 주인공 — 프로필 사진, 인물 초상화용',
    goofy: '장난스러운 표정 — 혀 내밀기, 윙크, 과장된 표정',
    chill: '쿨하고 여유로운 — 편안한 표정',
    edge: '날카롭고 강렬한 — 제임스딘',
    hype: '순간이 주인공 — 에너지 폭발',
    cut: '촬영 가치 없음',
    occluded: '얼굴 가려짐 — 마스크/선글라스/목도리',
    front: '정면', angle: '3/4 앵글', side: '측면',
    sync: '함께 웃기, 동시 반응', interact: '서로 교감',
}};
const EXPRESSIONS = ['cheese', 'goofy', 'chill', 'edge', 'hype', 'occluded'];
const POSES = ['front', 'angle', 'side'];
const CHEMS = ['sync', 'interact'];
let changes = {{}};
let deleted = new Set();
let currentView = 'expression';

function getColor(c) {{ return COLORS[c] || '#666'; }}

function setField(idx, field, value) {{
    const old = ROWS[idx][field];
    if (old === value) return;
    ROWS[idx][field] = value;
    if (!changes[idx]) changes[idx] = {{}};
    changes[idx][field] = value;
    renderAll();
    updateSaveBtn();
}}

function updateSaveBtn() {{
    const n = Object.keys(changes).length + deleted.size;
    document.getElementById('saveBtn').disabled = n === 0;
    const parts = [];
    if (Object.keys(changes).length > 0) parts.push(Object.keys(changes).length + ' modified');
    if (deleted.size > 0) parts.push(deleted.size + ' deleted');
    document.getElementById('changesCount').textContent = parts.join(', ');
}}

function downloadCSV() {{
    // labels.csv (삭제된 항목 제외)
    const header = 'filename,workflow_id,expression,pose,chemistry,source';
    const lines = [header];
    const deletedFiles = [];
    ROWS.forEach((r, i) => {{
        if (deleted.has(i)) {{
            deletedFiles.push(r.filename);
            return;
        }}
        lines.push([r.filename, r.workflow_id||'', r.expression||'', r.pose||'', r.chemistry||'', r.source||''].join(','));
    }});

    if (deletedFiles.length > 0) {{
        alert('삭제된 이미지 ' + deletedFiles.length + '건:\\n' + deletedFiles.join('\\n') + '\\n\\nimages/ 폴더에서 직접 삭제해주세요.');
    }}
    const blob = new Blob([lines.join('\\n') + '\\n'], {{ type: 'text/csv' }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'labels.csv';
    a.click();

    // videos.csv
    const vids = Object.values(VIDEOS);
    if (vids.length > 0) {{
        const vHeader = 'workflow_id,scene,main_gender,main_ethnicity,passenger_gender,passenger_ethnicity,member_id,notes';
        const vLines = [vHeader];
        for (const v of vids) {{
            vLines.push([v.workflow_id||'', v.scene||'', v.main_gender||'', v.main_ethnicity||'',
                v.passenger_gender||'', v.passenger_ethnicity||'', v.member_id||'', v.notes||''].join(','));
        }}
        const vBlob = new Blob([vLines.join('\\n') + '\\n'], {{ type: 'text/csv' }});
        const va = document.createElement('a');
        va.href = URL.createObjectURL(vBlob);
        va.download = 'videos.csv';
        va.click();
    }}
}}

function toggleDelete(idx) {{
    const fname = ROWS[idx].filename;
    if (deleted.has(idx)) {{
        deleted.delete(idx);
    }} else {{
        deleted.add(idx);
    }}
    updateSaveBtn();
    renderAll();
}}

function editVideoField(videoId, field, value) {{
    if (VIDEOS[videoId]) {{
        VIDEOS[videoId][field] = value;
        if (!changes['__videos__']) changes['__videos__'] = {{}};
        changes['__videos__'][videoId] = true;
        updateSaveBtn();
        renderVideoMeta();
    }}
}}

function renderVideoMeta() {{
    const el = document.getElementById('videoMeta');
    const vids = Object.values(VIDEOS);
    if (vids.length === 0) {{ el.innerHTML = ''; return; }}

    const fields = ['scene','main_gender','main_ethnicity','passenger_gender','passenger_ethnicity','member_id'];
    const opts = {{
        scene: ['solo','duo'], main_gender: ['male','female'], main_ethnicity: ['asian','western','other'],
        passenger_gender: ['male','female'], passenger_ethnicity: ['asian','western','other'],
    }};

    let html = '<div class="summary"><b>Videos</b><table style="margin-top:8px;border-collapse:collapse;font-size:12px;width:100%">';
    html += '<tr><th style="padding:4px 8px;text-align:left;color:#888">workflow_id</th>';
    for (const f of fields) html += `<th style="padding:4px 8px;text-align:left;color:#888">${{f}}</th>`;
    html += '<th style="padding:4px 8px;color:#888">notes</th></tr>';

    for (const v of vids) {{
        const modified = changes['__videos__'] && changes['__videos__'][v.workflow_id] ? 'color:#FF9800;' : '';
        html += `<tr style="${{modified}}"><td style="padding:4px 8px;color:#e94560">${{v.workflow_id}}</td>`;
        for (const f of fields) {{
            if (opts[f]) {{
                html += '<td style="padding:4px 8px">';
                for (const o of opts[f]) {{
                    const sel = v[f] === o;
                    const bg = sel ? `background:${{getColor(o) || '#444'}};color:#fff;` : '';
                    html += `<button class="edit-btn${{sel?' active':''}}" style="${{bg}}font-size:10px" onclick="editVideoField('${{v.workflow_id}}','${{f}}','${{o}}')">${{o}}</button> `;
                }}
                html += '</td>';
            }} else {{
                // member_id: text input
                html += `<td style="padding:4px 8px"><input type="text" value="${{v[f]||''}}" style="background:#222;border:1px solid #444;color:#eee;padding:2px 6px;border-radius:3px;width:80px;font-size:11px" onchange="editVideoField('${{v.workflow_id}}','${{f}}',this.value)"></td>`;
            }}
        }}
        html += `<td style="padding:4px 8px"><input type="text" value="${{v.notes||''}}" style="background:#222;border:1px solid #444;color:#eee;padding:2px 6px;border-radius:3px;width:150px;font-size:11px" onchange="editVideoField('${{v.workflow_id}}','notes',this.value)"></td>`;
        html += '</tr>';
    }}
    html += '</table></div>';
    el.innerHTML = html;
}}

function setView(view) {{
    currentView = view;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    renderAll();
}}

function renderCard(idx, row) {{
    const b64 = IMAGES[row.filename];
    if (!b64) return '';
    const expr = row.expression || '';
    const pose = row.pose || '';
    const chem = row.chemistry || '';
    const vid = VIDEOS[row.workflow_id] || {{}};
    const isModified = changes[idx] ? ' modified' : '';
    const isDeleted = deleted.has(idx);

    if (isDeleted) {{
        return `<div class="card" id="card-${{idx}}" style="opacity:0.3;position:relative">
            <img src="data:image/jpeg;base64,${{b64}}">
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#d32f2f;font-size:24px;font-weight:bold">DELETED</div>
            <div class="name">${{row.filename}}</div>
            <button class="edit-btn" style="background:#4CAF50;color:#fff;margin-top:4px" onclick="toggleDelete(${{idx}})">Undo</button>
        </div>`;
    }}

    let tags = '';
    if (expr) tags += `<span class="tag" style="background:${{getColor(expr)}}">${{expr}}</span>`;
    if (pose) tags += `<span class="tag" style="background:${{getColor(pose)}}">${{pose}}</span>`;
    if (chem) tags += `<span class="tag" style="background:${{getColor(chem)}}">${{chem}}</span>`;
    if (vid.scene) tags += `<span class="tag" style="background:${{getColor(vid.scene)}}">${{vid.scene}}</span>`;
    if (vid.main_gender) tags += `<span class="tag" style="background:#444">${{vid.main_gender}}</span>`;
    if (vid.main_ethnicity) tags += `<span class="tag" style="background:#444">${{vid.main_ethnicity}}</span>`;
    if (vid.member_id) tags += `<span class="tag" style="background:#333;color:#FF9800">${{vid.member_id}}</span>`;
    const srcColor = row.source === 'operational' ? '#8D6E63' : '#78909C';
    const srcLabel = row.source === 'operational' ? 'OP' : 'REF';
    tags += `<span class="tag" style="background:${{srcColor}};font-size:9px">${{srcLabel}}</span>`;

    let editHtml = '<div class="edit-btns">';
    for (const e of EXPRESSIONS) {{
        const act = expr === e;
        editHtml += `<button class="edit-btn${{act?' active':''}}" style="${{act?'background:'+getColor(e):''}}" onclick="setField(${{idx}},'expression','${{e}}')">${{e}}</button>`;
    }}
    editHtml += `<button class="edit-btn${{expr==='cut'?' active':''}}" style="${{expr==='cut'?'background:#d32f2f':''}}" onclick="setField(${{idx}},'expression','cut')">cut</button>`;
    editHtml += '</div><div class="edit-btns">';
    for (const p of POSES) {{
        const act = pose === p;
        editHtml += `<button class="edit-btn${{act?' active':''}}" style="${{act?'background:'+getColor(p):''}}" onclick="setField(${{idx}},'pose','${{p}}')">${{p}}</button>`;
    }}
    if (vid.scene === 'duo') {{
        editHtml += '&nbsp;';
        for (const c of CHEMS) {{
            const act = chem === c;
            editHtml += `<button class="edit-btn${{act?' active':''}}" style="${{act?'background:'+getColor(c):''}}" onclick="setField(${{idx}},'chemistry','${{c}}')">${{c}}</button>`;
        }}
    }}
    editHtml += `&nbsp;<button class="edit-btn" style="background:#d32f2f;color:#fff" onclick="toggleDelete(${{idx}})">✕</button>`;
    editHtml += '</div>';

    return `<div class="card${{isModified}}" id="card-${{idx}}">
        <img src="data:image/jpeg;base64,${{b64}}">
        <div class="name">${{row.filename}}</div>
        <div class="tags">${{tags}}</div>
        ${{editHtml}}
    </div>`;
}}

function renderGroup(title, color, items) {{
    if (items.length === 0) return '';
    let html = `<div class="section"><h2 style="color:${{color}}">${{title}} (${{items.length}})</h2>`;
    if (DESC[title]) html += `<div class="desc">${{DESC[title]}}</div>`;
    html += '<div class="grid">';
    for (const {{row, idx}} of items) html += renderCard(idx, row);
    html += '</div></div>';
    return html;
}}

function renderAll() {{
    const content = document.getElementById('content');
    let html = '';

    if (currentView === 'expression' || currentView === 'all') {{
        const groups = {{}};
        ROWS.forEach((r, i) => {{
            const e = r.expression || '(none)';
            if (!groups[e]) groups[e] = [];
            groups[e].push({{row: r, idx: i}});
        }});
        for (const e of [...EXPRESSIONS, 'cut', '(none)']) {{
            if (groups[e]) html += renderGroup(e, getColor(e), groups[e]);
        }}
    }}

    if (currentView === 'pose' || currentView === 'all') {{
        const groups = {{}};
        ROWS.forEach((r, i) => {{
            const p = r.pose || '(none)';
            if (!groups[p]) groups[p] = [];
            groups[p].push({{row: r, idx: i}});
        }});
        for (const p of [...POSES, '(none)']) {{
            if (groups[p]) html += renderGroup('pose:' + p, getColor(p), groups[p]);
        }}
    }}

    if (currentView === 'chemistry') {{
        const groups = {{}};
        ROWS.forEach((r, i) => {{
            const c = r.chemistry || '';
            if (c) {{
                if (!groups[c]) groups[c] = [];
                groups[c].push({{row: r, idx: i}});
            }}
        }});
        for (const c of CHEMS) {{
            if (groups[c]) html += renderGroup(c, getColor(c), groups[c]);
        }}
    }}

    if (currentView === 'cut') {{
        const items = [];
        ROWS.forEach((r, i) => {{ if (r.expression === 'cut') items.push({{row: r, idx: i}}); }});
        html += renderGroup('cut', '#d32f2f', items);
    }}

    content.innerHTML = html || '<p style="color:#888">No items to show</p>';
}}

renderAll();
renderVideoMeta();
</script>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Review: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path
