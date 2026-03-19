"""Local review server — 라벨 수정/삭제가 파일에 즉시 반영.

annotator review data/datasets/portrait-v1 --serve
→ localhost:8765 에서 리뷰 UI
→ 라벨 수정 → labels.csv 즉시 저장
→ 이미지 삭제 → images/ + labels.csv에서 즉시 제거
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("annotator.serve")


class ReviewHandler(SimpleHTTPRequestHandler):
    """Handles API requests + serves static review HTML."""

    dataset_dir: Path
    labels_path: Path
    videos_path: Path
    images_dir: Path

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_html()
        elif parsed.path == "/api/labels":
            self._send_json(self._read_labels())
        elif parsed.path == "/api/videos":
            self._send_json(self._read_videos())
        elif parsed.path.startswith("/api/image/"):
            fname = parsed.path[len("/api/image/"):]
            self._serve_image(fname)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_body()

        if parsed.path == "/api/update_label":
            data = json.loads(body)
            self._update_label(data)
            self._send_json({"ok": True})

        elif parsed.path == "/api/update_video":
            data = json.loads(body)
            self._update_video(data)
            self._send_json({"ok": True})

        elif parsed.path == "/api/delete":
            data = json.loads(body)
            self._delete_image(data["filename"])
            self._send_json({"ok": True})

        elif parsed.path == "/api/delete_batch":
            data = json.loads(body)
            count = 0
            for fname in data.get("filenames", []):
                self._delete_image(fname)
                count += 1
            self._send_json({"ok": True, "deleted": count})

        else:
            self.send_error(404)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, fname: str):
        img_path = self.images_dir / fname
        if not img_path.exists():
            self.send_error(404)
            return
        data = img_path.read_bytes()
        ct = "image/jpeg" if fname.endswith(".jpg") else "image/png" if fname.endswith(".png") else "image/avif"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _read_labels(self) -> list[dict]:
        if not self.labels_path.exists():
            return []
        with open(self.labels_path, newline="") as f:
            return list(csv.DictReader(f))

    def _read_videos(self) -> dict[str, dict]:
        if not self.videos_path.exists():
            return {}
        with open(self.videos_path, newline="") as f:
            return {r["video_id"]: r for r in csv.DictReader(f)}

    def _write_labels(self, rows: list[dict]):
        fieldnames = ["filename", "video_id", "expression", "pose", "chemistry", "source"]
        with open(self.labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    def _write_videos(self, videos: dict[str, dict]):
        fieldnames = ["video_id", "scene", "main_gender", "main_ethnicity",
                       "passenger_gender", "passenger_ethnicity", "member_id", "notes"]
        with open(self.videos_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in videos.values():
                writer.writerow({k: v.get(k, "") for k in fieldnames})

    def _update_label(self, data: dict):
        rows = self._read_labels()
        fname = data["filename"]
        found = False
        for r in rows:
            if r["filename"] == fname:
                for k, v in data.items():
                    if k != "filename":
                        r[k] = v
                found = True
                break
        if not found:
            rows.append(data)
        self._write_labels(rows)
        logger.info("Label updated: %s → %s", fname, {k: v for k, v in data.items() if k != "filename"})

    def _update_video(self, data: dict):
        videos = self._read_videos()
        vid = data["video_id"]
        if vid in videos:
            videos[vid].update(data)
        else:
            videos[vid] = data
        self._write_videos(videos)
        logger.info("Video updated: %s", vid)

    def _delete_image(self, fname: str):
        # Remove from labels.csv
        rows = self._read_labels()
        rows = [r for r in rows if r["filename"] != fname]
        self._write_labels(rows)
        # Remove image file
        img_path = self.images_dir / fname
        if img_path.exists():
            img_path.unlink()
            logger.info("Deleted: %s", fname)
        else:
            logger.warning("File not found for deletion: %s", fname)

    def _serve_html(self):
        html = _build_review_html()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress noisy request logs
        pass


_REVIEW_HTML = None


def _build_review_html() -> str:
    global _REVIEW_HTML
    if _REVIEW_HTML:
        return _REVIEW_HTML

    _REVIEW_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dataset Review</title>
<style>
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }
h1 { color: #e94560; }
h2 { margin-top: 30px; }
.toolbar { background: #0f0f23; padding: 12px 20px; border-radius: 0;
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.toolbar button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
.filter-btn { background: #333; color: #ccc; padding: 4px 10px; border: 1px solid #555;
    border-radius: 3px; cursor: pointer; font-size: 12px; }
.filter-btn.active { background: #e94560; color: #fff; border-color: #e94560; }
.summary { background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.card { background: #16213e; border-radius: 6px; padding: 6px; text-align: center; }
.card img { width: 100%; border-radius: 4px; }
.card .name { font-size: 10px; color: #666; margin-top: 3px; }
.tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 1px; }
.section { margin: 20px 0; }
.edit-btns { display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 4px; }
.edit-btn { padding: 2px 6px; border: 1px solid #444; border-radius: 3px; background: #222;
    color: #aaa; cursor: pointer; font-size: 10px; }
.edit-btn:hover { background: #444; color: #fff; }
.edit-btn.active { color: #fff; font-weight: bold; }
.status { font-size: 12px; color: #4CAF50; margin-left: 16px; }
.desc { font-size: 11px; color: #888; margin-top: 2px; }
</style>
</head><body>
<h1>Dataset Review <span class="status" id="status"></span></h1>

<div class="toolbar">
    <div style="display:flex;gap:6px">
        <button class="filter-btn active" onclick="setView('expression',this)">Expression</button>
        <button class="filter-btn" onclick="setView('pose',this)">Pose</button>
        <button class="filter-btn" onclick="setView('chemistry',this)">Chemistry</button>
        <button class="filter-btn" onclick="setView('cut',this)">Cut</button>
        <button class="filter-btn" onclick="setView('occluded',this)">Occluded</button>
        <button class="filter-btn" onclick="setView('all',this)">All</button>
    </div>
    <div style="display:flex;gap:6px;margin-left:auto">
        <button class="filter-btn" id="selectBtn" onclick="toggleSelectMode()" style="background:#FF9800;color:#fff;display:none">Select Mode</button>
        <button class="filter-btn" id="deleteSelBtn" onclick="deleteSelected()" style="background:#d32f2f;color:#fff;display:none">Delete Selected (<span id="selCount">0</span>)</button>
        <button class="filter-btn" id="cancelSelBtn" onclick="cancelSelect()" style="display:none">Cancel</button>
    </div>
</div>

<div id="videoMeta"></div>
<div class="summary" id="summary"></div>
<div id="content"></div>

<script>
const COLORS = {
    cheese:'#4CAF50', goofy:'#E91E63', chill:'#2196F3', edge:'#FF5722', hype:'#9C27B0',
    cut:'#d32f2f', occluded:'#795548', front:'#00BCD4', angle:'#FF9800', side:'#795548',
    sync:'#FFD700', interact:'#00E676', solo:'#607D8B', duo:'#E91E63',
};
const DESC = {
    cheese:'얼굴이 주인공 — 프로필 사진', goofy:'장난스러운 표정 — 혀 내밀기, 윙크', chill:'쿨하고 여유로운', edge:'날카롭고 강렬한',
    hype:'순간이 주인공 — 에너지 폭발', occluded:'얼굴 가려짐', cut:'촬영 불가',
    front:'정면', angle:'3/4', side:'측면', sync:'동시 반응', interact:'교감',
};
const EXPRESSIONS = ['cheese','goofy','chill','edge','hype','occluded'];
const POSES = ['front','angle','side'];
const CHEMS = ['sync','interact'];

let ROWS = [];
let VIDEOS = {};
let currentView = 'expression';
let selectMode = false;
let selected = new Set();

function getColor(c) { return COLORS[c] || '#666'; }
function status(msg) { document.getElementById('status').textContent = msg; setTimeout(() => document.getElementById('status').textContent = '', 2000); }

async function loadData() {
    ROWS = await (await fetch('/api/labels')).json();
    VIDEOS = await (await fetch('/api/videos')).json();
    // Scan for new images
    renderSummary();
    renderVideoMeta();
    renderAll();
}

async function updateLabel(idx, field, value) {
    ROWS[idx][field] = value;
    await fetch('/api/update_label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(ROWS[idx]),
    });
    status('Saved ✓');
    renderAll();
}

async function updateVideo(videoId, field, value) {
    VIDEOS[videoId][field] = value;
    await fetch('/api/update_video', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(VIDEOS[videoId]),
    });
    status('Video saved ✓');
    renderVideoMeta();
}

async function deleteImage(idx) {
    const fname = ROWS[idx].filename;
    if (!confirm('Delete ' + fname + '?')) return;
    await fetch('/api/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: fname}),
    });
    ROWS.splice(idx, 1);
    status('Deleted ✓');
    renderSummary();
    renderAll();
}

function toggleSelectMode() {
    selectMode = !selectMode;
    selected.clear();
    updateSelectUI();
    renderAll();
}

function cancelSelect() {
    selectMode = false;
    selected.clear();
    updateSelectUI();
    renderAll();
}

function toggleSelect(idx) {
    if (selected.has(idx)) selected.delete(idx);
    else selected.add(idx);
    updateSelectUI();
    renderAll();
}

function updateSelectUI() {
    document.getElementById('selectBtn').style.display = selectMode ? 'none' : '';
    document.getElementById('deleteSelBtn').style.display = selectMode ? '' : 'none';
    document.getElementById('cancelSelBtn').style.display = selectMode ? '' : 'none';
    document.getElementById('selCount').textContent = selected.size;
}

async function deleteSelected() {
    if (selected.size === 0) return;
    if (!confirm('Delete ' + selected.size + ' images?')) return;
    const filenames = [...selected].map(i => ROWS[i].filename);
    await fetch('/api/delete_batch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filenames}),
    });
    // Remove from ROWS in reverse order
    const indices = [...selected].sort((a,b) => b-a);
    indices.forEach(i => ROWS.splice(i, 1));
    selected.clear();
    selectMode = false;
    updateSelectUI();
    status('Deleted ' + filenames.length + ' ✓');
    renderSummary();
    renderAll();
}

// Show select button always
document.getElementById('selectBtn').style.display = '';

function setView(view, btn) {
    currentView = view;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderAll();
}

function renderSummary() {
    const el = document.getElementById('summary');
    const ec = {}, pc = {};
    ROWS.forEach(r => { if (r.expression) ec[r.expression] = (ec[r.expression]||0)+1; if (r.pose) pc[r.pose] = (pc[r.pose]||0)+1; });
    el.innerHTML = `<b>Total:</b> ${ROWS.length} | <b>Expression:</b> ${Object.entries(ec).map(([k,v])=>k+'='+v).join(', ')} | <b>Pose:</b> ${Object.entries(pc).map(([k,v])=>k+'='+v).join(', ')} | <b>Videos:</b> ${Object.keys(VIDEOS).length}`;
}

function renderVideoMeta() {
    const el = document.getElementById('videoMeta');
    const vids = Object.values(VIDEOS);
    if (!vids.length) { el.innerHTML = ''; return; }
    const fields = ['scene','main_gender','main_ethnicity','passenger_gender','passenger_ethnicity','member_id'];
    const opts = { scene:['solo','duo'], main_gender:['male','female'], main_ethnicity:['asian','western','other'],
        passenger_gender:['male','female'], passenger_ethnicity:['asian','western','other'] };
    let html = '<div class="summary"><b>Videos</b><table style="margin-top:8px;border-collapse:collapse;font-size:12px;width:100%">';
    html += '<tr><th style="padding:4px 8px;text-align:left;color:#888">video_id</th>';
    fields.forEach(f => html += `<th style="padding:4px 8px;text-align:left;color:#888">${f}</th>`);
    html += '<th style="padding:4px 8px;color:#888">notes</th></tr>';
    for (const v of vids) {
        html += `<tr><td style="padding:4px 8px;color:#e94560">${v.video_id}</td>`;
        for (const f of fields) {
            if (opts[f]) {
                html += '<td style="padding:4px 8px">';
                opts[f].forEach(o => {
                    const sel = v[f]===o;
                    html += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(o)+';color:#fff':''}" onclick="updateVideo('${v.video_id}','${f}','${o}')">${o}</button> `;
                });
                html += '</td>';
            } else {
                html += `<td style="padding:4px 8px"><input type="text" value="${v[f]||''}" style="background:#222;border:1px solid #444;color:#eee;padding:2px 6px;border-radius:3px;width:80px;font-size:11px" onchange="updateVideo('${v.video_id}','${f}',this.value)"></td>`;
            }
        }
        html += `<td style="padding:4px 8px"><input type="text" value="${v.notes||''}" style="background:#222;border:1px solid #444;color:#eee;padding:2px 6px;border-radius:3px;width:120px;font-size:11px" onchange="updateVideo('${v.video_id}','notes',this.value)"></td></tr>`;
    }
    html += '</table></div>';
    el.innerHTML = html;
}

function renderCard(idx) {
    const r = ROWS[idx];
    const vid = VIDEOS[r.video_id] || {};
    let tags = '';
    if (r.expression) tags += `<span class="tag" style="background:${getColor(r.expression)}">${r.expression}</span>`;
    if (r.pose) tags += `<span class="tag" style="background:${getColor(r.pose)}">${r.pose}</span>`;
    if (r.chemistry) tags += `<span class="tag" style="background:${getColor(r.chemistry)}">${r.chemistry}</span>`;
    if (vid.scene) tags += `<span class="tag" style="background:${getColor(vid.scene)}">${vid.scene}</span>`;
    if (vid.main_gender) tags += `<span class="tag" style="background:#444">${vid.main_gender}</span>`;
    if (vid.main_ethnicity) tags += `<span class="tag" style="background:#444">${vid.main_ethnicity}</span>`;
    if (vid.member_id) tags += `<span class="tag" style="background:#333;color:#FF9800">${vid.member_id}</span>`;

    let btns = '<div class="edit-btns">';
    EXPRESSIONS.forEach(e => {
        const sel = r.expression===e;
        btns += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(e)+';color:#fff':''}" onclick="updateLabel(${idx},'expression','${e}')">${e}</button>`;
    });
    btns += `<button class="edit-btn${r.expression==='cut'?' active':''}" style="${r.expression==='cut'?'background:#d32f2f;color:#fff':''}" onclick="updateLabel(${idx},'expression','cut')">cut</button>`;
    btns += '</div><div class="edit-btns">';
    POSES.forEach(p => {
        const sel = r.pose===p;
        btns += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(p)+';color:#fff':''}" onclick="updateLabel(${idx},'pose','${p}')">${p}</button>`;
    });
    if (vid.scene==='duo') {
        btns += '&nbsp;';
        CHEMS.forEach(c => {
            const sel = r.chemistry===c;
            btns += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(c)+';color:#fff':''}" onclick="updateLabel(${idx},'chemistry','${c}')">${c}</button>`;
        });
    }
    btns += `&nbsp;<button class="edit-btn" style="background:#d32f2f;color:#fff" onclick="deleteImage(${idx})">✕</button>`;
    btns += '</div>';

    if (selectMode) {
        const isSel = selected.has(idx);
        const selStyle = isSel ? 'box-shadow:0 0 0 3px #e94560;opacity:1' : 'opacity:0.7';
        return `<div class="card" style="${selStyle};cursor:pointer" onclick="toggleSelect(${idx})"><img src="/api/image/${r.filename}" loading="lazy"><div class="name">${r.filename}</div><div class="tags">${tags}</div></div>`;
    }
    return `<div class="card"><img src="/api/image/${r.filename}" loading="lazy"><div class="name">${r.filename}</div><div class="tags">${tags}</div>${btns}</div>`;
}

function renderGroup(title, color, indices) {
    if (!indices.length) return '';
    let html = `<div class="section"><h2 style="color:${color}">${title} (${indices.length})</h2>`;
    if (DESC[title]) html += `<div class="desc">${DESC[title]}</div>`;
    html += '<div class="grid">';
    indices.forEach(i => html += renderCard(i));
    html += '</div></div>';
    return html;
}

function renderAll() {
    const el = document.getElementById('content');
    let html = '';

    if (currentView === 'expression' || currentView === 'all') {
        const groups = {};
        ROWS.forEach((r,i) => { const e = r.expression||'(none)'; (groups[e]=groups[e]||[]).push(i); });
        [...EXPRESSIONS, 'cut', '(none)'].forEach(e => { if (groups[e]) html += renderGroup(e, getColor(e), groups[e]); });
    }
    if (currentView === 'pose' || currentView === 'all') {
        const groups = {};
        ROWS.forEach((r,i) => { const p = r.pose||'(none)'; (groups[p]=groups[p]||[]).push(i); });
        [...POSES, '(none)'].forEach(p => { if (groups[p]) html += renderGroup('pose:'+p, getColor(p), groups[p]); });
    }
    if (currentView === 'chemistry') {
        const groups = {};
        ROWS.forEach((r,i) => { if (r.chemistry) (groups[r.chemistry]=groups[r.chemistry]||[]).push(i); });
        CHEMS.forEach(c => { if (groups[c]) html += renderGroup(c, getColor(c), groups[c]); });
    }
    if (currentView === 'cut') {
        const items = []; ROWS.forEach((r,i) => { if (r.expression==='cut') items.push(i); });
        html += renderGroup('cut', '#d32f2f', items);
    }
    if (currentView === 'occluded') {
        const items = []; ROWS.forEach((r,i) => { if (r.expression==='occluded') items.push(i); });
        html += renderGroup('occluded', '#795548', items);
    }

    el.innerHTML = html || '<p style="color:#888">No items</p>';
}

loadData();
</script>
</body></html>"""
    return _REVIEW_HTML


def start_server(dataset_dir: str | Path, port: int = 8765):
    """Start local review server."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"
    videos_path = dataset_dir / "videos.csv"

    # Scan for new images not in labels.csv
    if images_dir.exists() and labels_path.exists():
        with open(labels_path, newline="") as f:
            existing = {r["filename"] for r in csv.DictReader(f)}
        new_files = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".avif") and p.name not in existing:
                new_files.append(p.name)
        if new_files:
            fieldnames = ["filename", "video_id", "expression", "pose", "chemistry", "source"]
            with open(labels_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for fname in new_files:
                    writer.writerow({"filename": fname, "video_id": "", "expression": "",
                                     "pose": "", "chemistry": "", "source": "reference"})
            logger.info("Added %d new images to labels.csv", len(new_files))

    # Configure handler
    ReviewHandler.dataset_dir = dataset_dir
    ReviewHandler.labels_path = labels_path
    ReviewHandler.videos_path = videos_path
    ReviewHandler.images_dir = images_dir

    server = HTTPServer(("localhost", port), ReviewHandler)
    logger.info("Review server: http://localhost:%d", port)
    logger.info("Dataset: %s", dataset_dir)
    logger.info("Press Ctrl+C to stop")

    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()
