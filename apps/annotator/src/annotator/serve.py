"""Local review server — 라벨 수정/삭제가 파일에 즉시 반영.

annotator review data/datasets/portrait-v1 --serve
→ localhost:8765 에서 리뷰 UI
→ 라벨 수정 → labels.csv 즉시 저장
→ 이미지 삭제 → images/ + labels.csv에서 즉시 제거

images/ 하위 디렉토리 지원:
  labels.csv filename은 파일명만 저장 (경로 없음).
  서버가 재귀 스캔하여 이름→경로 매핑.
  동일 파일명이 여러 폴더에 있으면 경고.
"""

from __future__ import annotations

import csv
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger("annotator.serve")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".avif", ".webp"}


def _scan_images(images_dir: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Scan images/ recursively. Returns (name→path, duplicates: name→[rel_paths])."""
    name_to_path: dict[str, Path] = {}
    all_paths: dict[str, list[str]] = {}  # name → [relative paths]

    if not images_dir.exists():
        return name_to_path, {}

    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _IMG_EXTS:
            continue
        name = p.name
        rel = str(p.relative_to(images_dir))
        all_paths.setdefault(name, []).append(rel)
        if name not in name_to_path:
            name_to_path[name] = p

    duplicates = {k: v for k, v in all_paths.items() if len(v) > 1}
    return name_to_path, duplicates


class ReviewHandler(SimpleHTTPRequestHandler):
    """Handles API requests + serves static review HTML."""

    dataset_dir: Path
    labels_path: Path
    videos_path: Path
    images_dir: Path
    image_index: dict[str, Path]  # filename → full path
    duplicates: dict[str, list[str]]  # filename → [relative paths]

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_html()
        elif parsed.path == "/api/labels":
            self._send_json(self._read_labels())
        elif parsed.path == "/api/videos":
            self._send_json(self._read_videos())
        elif parsed.path == "/api/warnings":
            self._send_json(self._get_warnings())
        elif parsed.path == "/api/folders":
            self._send_json(self._get_folders())
        elif parsed.path.startswith("/api/image/"):
            fname = unquote(parsed.path[len("/api/image/"):])
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
        img_path = self.image_index.get(fname)
        # Fallback: rescan disk if not in index (e.g. newly merged images)
        if not img_path or not img_path.exists():
            self._refresh_index()
            img_path = self.image_index.get(fname)
        if not img_path or not img_path.exists():
            self.send_error(404)
            return
        data = img_path.read_bytes()
        suffix = img_path.suffix.lower()
        ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
              ".avif": "image/avif", ".webp": "image/webp"}.get(suffix, "image/jpeg")
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
            return {r["workflow_id"]: r for r in csv.DictReader(f)}

    def _write_labels(self, rows: list[dict]):
        fieldnames = ["filename", "workflow_id", "expression", "pose", "chemistry", "source"]
        with open(self.labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    def _write_videos(self, videos: dict[str, dict]):
        fieldnames = ["workflow_id", "scene", "main_gender", "main_ethnicity",
                       "passenger_gender", "passenger_ethnicity", "member_id",
                       "source_video", "total_frames", "labeled_count", "summary", "notes"]
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
        vid = data["workflow_id"]
        if vid in videos:
            videos[vid].update(data)
        else:
            videos[vid] = data
        self._write_videos(videos)
        logger.info("Video updated: %s", vid)

    def _refresh_index(self):
        """Rescan images directory and update index."""
        new_index, new_dupes = _scan_images(self.images_dir)
        self.__class__.image_index = new_index
        self.__class__.duplicates = new_dupes

    def _delete_image(self, fname: str):
        rows = self._read_labels()
        rows = [r for r in rows if r["filename"] != fname]
        self._write_labels(rows)
        img_path = self.image_index.pop(fname, None)
        if img_path and img_path.exists():
            img_path.unlink()
            logger.info("Deleted: %s", fname)
        else:
            logger.warning("File not found for deletion: %s", fname)

    def _get_warnings(self) -> list[str]:
        warnings = []
        for name, paths in self.duplicates.items():
            warnings.append(f"Duplicate filename '{name}' in: {', '.join(paths)}")
        return warnings

    def _get_folders(self) -> list[str]:
        """Return list of subfolder names under images/."""
        folders = set()
        for p in self.image_index.values():
            rel = p.relative_to(self.images_dir)
            if len(rel.parts) > 1:
                folders.add(str(rel.parent))
        return sorted(folders)

    def _serve_html(self):
        html = _build_review_html()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _build_review_html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dataset Review</title>
<style>
body { font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; margin: 20px; }
h1 { color: #e94560; }
h2 { margin-top: 30px; color: #444; }
.toolbar { background: #fff; padding: 12px 20px; border-radius: 0;
    position: sticky; top: 0; z-index: 100; border-bottom: 1px solid #ddd;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.toolbar button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
.filter-btn { background: #e8e8e8; color: #555; padding: 4px 10px; border: 1px solid #ccc;
    border-radius: 3px; cursor: pointer; font-size: 12px; }
.filter-btn.active { background: #e94560; color: #fff; border-color: #e94560; }
.summary { background: #fff; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 13px; border: 1px solid #e0e0e0; }
.warning-bar { background: #fff3e0; border: 1px solid #ffb74d; border-radius: 8px; padding: 10px 16px; margin: 10px 0; font-size: 12px; color: #e65100; }
.warning-bar b { color: #bf360c; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.card { background: #fff; border-radius: 6px; padding: 6px; text-align: center; border: 1px solid #e0e0e0; }
.card img { width: 100%; border-radius: 4px; }
.card .name { font-size: 10px; color: #999; margin-top: 3px; }
.tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 1px; color: #fff; }
.section { margin: 20px 0; }
.edit-btns { display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 4px; }
.edit-btn { padding: 2px 6px; border: 1px solid #ccc; border-radius: 3px; background: #f0f0f0;
    color: #666; cursor: pointer; font-size: 10px; }
.edit-btn:hover { background: #ddd; color: #333; }
.edit-btn.active { color: #fff; font-weight: bold; }
.status { font-size: 12px; color: #4CAF50; margin-left: 16px; }
.desc { font-size: 11px; color: #999; margin-top: 2px; }
.bucket-grid { display: grid; gap: 2px; }
.bucket-cell { display: flex; align-items: center; justify-content: center;
    height: 28px; border-radius: 4px; font-size: 12px; font-weight: 600;
    transition: transform 0.1s; position: relative; overflow: hidden; }
.bucket-cell:hover { transform: scale(1.06); z-index: 1; }
.bucket-cell.clickable { cursor: pointer; }
.bucket-cell.selected { box-shadow: inset 0 0 0 2px #e94560; }
.bucket-cell .bc-fill { position: absolute; inset: 0; border-radius: 4px; }
.bucket-cell .bc-num { position: relative; z-index: 1; }
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
    <div id="folderFilters" style="display:flex;gap:4px;border-left:1px solid #ddd;padding-left:12px"></div>
    <div style="display:flex;gap:6px;margin-left:auto">
        <button class="filter-btn" id="selectBtn" onclick="toggleSelectMode()" style="background:#FF9800;color:#fff">Select Mode</button>
        <button class="filter-btn" id="deleteSelBtn" onclick="deleteSelected()" style="background:#d32f2f;color:#fff;display:none">Delete Selected (<span id="selCount">0</span>)</button>
        <button class="filter-btn" id="cancelSelBtn" onclick="cancelSelect()" style="display:none">Cancel</button>
    </div>
</div>

<div id="warnings"></div>
<div id="videoMeta"></div>
<div class="summary" id="summary"></div>
<div id="bucketTable"></div>
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
let FOLDERS = [];
let currentView = 'expression';
let currentFolder = null; // null = all
let bucketFilter = null; // {expression, pose} or null
let selectMode = false;
let selected = new Set();

function getColor(c) { return COLORS[c] || '#666'; }
function status(msg) { document.getElementById('status').textContent = msg; setTimeout(() => document.getElementById('status').textContent = '', 2000); }

// Folder filter: match by workflow_id prefix convention (workflow_id often equals folder name)
// But since filename has no path, we use the /api/folders endpoint for folder buttons
// and filter by workflow_id grouping
function getFilteredIndices() {
    const indices = [];
    ROWS.forEach((r, i) => {
        // Folder filter
        if (currentFolder !== null) {
            const vid = r.workflow_id || '';
            const fname = r.filename || '';
            if (currentFolder === '') { if (vid) return; }
            else { if (vid !== currentFolder && !fname.startsWith(currentFolder + '_')) return; }
        }
        // Bucket filter
        if (bucketFilter) {
            const e = r.expression || '(none)';
            const p = r.pose || '';
            if (e !== bucketFilter.expression || p !== bucketFilter.pose) return;
        }
        indices.push(i);
    });
    return indices;
}

function selectBucket(expr, pose) {
    if (!expr || (bucketFilter && bucketFilter.expression === expr && bucketFilter.pose === pose)) {
        bucketFilter = null;
    } else {
        bucketFilter = { expression: expr, pose: pose };
    }
    renderBucketTable();
    renderSummary();
    renderAll();
    // Scroll to content
    if (bucketFilter) document.getElementById('content').scrollIntoView({behavior:'smooth'});
}

async function loadData() {
    ROWS = await (await fetch('/api/labels')).json();
    VIDEOS = await (await fetch('/api/videos')).json();
    FOLDERS = await (await fetch('/api/folders')).json();
    const warnings = await (await fetch('/api/warnings')).json();
    renderWarnings(warnings);
    renderFolderFilters();
    renderSummary();
    renderBucketTable();
    renderVideoMeta();
    renderAll();
}

function renderBucketTable() {
    const el = document.getElementById('bucketTable');
    // Bucket table counts should ignore bucketFilter itself (show full distribution)
    const savedBucket = bucketFilter;
    bucketFilter = null;
    const tableVisible = new Set(getFilteredIndices());
    bucketFilter = savedBucket;

    const allExpr = [...EXPRESSIONS, 'cut'];
    const allPose = [...POSES, ''];
    const poseLabel = p => p || '(none)';

    const counts = {};
    allExpr.forEach(e => { counts[e] = {}; allPose.forEach(p => counts[e][p] = 0); });
    counts['(none)'] = {}; allPose.forEach(p => counts['(none)'][p] = 0);

    ROWS.forEach((r,i) => {
        if (!tableVisible.has(i)) return;
        const e = r.expression || '(none)';
        const p = r.pose || '';
        if (counts[e]) counts[e][p] = (counts[e][p]||0) + 1;
    });

    const isSel = (e,p) => bucketFilter && bucketFilter.expression === e && bucketFilter.pose === p;
    let maxCount = 1;
    allExpr.forEach(e => allPose.forEach(p => { if (counts[e]?.[p] > maxCount) maxCount = counts[e][p]; }));

    const exprRows = [...allExpr, '(none)'].filter(e => e !== '(none)' || allPose.some(p => counts['(none)']?.[p] > 0));
    const poseRows = [...allPose];

    let html = '<div class="summary" style="overflow-x:auto">';
    if (bucketFilter) html += `<div style="margin-bottom:8px;font-size:12px;color:#e94560;cursor:pointer" onclick="selectBucket(null,null)">Showing: <b>${bucketFilter.expression}</b> × <b>${poseLabel(bucketFilter.pose)}</b> — click to clear</div>`;

    html += `<div class="bucket-grid" style="grid-template-columns:50px repeat(${exprRows.length}, 1fr) 36px;max-width:${60 + exprRows.length * 52 + 40}px">`;
    // Header: expression names
    html += '<div></div>';
    exprRows.forEach(e => html += `<div style="text-align:center;font-size:10px;font-weight:600;color:${getColor(e)||'#999'};padding:2px 0">${e}</div>`);
    html += '<div></div>';

    // Rows: one per pose
    for (const p of poseRows) {
        const colTotal = exprRows.reduce((s,e) => s + (counts[e]?.[p]||0), 0);
        html += `<div style="display:flex;align-items:center;justify-content:flex-end;padding-right:6px;font-size:11px;font-weight:600;color:${getColor(p)||'#999'}">${poseLabel(p)}</div>`;
        exprRows.forEach(e => {
            const v = counts[e]?.[p] || 0;
            const sel = isSel(e, p);
            const baseColor = getColor(e) || '#999';
            const t = v > 0 ? Math.min(v / maxCount, 1) : 0;
            const opacity = v > 0 ? 0.15 + 0.85 * t : 0;
            const click = v > 0 ? ` onclick="selectBucket('${e}','${p}')"` : '';
            const cls = 'bucket-cell' + (v > 0 ? ' clickable' : '') + (sel ? ' selected' : '');
            const textColor = sel ? '#fff' : opacity > 0.45 ? '#fff' : v > 0 ? '#444' : '#d0d0d0';
            const fillBg = sel ? '#e94560' : baseColor;
            const fillOpacity = sel ? 1 : opacity;
            html += `<div class="${cls}"${click}><div class="bc-fill" style="background:${fillBg};opacity:${fillOpacity}"></div><span class="bc-num" style="color:${textColor}">${v || '-'}</span></div>`;
        });
        html += `<div style="display:flex;align-items:center;justify-content:center;font-size:11px;color:#bbb">${colTotal}</div>`;
    }
    html += '</div></div>';
    el.innerHTML = html;
}

function renderWarnings(warnings) {
    const el = document.getElementById('warnings');
    if (!warnings.length) { el.innerHTML = ''; return; }
    el.innerHTML = '<div class="warning-bar"><b>Filename conflicts:</b> ' +
        warnings.map(w => '<br>' + w).join('') +
        '<br><span style="font-size:11px;color:#bf360c">Rename duplicates to avoid label mismatch.</span></div>';
}

function renderFolderFilters() {
    const el = document.getElementById('folderFilters');
    // Derive folders from workflow_ids
    const videoIds = new Set();
    ROWS.forEach(r => { if (r.workflow_id) videoIds.add(r.workflow_id); });
    // Also include filesystem folders
    FOLDERS.forEach(f => videoIds.add(f));
    if (videoIds.size === 0) { el.innerHTML = ''; return; }
    let html = `<button class="filter-btn${currentFolder===null?' active':''}" onclick="setFolder(null)" style="font-size:11px">All</button>`;
    [...videoIds].sort().forEach(f => {
        html += `<button class="filter-btn${currentFolder===f?' active':''}" onclick="setFolder('${f}')" style="font-size:11px">${f}</button>`;
    });
    el.innerHTML = html;
}

function setFolder(f) {
    currentFolder = f;
    renderFolderFilters();
    renderSummary();
    renderBucketTable();
    renderAll();
}

async function updateLabel(idx, field, value) {
    ROWS[idx][field] = value;
    await fetch('/api/update_label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(ROWS[idx]),
    });
    status('Saved');
    renderSummary();
    renderBucketTable();
    renderAll();
}

async function updateVideo(videoId, field, value) {
    VIDEOS[videoId][field] = value;
    await fetch('/api/update_video', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(VIDEOS[videoId]),
    });
    status('Video saved');
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
    status('Deleted');
    renderSummary();
    renderBucketTable();
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
    const indices = [...selected].sort((a,b) => b-a);
    indices.forEach(i => ROWS.splice(i, 1));
    selected.clear();
    selectMode = false;
    updateSelectUI();
    status('Deleted ' + filenames.length);
    renderSummary();
    renderBucketTable();
    renderAll();
}

function setView(view, btn) {
    currentView = view;
    document.querySelectorAll('.toolbar > div:first-child .filter-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderAll();
}

function renderSummary() {
    const el = document.getElementById('summary');
    const visible = new Set(getFilteredIndices());
    const ec = {}, pc = {};
    let count = 0;
    ROWS.forEach((r,i) => { if (!visible.has(i)) return; count++; if (r.expression) ec[r.expression] = (ec[r.expression]||0)+1; if (r.pose) pc[r.pose] = (pc[r.pose]||0)+1; });
    const folderLabel = currentFolder === null ? '' : ` (${currentFolder || 'ungrouped'})`;
    el.innerHTML = `<b>Total:</b> ${count}${folderLabel} | <b>Expression:</b> ${Object.entries(ec).map(([k,v])=>k+'='+v).join(', ')} | <b>Pose:</b> ${Object.entries(pc).map(([k,v])=>k+'='+v).join(', ')} | <b>Videos:</b> ${Object.keys(VIDEOS).length}`;
}

function renderVideoMeta() {
    const el = document.getElementById('videoMeta');
    const vids = Object.values(VIDEOS);
    if (!vids.length) { el.innerHTML = ''; return; }
    const fields = ['scene','main_gender','main_ethnicity','passenger_gender','passenger_ethnicity','member_id'];
    const infoFields = ['source_video','total_frames','labeled_count','summary'];
    const opts = { scene:['solo','duo'], main_gender:['male','female'], main_ethnicity:['asian','western','other'],
        passenger_gender:['male','female'], passenger_ethnicity:['asian','western','other'] };
    let html = '<div class="summary"><b>Videos</b><table style="margin-top:8px;border-collapse:collapse;font-size:12px;width:100%">';
    html += '<tr><th style="padding:4px 8px;text-align:left;color:#888">workflow_id</th>';
    fields.forEach(f => html += `<th style="padding:4px 8px;text-align:left;color:#888">${f}</th>`);
    infoFields.forEach(f => html += `<th style="padding:4px 8px;text-align:left;color:#aaa;font-size:10px">${f}</th>`);
    html += '<th style="padding:4px 8px;color:#888">notes</th></tr>';
    for (const v of vids) {
        html += `<tr><td style="padding:4px 8px;color:#e94560">${v.workflow_id}</td>`;
        for (const f of fields) {
            if (opts[f]) {
                html += '<td style="padding:4px 8px">';
                opts[f].forEach(o => {
                    const sel = v[f]===o;
                    html += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(o)+';color:#fff':''}" onclick="updateVideo('${v.workflow_id}','${f}','${o}')">${o}</button> `;
                });
                html += '</td>';
            } else {
                html += `<td style="padding:4px 8px"><input type="text" value="${v[f]||''}" style="background:#fff;border:1px solid #ccc;color:#333;padding:2px 6px;border-radius:3px;width:80px;font-size:11px" onchange="updateVideo('${v.workflow_id}','${f}',this.value)"></td>`;
            }
        }
        infoFields.forEach(f => html += `<td style="padding:4px 8px;font-size:10px;color:#999">${v[f]||''}</td>`);
        html += `<td style="padding:4px 8px"><input type="text" value="${v.notes||''}" style="background:#fff;border:1px solid #ccc;color:#333;padding:2px 6px;border-radius:3px;width:120px;font-size:11px" onchange="updateVideo('${v.workflow_id}','notes',this.value)"></td></tr>`;
    }
    html += '</table></div>';
    el.innerHTML = html;
}

let editingIdx = null;

function toggleEdit(idx) {
    editingIdx = (editingIdx === idx) ? null : idx;
    renderAll();
}

function renderCard(idx) {
    const r = ROWS[idx];
    const vid = VIDEOS[r.workflow_id] || {};
    const srcColor = r.source === 'operational' ? '#8D6E63' : '#78909C';
    const srcLabel = r.source === 'operational' ? 'OP' : 'REF';

    let tags = `<span class="tag" style="background:${srcColor};font-size:9px">${srcLabel}</span>`;
    if (r.expression) tags += `<span class="tag" style="background:${getColor(r.expression)}">${r.expression}</span>`;
    if (r.pose) tags += `<span class="tag" style="background:${getColor(r.pose)}">${r.pose}</span>`;
    if (r.chemistry) tags += `<span class="tag" style="background:${getColor(r.chemistry)}">${r.chemistry}</span>`;

    if (selectMode) {
        const isSel = selected.has(idx);
        const selStyle = isSel ? 'box-shadow:0 0 0 3px #e94560;opacity:1' : 'opacity:0.7';
        return `<div class="card" style="${selStyle};cursor:pointer" onclick="toggleSelect(${idx})"><img src="/api/image/${encodeURIComponent(r.filename)}" loading="lazy"><div class="name">${r.filename}</div><div class="tags">${tags}</div></div>`;
    }

    const isEditing = editingIdx === idx;
    let editPanel = '';
    if (isEditing) {
        editPanel += '<div class="edit-btns" style="margin-top:6px;padding:4px;background:#f0f0f0;border-radius:4px">';
        EXPRESSIONS.forEach(e => {
            const sel = r.expression===e;
            editPanel += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(e)+';color:#fff':''}" onclick="event.stopPropagation();updateLabel(${idx},'expression','${e}')">${e}</button>`;
        });
        editPanel += `<button class="edit-btn${r.expression==='cut'?' active':''}" style="${r.expression==='cut'?'background:#d32f2f;color:#fff':''}" onclick="event.stopPropagation();updateLabel(${idx},'expression','cut')">cut</button>`;
        editPanel += '</div><div class="edit-btns" style="padding:4px;background:#f0f0f0;border-radius:4px">';
        POSES.forEach(p => {
            const sel = r.pose===p;
            editPanel += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(p)+';color:#fff':''}" onclick="event.stopPropagation();updateLabel(${idx},'pose','${p}')">${p}</button>`;
        });
        if (vid.scene==='duo') {
            editPanel += '&nbsp;';
            CHEMS.forEach(c => {
                const sel = r.chemistry===c;
                editPanel += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(c)+';color:#fff':''}" onclick="event.stopPropagation();updateLabel(${idx},'chemistry','${c}')">${c}</button>`;
            });
        }
        editPanel += `&nbsp;<button class="edit-btn" style="background:#d32f2f;color:#fff" onclick="event.stopPropagation();deleteImage(${idx})">delete</button>`;
        editPanel += '</div>';
    }

    const isComplete = r.expression && r.expression !== 'cut' && r.pose;
    const isCut = r.expression === 'cut';
    const isEmpty = !r.expression && !r.pose;
    const bg = isEditing ? '#f3e8f9' : isComplete ? '#e8f5e9' : isCut ? '#fce4e4' : isEmpty ? '#fff' : '#fff8e1';
    return `<div class="card" style="background:${bg};cursor:pointer" onclick="toggleEdit(${idx})"><img src="/api/image/${encodeURIComponent(r.filename)}" loading="lazy"><div class="name">${r.filename}</div><div class="tags">${tags}</div>${editPanel}</div>`;
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
    const visible = new Set(getFilteredIndices());
    let html = '';

    // Bucket filter: flat grid (no grouping)
    if (bucketFilter) {
        const items = [...visible];
        if (items.length) {
            html += '<div class="grid">';
            items.forEach(i => html += renderCard(i));
            html += '</div>';
        } else {
            html += '<p style="color:#999">No images in this bucket</p>';
        }
        el.innerHTML = html;
        return;
    }

    if (currentView === 'expression' || currentView === 'all') {
        const groups = {};
        ROWS.forEach((r,i) => { if (!visible.has(i)) return; const e = r.expression||'(none)'; (groups[e]=groups[e]||[]).push(i); });
        [...EXPRESSIONS, 'cut', '(none)'].forEach(e => { if (groups[e]) html += renderGroup(e, getColor(e), groups[e]); });
    }
    if (currentView === 'pose' || currentView === 'all') {
        const groups = {};
        ROWS.forEach((r,i) => { if (!visible.has(i)) return; const p = r.pose||'(none)'; (groups[p]=groups[p]||[]).push(i); });
        [...POSES, '(none)'].forEach(p => { if (groups[p]) html += renderGroup('pose:'+p, getColor(p), groups[p]); });
    }
    if (currentView === 'chemistry') {
        const groups = {};
        ROWS.forEach((r,i) => { if (!visible.has(i)) return; if (r.chemistry) (groups[r.chemistry]=groups[r.chemistry]||[]).push(i); });
        CHEMS.forEach(c => { if (groups[c]) html += renderGroup(c, getColor(c), groups[c]); });
    }
    if (currentView === 'cut') {
        const items = []; ROWS.forEach((r,i) => { if (!visible.has(i)) return; if (r.expression==='cut') items.push(i); });
        html += renderGroup('cut', '#d32f2f', items);
    }
    if (currentView === 'occluded') {
        const items = []; ROWS.forEach((r,i) => { if (!visible.has(i)) return; if (r.expression==='occluded') items.push(i); });
        html += renderGroup('occluded', '#795548', items);
    }

    el.innerHTML = html || '<p style="color:#888">No items</p>';
}

loadData();
</script>
</body></html>"""


def start_server(dataset_dir: str | Path, port: int = 8765):
    """Start local review server."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"
    videos_path = dataset_dir / "videos.csv"

    # Build image index (name → path), detect duplicates
    image_index, duplicates = _scan_images(images_dir)
    if duplicates:
        for name, paths in duplicates.items():
            logger.warning("Duplicate filename '%s' found in: %s", name, ", ".join(paths))
        logger.warning(
            "%d filename conflict(s) detected. "
            "Only the first occurrence is used. Rename to avoid label mismatch.",
            len(duplicates),
        )
    logger.info("Indexed %d images", len(image_index))

    # Register new images not yet in labels.csv
    if labels_path.exists():
        with open(labels_path, newline="") as f:
            existing = {r["filename"] for r in csv.DictReader(f)}
        new_files = [name for name in sorted(image_index) if name not in existing]
        if new_files:
            fieldnames = ["filename", "workflow_id", "expression", "pose", "chemistry", "source"]
            with open(labels_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for fname in new_files:
                    writer.writerow({"filename": fname, "workflow_id": "", "expression": "",
                                     "pose": "", "chemistry": "", "source": "reference"})
            logger.info("Added %d new images to labels.csv", len(new_files))

    # Configure handler
    ReviewHandler.dataset_dir = dataset_dir
    ReviewHandler.labels_path = labels_path
    ReviewHandler.videos_path = videos_path
    ReviewHandler.images_dir = images_dir
    ReviewHandler.image_index = image_index
    ReviewHandler.duplicates = duplicates

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
