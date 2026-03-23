/* review.js — review-specific JS (depends on common.js) */

const CONFIG = window.__CONFIG__ || {};

document.addEventListener('DOMContentLoaded', () => {
    const el = document.getElementById('datasetDir');
    if (el && CONFIG.dataset_dir) el.textContent = CONFIG.dataset_dir;
});

let ROWS = [];
let VIDEOS = {};
let FOLDERS = [];
let SIGNALS = {};       // filename → {signal: value}
let PREDICTIONS = {};   // filename → {expression, pose, model, confidence}
let currentView = 'expression';
let currentFolder = null;
let bucketFilter = null;
let selectMode = false;
let selected = new Set();

// --- Tab navigation ---
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`.tab-btn[onclick="switchTab('${tab}')"]`).classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    if (tab === 'videos') renderVideoCards();
}

function viewVideoImages(wfId) {
    switchTab('images');
    setFolder(wfId);
}

// --- Filtering ---
function getFilteredIndices() {
    const indices = [];
    ROWS.forEach((r, i) => {
        if (currentFolder !== null) {
            const vid = r.workflow_id || '';
            const fname = r.filename || '';
            if (currentFolder === '') { if (vid) return; }
            else { if (vid !== currentFolder && !fname.startsWith(currentFolder + '_')) return; }
        }
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
    if (bucketFilter) document.getElementById('content').scrollIntoView({behavior:'smooth'});
}

// --- Data loading ---
async function loadData() {
    try {
        ROWS = await (await fetch('/api/labels')).json();
        VIDEOS = await (await fetch('/api/videos')).json();
        FOLDERS = await (await fetch('/api/folders')).json();
        SIGNALS = await (await fetch('/api/signals')).json();
        PREDICTIONS = await (await fetch('/api/predictions')).json();
        const warnings = await (await fetch('/api/warnings')).json();
        renderWarnings(warnings);
        renderFolderFilters();
        renderSummary();
        renderBucketTable();
        renderAll();
    } catch (e) {
        showStatus('Load failed: ' + e.message);
    }
}

// --- Bucket heatmap ---
function renderBucketTable() {
    const el = document.getElementById('bucketTable');
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

    let html = '<div class="summary" style="overflow-x:auto">';
    if (bucketFilter) html += `<div style="margin-bottom:8px;font-size:12px;color:#e94560;cursor:pointer" onclick="selectBucket(null,null)">Showing: <b>${bucketFilter.expression}</b> × <b>${poseLabel(bucketFilter.pose)}</b> — click to clear</div>`;

    html += `<div class="bucket-grid" style="grid-template-columns:50px repeat(${exprRows.length}, 1fr) 36px;max-width:${60 + exprRows.length * 52 + 40}px">`;
    html += '<div></div>';
    exprRows.forEach(e => html += `<div style="text-align:center;font-size:10px;font-weight:600;color:${getColor(e)||'#999'};padding:2px 0">${e}</div>`);
    html += '<div></div>';

    for (const p of allPose) {
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
    const videoIds = new Set();
    ROWS.forEach(r => { if (r.workflow_id) videoIds.add(r.workflow_id); });
    FOLDERS.forEach(f => videoIds.add(f));

    // Count per workflow
    const counts = {};
    ROWS.forEach(r => { const wf = r.workflow_id || ''; counts[wf] = (counts[wf]||0) + 1; });

    let html = `<option value="__all__"${currentFolder===null?' selected':''}>All (${ROWS.length})</option>`;
    [...videoIds].sort().forEach(f => {
        const n = counts[f] || 0;
        html += `<option value="${f}"${currentFolder===f?' selected':''}>${f} (${n})</option>`;
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

// --- CRUD operations ---
async function updateLabel(idx, field, value) {
    ROWS[idx][field] = value;
    try {
        await fetch('/api/update_label', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(ROWS[idx]),
        });
        showStatus('Saved');
    } catch (e) {
        showStatus('Save failed');
    }
    renderSummary();
    renderBucketTable();
    renderAll();
}

async function updateVideo(videoId, field, value) {
    VIDEOS[videoId][field] = value;
    if (field === 'scene' && value === 'solo') {
        VIDEOS[videoId].passenger_gender = '';
        VIDEOS[videoId].passenger_ethnicity = '';
    }
    try {
        await fetch('/api/update_video', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(VIDEOS[videoId]),
        });
        showStatus('Video saved');
    } catch (e) {
        showStatus('Save failed');
    }
    renderVideoCards();
}

async function deleteImage(idx) {
    const fname = ROWS[idx].filename;
    if (!confirm('Delete ' + fname + '?')) return;
    try {
        await fetch('/api/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: fname}),
        });
        ROWS.splice(idx, 1);
        showStatus('Deleted');
        // Close lightbox if deleting current image
        if (lbIdx === idx) closeLightbox();
        else if (lbIdx > idx) lbIdx--;
    } catch (e) {
        showStatus('Delete failed');
    }
    renderSummary();
    renderBucketTable();
    renderAll();
}

// --- Select mode ---
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
    try {
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
        showStatus('Deleted ' + filenames.length);
    } catch (e) {
        showStatus('Delete failed');
    }
    renderSummary();
    renderBucketTable();
    renderAll();
}

function setCardSize(size) {
    document.documentElement.style.setProperty('--card-size', size);
    document.querySelectorAll('.size-toggle button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
}

function setView(view, btn) {
    currentView = view;
    document.querySelectorAll('.toolbar > div:first-child .filter-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderAll();
}

// --- Summary ---
function renderSummary() {
    const el = document.getElementById('summary');
    const visible = new Set(getFilteredIndices());
    const ec = {}, pc = {};
    let count = 0;
    ROWS.forEach((r,i) => { if (!visible.has(i)) return; count++; if (r.expression) ec[r.expression] = (ec[r.expression]||0)+1; if (r.pose) pc[r.pose] = (pc[r.pose]||0)+1; });
    const folderLabel = currentFolder === null ? '' : ` (${currentFolder || 'ungrouped'})`;
    el.innerHTML = `<b>Total:</b> ${count}${folderLabel} | <b>Expression:</b> ${Object.entries(ec).map(([k,v])=>k+'='+v).join(', ')} | <b>Pose:</b> ${Object.entries(pc).map(([k,v])=>k+'='+v).join(', ')} | <b>Videos:</b> ${Object.keys(VIDEOS).length}`;
}

// --- Video cards ---
function renderVideoCards() {
    const el = document.getElementById('videoCards');
    const sumEl = document.getElementById('videoSummary');
    const vids = Object.values(VIDEOS);

    const imgCounts = {};
    ROWS.forEach(r => { const wf = r.workflow_id || ''; imgCounts[wf] = (imgCounts[wf]||0) + 1; });

    sumEl.innerHTML = `<b>${vids.length}</b> videos, <b>${ROWS.length}</b> total images`;

    if (!vids.length) { el.innerHTML = '<p style="color:#999">No videos registered</p>'; return; }

    const opts = { scene:['solo','duo'], main_gender:['male','female'], main_ethnicity:['asian','western','other'],
        passenger_gender:['male','female'], passenger_ethnicity:['asian','western','other'] };
    const editFields = ['scene','main_gender','main_ethnicity','passenger_gender','passenger_ethnicity','member_id'];

    let html = '';
    for (const v of vids) {
        const wf = v.workflow_id;
        const isDuo = v.scene === 'duo';
        const count = imgCounts[wf] || 0;
        const hasVideo = v.source_video && v.source_video.endsWith('.mp4');

        html += `<div class="video-card" id="vc-${wf}">`;
        if (hasVideo) {
            html += `<video width="240" height="135" controls preload="metadata" muted><source src="/api/video/${encodeURIComponent(v.source_video)}" type="video/mp4"></video>`;
        } else {
            html += '<div style="width:240px;height:135px;background:#e8e8e8;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:12px">No video</div>';
        }

        html += '<div class="vc-info">';
        html += `<div class="vc-title">${wf}</div>`;
        html += '<div class="vc-tags">';
        html += `<span class="tag" style="background:${getColor(v.scene)}">${v.scene||'?'}</span>`;
        html += `<span class="tag" style="background:#607D8B">${v.main_gender||'?'} / ${v.main_ethnicity||'?'}</span>`;
        if (isDuo) html += `<span class="tag" style="background:#E91E63">${v.passenger_gender||'?'} / ${v.passenger_ethnicity||'?'}</span>`;
        if (v.member_id) html += `<span class="tag" style="background:#78909C">${v.member_id}</span>`;
        html += '</div>';
        html += `<div class="vc-meta">${v.total_frames||'?'} frames · <b>${count}</b> images in dataset</div>`;
        if (v.summary) {
            html += `<div class="vc-summary">${v.summary}</div>`;
            // Distribution bar
            const parts = v.summary.split(' ').map(s => { const [k,n] = s.split('='); return {k, n:parseInt(n)||0}; });
            const total = parts.reduce((s,p) => s + p.n, 0);
            if (total > 0) {
                html += '<div class="vc-bar">';
                parts.forEach(p => {
                    const pct = (p.n / total * 100).toFixed(1);
                    html += `<div class="vc-bar-seg" style="width:${pct}%;background:${getColor(p.k)}" title="${p.k}: ${p.n}"></div>`;
                });
                html += '</div>';
            }
        }
        html += `<div style="margin-top:8px"><button class="filter-btn" onclick="viewVideoImages('${wf}')" style="font-size:11px">View ${count} images →</button></div>`;

        html += '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #eee">';
        for (const f of editFields) {
            const isPassenger = f.startsWith('passenger_');
            if (isPassenger && !isDuo) continue;
            if (opts[f]) {
                html += `<div style="display:inline-flex;gap:3px;margin:2px 4px 2px 0;align-items:center"><span style="font-size:10px;color:#aaa">${f.replace(/_/g,' ')}:</span>`;
                opts[f].forEach(o => {
                    const sel = v[f]===o;
                    html += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(o)+';color:#fff':''}" onclick="event.stopPropagation();updateVideo('${wf}','${f}','${o}')">${o}</button>`;
                });
                html += '</div>';
            } else {
                html += `<span style="font-size:10px;color:#aaa;margin-right:2px">${f}:</span><input type="text" value="${v[f]||''}" style="background:#fff;border:1px solid #ccc;color:#333;padding:1px 4px;border-radius:3px;width:60px;font-size:10px;margin-right:8px" onchange="updateVideo('${wf}','${f}',this.value)">`;
            }
        }
        html += `<span style="font-size:10px;color:#aaa;margin-right:2px">notes:</span><input type="text" value="${v.notes||''}" style="background:#fff;border:1px solid #ccc;color:#333;padding:1px 4px;border-radius:3px;width:120px;font-size:10px" onchange="updateVideo('${wf}','notes',this.value)">`;
        html += '</div>';
        html += '</div></div>';
    }
    el.innerHTML = html;
}

// --- Image cards ---
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
    if (r.moment === 'yes') tags += `<span class="tag" style="background:${getColor('moment')}">moment</span>`;

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
            const mSel = r.moment==='yes';
            editPanel += `<button class="edit-btn${mSel?' active':''}" style="${mSel?'background:'+getColor('moment')+';color:#fff':''}" onclick="event.stopPropagation();updateLabel(${idx},'moment',${mSel}?'':'yes')">moment</button>`;
        }
        editPanel += `&nbsp;<button class="edit-btn" style="background:#d32f2f;color:#fff" onclick="event.stopPropagation();deleteImage(${idx})">delete</button>`;
        editPanel += '</div>';
    }

    const isComplete = r.expression && r.expression !== 'cut' && r.pose;
    const isCut = r.expression === 'cut';
    const isEmpty = !r.expression && !r.pose;
    const bg = isEditing ? '#f3e8f9' : isComplete ? '#e8f5e9' : isCut ? '#fce4e4' : isEmpty ? '#fff' : '#fff8e1';
    return `<div class="card" style="background:${bg};cursor:pointer" onclick="toggleEdit(${idx})"><div class="zoom-hint">&#x1F50D;</div><img src="/api/image/${encodeURIComponent(r.filename)}" loading="lazy" onclick="event.stopPropagation();openLightbox(${idx})"><div class="name">${r.filename}</div><div class="tags">${tags}</div>${editPanel}</div>`;
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
    if (currentView === 'moment') {
        const items = []; ROWS.forEach((r,i) => { if (!visible.has(i)) return; if (r.moment==='yes') items.push(i); });
        html += renderGroup('moment', getColor('moment'), items);
    }
    if (currentView === 'cut') {
        const items = []; ROWS.forEach((r,i) => { if (!visible.has(i)) return; if (r.expression==='cut') items.push(i); });
        html += renderGroup('cut', '#d32f2f', items);
    }
    if (currentView === 'occluded') {
        const items = []; ROWS.forEach((r,i) => { if (!visible.has(i)) return; if (r.expression==='occluded') items.push(i); });
        html += renderGroup('occluded', '#795548', items);
    }
    if (currentView === 'mismatch') {
        const items = []; ROWS.forEach((r,i) => {
            if (!visible.has(i)) return;
            const pred = PREDICTIONS[r.filename];
            if (!pred || !r.expression) return;
            if (pred.expression !== r.expression || (r.pose && pred.pose && pred.pose !== r.pose)) items.push(i);
        });
        html += renderGroup('mismatch (manual ≠ model)', '#FF9800', items);
    }

    el.innerHTML = html || '<p style="color:#888">No items</p>';
}

// --- Lightbox ---
let lbIdx = null;
let lbList = [];

function openLightbox(idx) {
    lbIdx = idx;
    lbList = getFilteredIndices();
    renderLightbox();
    document.getElementById('lightbox').classList.add('open');
}

function closeLightbox() {
    document.getElementById('lightbox').classList.remove('open');
    lbIdx = null;
}

function lbNav(delta) {
    const pos = lbList.indexOf(lbIdx);
    if (pos < 0) return;
    const next = pos + delta;
    if (next >= 0 && next < lbList.length) {
        lbIdx = lbList[next];
        renderLightbox();
    }
}

function renderLightbox() {
    if (lbIdx === null || lbIdx >= ROWS.length) return;
    const r = ROWS[lbIdx];
    if (!r) return;
    const vid = VIDEOS[r.workflow_id] || {};
    document.getElementById('lbImg').src = '/api/image/' + encodeURIComponent(r.filename);

    // Source + label tags
    const srcColor = r.source === 'operational' ? '#8D6E63' : '#78909C';
    const srcLabel = r.source === 'operational' ? 'OP' : 'REF';
    let tags = `<span class="tag" style="background:${srcColor};font-size:9px">${srcLabel}</span>`;
    if (r.expression) tags += `<span class="tag" style="background:${getColor(r.expression)}">${r.expression}</span>`;
    if (r.pose) tags += `<span class="tag" style="background:${getColor(r.pose)}">${r.pose}</span>`;
    if (r.moment === 'yes') tags += `<span class="tag" style="background:${getColor('moment')}">moment</span>`;
    document.getElementById('lbTags').innerHTML = tags;

    // Edit: expression row
    let edit = '<div class="edit-btns" style="margin-top:4px">';
    EXPRESSIONS.forEach(e => {
        const sel = r.expression===e;
        edit += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(e)+';color:#fff':''}" onclick="lbUpdate('expression','${e}')">${e}</button>`;
    });
    edit += `<button class="edit-btn${r.expression==='cut'?' active':''}" style="${r.expression==='cut'?'background:#d32f2f;color:#fff':''}" onclick="lbUpdate('expression','cut')">cut</button>`;

    // Edit: pose + moment + delete row
    edit += '</div><div class="edit-btns" style="margin-top:2px">';
    POSES.forEach(p => {
        const sel = r.pose===p;
        edit += `<button class="edit-btn${sel?' active':''}" style="${sel?'background:'+getColor(p)+';color:#fff':''}" onclick="lbUpdate('pose','${p}')">${p}</button>`;
    });
    if (vid.scene==='duo') {
        edit += '&nbsp;';
        const mSel = r.moment==='yes';
        edit += `<button class="edit-btn${mSel?' active':''}" style="${mSel?'background:'+getColor('moment')+';color:#fff':''}" onclick="lbUpdate('moment',${mSel}?'':'yes')">moment</button>`;
    }
    edit += `&nbsp;<button class="edit-btn" style="background:#d32f2f;color:#fff" onclick="lbDelete()">delete</button>`;
    edit += '</div>';
    document.getElementById('lbEdit').innerHTML = edit;

    // Prediction comparison
    const pred = PREDICTIONS[r.filename];
    let predHtml = '';
    if (pred) {
        const exprMatch = !r.expression || r.expression === pred.expression;
        const poseMatch = !r.pose || r.pose === pred.pose;
        const exprColor = exprMatch ? '#4CAF50' : '#FF9800';
        const poseColor = poseMatch ? '#4CAF50' : '#FF9800';
        predHtml = `<div style="margin-top:6px;font-size:11px;padding:4px 8px;background:#f5f5f5;border-radius:4px">`;
        predHtml += `<span style="color:#888">model(${pred.model}):</span> `;
        predHtml += `<span style="color:${exprColor}">${pred.expression}</span>`;
        predHtml += ` <span style="color:${poseColor}">${pred.pose}</span>`;
        predHtml += ` <span style="color:#aaa">(${(parseFloat(pred.confidence)*100).toFixed(0)}%)</span>`;
        if (!exprMatch || !poseMatch) predHtml += ` <span style="color:#FF9800">≠ manual</span>`;
        predHtml += `</div>`;
    }

    // Signal hints
    const sig = SIGNALS[r.filename];
    let sigHtml = '';
    if (sig) {
        const yaw = sig.head_yaw_dev || 0;
        const yawDeg = yaw * 90;  // denormalize to degrees
        const poseHint = yawDeg < 8 ? 'front' : yawDeg < 25 ? 'angle' : 'side';
        const poseMatch = r.pose === poseHint;
        sigHtml += '<div style="margin-top:6px;font-size:11px;color:#aaa;line-height:1.6">';
        sigHtml += `<span style="color:${poseMatch ? '#4CAF50' : '#FF9800'}">yaw=${yawDeg.toFixed(0)}° → ${poseHint}</span>`;
        if (sig.em_happy !== undefined) sigHtml += ` | happy=${(sig.em_happy*100).toFixed(0)}%`;
        if (sig.mouth_open_ratio !== undefined) sigHtml += ` | mouth=${(sig.mouth_open_ratio*100).toFixed(0)}%`;
        if (sig.eye_visible_ratio !== undefined) sigHtml += ` | eye=${(sig.eye_visible_ratio*100).toFixed(0)}%`;
        if (sig.glasses_ratio !== undefined && sig.glasses_ratio > 0.05) sigHtml += ` | <span style="color:#FF9800">glasses</span>`;
        if (sig.backlight_score !== undefined && sig.backlight_score > 0.3) sigHtml += ` | <span style="color:#d32f2f">backlight</span>`;
        if (sig.face_confidence !== undefined) sigHtml += ` | conf=${(sig.face_confidence*100).toFixed(0)}%`;
        sigHtml += '</div>';
    }
    document.getElementById('lbEdit').innerHTML = edit + predHtml + sigHtml;

    // Position
    const pos = lbList.indexOf(lbIdx);
    document.getElementById('lbName').textContent = `${r.filename}  (${pos+1}/${lbList.length})`;
    document.getElementById('lbHint').textContent = '← → or H L: navigate | Esc: close';
}

async function lbUpdate(field, value) {
    await updateLabel(lbIdx, field, value);
    renderLightbox();
}

async function lbDelete() {
    await deleteImage(lbIdx);
    // Navigate to next or close
    if (lbList.length === 0) { closeLightbox(); return; }
    lbList = getFilteredIndices();
    if (lbIdx >= ROWS.length) lbIdx = ROWS.length - 1;
    if (!lbList.includes(lbIdx)) {
        const pos = lbList.findIndex(i => i >= lbIdx);
        lbIdx = pos >= 0 ? lbList[pos] : lbList[lbList.length - 1];
    }
    if (lbIdx === null || lbIdx === undefined) { closeLightbox(); return; }
    renderLightbox();
}

// Keyboard
document.addEventListener('keydown', e => {
    if (!document.getElementById('lightbox').classList.contains('open')) return;
    if (e.key === 'Escape') { closeLightbox(); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || e.key === 'h') { lbNav(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight' || e.key === 'l') { lbNav(1); e.preventDefault(); }
});

loadData();
