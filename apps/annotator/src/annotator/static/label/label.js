/* label tool — all JS logic */

let FRAME_COUNT = 0;
let VIDEO_NAME = "";
let VIDEO_STEM = "";
let DATASET_DIR = "";
let videoCacheBust = 0;

let labels = {};
let poses = {};
let lightings = {};
let moments = {};
let videoMeta = null;

let currentPos = 0;
let currentFilter = 'all';
let filteredList = [];
let manualStep = null;
let saving = false;

function frameUrl(idx) { return `/api/frame/${idx}?v=${videoCacheBust}`; }

// --- Server communication ---
async function loadStaging() {
    const data = await (await fetch('/api/staging')).json();
    labels = data.labels || {};
    poses = data.poses || {};
    lightings = data.lightings || {};
    moments = data.moments || {};
    videoMeta = data.video_meta || null;
    buildFilteredList();
    renderAll();
    updateMetaIndicator();
    loadVideoList();
}

async function loadVideoList() {
    const data = await (await fetch('/api/video_list')).json();
    const sel = document.getElementById('videoSelect');
    if (!data.videos.length) { sel.style.display = 'none'; return; }
    sel.style.display = '';
    sel.innerHTML = data.videos.map(v =>
        `<option value="${v.filename}" ${v.current ? 'selected' : ''}>${v.stem} (${v.size_mb}MB)</option>`
    ).join('');
}

async function switchVideo(filename) {
    if (!filename) return;
    const sel = document.getElementById('videoSelect');
    sel.disabled = true;
    showStatus('Loading...');
    try {
        const res = await (await fetch('/api/load_video', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ filename }),
        })).json();
        if (res.ok) {
            // Update UI state
            VIDEO_NAME = res.video_name;
            VIDEO_STEM = VIDEO_NAME.replace(/\.[^.]+$/, '');
            FRAME_COUNT = res.frame_count;
            videoCacheBust++;
            document.getElementById('videoNameLabel').textContent = VIDEO_NAME;
            document.title = 'Label Tool — ' + VIDEO_NAME;
            document.getElementById('total').textContent = res.frame_count;
            currentPos = 0;
            manualStep = null;
            // Reload staging (new video's labels)
            const data = await (await fetch('/api/staging')).json();
            labels = data.labels || {};
            poses = data.poses || {};
    lightings = data.lightings || {};
            moments = data.moments || {};
            videoMeta = data.video_meta || null;
            buildFilteredList();
            renderAll();
            updateMetaIndicator();
            loadVideoList();
            showStatus(res.video_name + ' loaded');
                } else {
            alert('Failed: ' + res.error);
        }
    } finally {
        sel.disabled = false;
    }
}

async function saveToServer(index) {
    if (saving) return;
    saving = true;
    try {
        await fetch('/api/save_label', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                index: index,
                expression: labels[index] || '',
                pose: poses[index] || '',
                lighting: lightings[index] || '',
                moment: moments[index] || '',
            }),
        });
    } finally { saving = false; }
}

async function saveMetaToServer() {
    await fetch('/api/save_meta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(videoMeta),
    });
}

// --- Frame list ---
function buildFilteredList() {
    filteredList = [];
    for (let i = 0; i < FRAME_COUNT; i++) {
        const lbl = labels[i];
        if (currentFilter === 'unlabeled' && lbl !== undefined) continue;
        if (currentFilter === 'labeled' && lbl === undefined) continue;
        filteredList.push(i);
    }
    filteredList.sort((a, b) => a - b);
}

// --- Rendering ---
function renderAll() {
    renderFocus();
    renderStrip();
    renderTimeline();
    updateCount();
}

function renderStrip() {
    const sb = document.getElementById('strip');
    sb.innerHTML = '';
    filteredList.forEach((idx, pos) => {
        const div = document.createElement('div');
        const lbl = labels[idx];
        const isLabeled = lbl !== undefined;
        const borderColor = isLabeled ? (getColor(lbl) || '#4CAF50') : 'transparent';
        div.className = 'thumb' + (pos === currentPos ? ' active' : '');
        div.style.borderBottomColor = borderColor;
        div.style.opacity = isLabeled ? '0.7' : (pos === currentPos ? '1' : '0.5');
        div.innerHTML = `<img src="${frameUrl(idx)}" loading="lazy">`;
        div.onclick = () => { currentPos = pos; renderAll(); };
        sb.appendChild(div);
        if (pos === currentPos) {
            requestAnimationFrame(() => {
                const left = div.offsetLeft - sb.clientWidth / 2 + div.offsetWidth / 2;
                sb.scrollLeft = left;
            });
        }
    });
}

function renderFocus() {
    const panel = document.getElementById('focus');
    if (!filteredList.length) { panel.innerHTML = '<p style="color:#888">No frames</p>'; return; }
    if (currentPos >= filteredList.length) currentPos = filteredList.length - 1;
    if (currentPos < 0) currentPos = 0;

    const idx = filteredList[currentPos];
    const label = labels[idx];
    const pose = poses[idx];
    const mom = moments[idx];
    const scene = videoMeta ? videoMeta.scene : null;
    const isDuo = scene === 'duo';
    const isAccepted = label && label !== 'cut' && label !== '__shoot__';
    const displayLabel = label === '__shoot__' ? 'SHOOT' : label;
    const lighting = lightings[idx];
    const parts = [displayLabel, pose, lighting, mom === 'yes' ? 'moment' : null].filter(x => x && x !== '__shoot__');
    const labelColor = label === 'cut' ? '#d32f2f' : label === '__shoot__' ? '#FF9800' : label === 'occluded' ? '#795548' : '#4CAF50';
    const labelHtml = parts.length > 0
        ? `<div class="focus-label" style="color:${labelColor}">${parts.join(' + ')}</div>`
        : `<div class="focus-label" style="color:#888">Unlabeled</div>`;

    const K = '<span style="font-size:10px;opacity:0.5;margin-left:4px">';
    const FOCUS = 'border:3px solid #e94560;';
    const descHtml = (val) => val && DESC[val] ? `<div style="font-size:11px;color:#888;margin-top:2px;text-align:center">${DESC[val]}</div>` : '';

    const autoStep = getStep(idx);
    const step = (manualStep !== null && manualStep >= -1) ? manualStep : autoStep;

    let btnsHtml = '';
    // Step 0: SHOOT/CUT
    const isShot = label && label !== 'cut';
    const isCut = label === 'cut';
    btnsHtml += `<div class="buttons" style="${step === 0 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}">`;
    btnsHtml += `<button class="cat-btn${isShot ? ' selected' : ''}" style="${isShot ? 'background:#4CAF50;color:#fff;' : ''}" onclick="setLabel(${idx},'__shoot__')">SHOOT ${K}Q</span></button>`;
    btnsHtml += `<button class="cat-btn${isCut ? ' selected' : ''}" style="${isCut ? 'background:#d32f2f;color:#fff;' : 'background:#f5f5f5;color:#999;'}" onclick="setLabel(${idx},'cut')">CUT ${K}W</span></button>`;
    btnsHtml += '</div>';

    // MOMENT toggle (duo only, non-blocking)
    if (isDuo) {
        const mYes = mom === 'yes';
        btnsHtml += `<div class="buttons" style="margin-top:6px">`;
        btnsHtml += `<button class="cat-btn${mYes ? ' selected' : ''}" style="${mYes ? 'background:'+getColor('moment')+';color:#fff;' : ''}" onclick="setMoment(${idx})">Moment ${K}M</span></button>`;
        btnsHtml += '</div>';
    }

    // Step 2: EXPRESSION
    btnsHtml += `<div class="buttons" style="margin-top:6px;${step === 2 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}">`;
    for (const [cat, key] of [['cheese','Q'],['goofy','W'],['chill','E'],['edge','R'],['hype','T'],['occluded','Y']]) {
        const sel = label === cat;
        btnsHtml += `<button class="cat-btn${sel ? ' selected' : ''}" style="${sel ? 'background:'+getColor(cat)+';color:#fff;' : ''}" onclick="setLabel(${idx},'${cat}')">${cat} ${K}${key}</span></button>`;
    }
    btnsHtml += '</div>';
    btnsHtml += descHtml(isAccepted ? label : null);

    // Step 3: POSE
    btnsHtml += `<div class="buttons" style="margin-top:6px;${step === 3 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}">`;
    for (const [p, key] of [['front','Q'],['angle','W'],['side','E']]) {
        const sel = pose === p;
        btnsHtml += `<button class="cat-btn${sel ? ' selected' : ''}" style="${sel ? 'background:'+getColor(p)+';color:#fff;' : ''}" onclick="setPose(${idx},'${p}')">${p} ${K}${key}</span></button>`;
    }
    btnsHtml += '</div>';
    btnsHtml += descHtml(pose);

    // Step 4: LIGHTING
    btnsHtml += `<div class="buttons" style="margin-top:6px;${step === 4 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}">`;
    for (const [lt, key] of [['dramatic','Q'],['natural','W'],['backlit','E']]) {
        const sel = lighting === lt;
        btnsHtml += `<button class="cat-btn${sel ? ' selected' : ''}" style="${sel ? 'background:'+getColor(lt)+';color:#fff;' : ''}" onclick="setLighting(${idx},'${lt}')">${lt} ${K}${key}</span></button>`;
    }
    btnsHtml += '</div>';
    btnsHtml += descHtml(lighting);

    if (step === 5) btnsHtml += '<div style="color:#4CAF50;font-size:13px;margin-top:8px;text-align:center">Complete</div>';

    const stepHint = isDuo ? 'shoot\u2192moment\u2192expression\u2192pose\u2192lighting' : 'shoot\u2192expression\u2192pose\u2192lighting';
    panel.innerHTML = `
        <img class="focus-img" src="${frameUrl(idx)}">
        <div class="focus-meta">Frame #${idx} &nbsp; ${currentPos + 1} / ${filteredList.length}</div>
        ${labelHtml}
        ${btnsHtml}
        <div id="signalHint" style="font-size:11px;color:#aaa;text-align:center;margin-top:4px;min-height:16px"></div>
        <div class="nav">
            <button onclick="go(-1)" ${currentPos <= 0 ? 'disabled' : ''}>Prev ${K}H</span></button>
            <div class="pos">${currentPos + 1} / ${filteredList.length}</div>
            <button onclick="go(1)" ${currentPos >= filteredList.length - 1 ? 'disabled' : ''}>Next ${K}L</span></button>
        </div>
        <div class="shortcut-hint">H prev L next | J step\u2193 K step\u2191 | Q W E R T Y = select | ${stepHint}</div>
    `;
    // Async signal hint
    loadSignalHint(idx);
}

async function loadSignalHint(idx) {
    try {
        const sig = await (await fetch(`/api/signal/${idx}`)).json();
        const el = document.getElementById('signalHint');
        if (!el || !sig || Object.keys(sig).length === 0) return;
        // Check if still viewing the same frame
        if (filteredList[currentPos] !== idx) return;

        const yaw = sig.head_yaw_dev || 0;
        const yawDeg = yaw * 90;
        const poseHint = yawDeg < 8 ? 'front' : yawDeg < 25 ? 'angle' : 'side';
        let parts = [];
        parts.push(`yaw=${yawDeg.toFixed(0)}°→<b>${poseHint}</b>`);
        if (sig.em_happy !== undefined) parts.push(`happy=${(sig.em_happy*100).toFixed(0)}%`);
        if (sig.mouth_open_ratio !== undefined) parts.push(`mouth=${(sig.mouth_open_ratio*100).toFixed(0)}%`);
        if (sig.eye_visible_ratio !== undefined) parts.push(`eye=${(sig.eye_visible_ratio*100).toFixed(0)}%`);
        if (sig.glasses_ratio > 0.05) parts.push('<span style="color:#FF9800">glasses</span>');
        if (sig.backlight_score > 0.3) parts.push('<span style="color:#d32f2f">backlight</span>');
        if (sig.face_confidence !== undefined) parts.push(`conf=${(sig.face_confidence*100).toFixed(0)}%`);
        el.innerHTML = parts.join(' | ');
    } catch(e) { /* silent */ }
}

function go(delta) {
    currentPos = Math.max(0, Math.min(filteredList.length - 1, currentPos + delta));
    manualStep = null;
    renderAll();
}

function getStep(idx) {
    const lbl = labels[idx];
    const m = moments[idx];
    const p = poses[idx];
    const lt = lightings[idx];
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const isAccepted = lbl && lbl !== 'cut' && lbl !== '__shoot__';
    if (!lbl) return 0;                          // shoot/cut
    if (lbl === 'cut') return -1;                // cut done
    if (lbl === '__shoot__' || !isAccepted) return 2;  // expression
    if (!p) return 3;                            // pose
    if (!lt) return 4;                           // lighting
    return 5;                                    // complete
}

function setLabel(idx, value) {
    if (labels[idx] === value) delete labels[idx];
    else labels[idx] = value;
    saveToServer(idx);
    if (value === 'cut') setTimeout(() => go(1), 300);
    renderAll();
}

function setPose(idx, value) {
    if (poses[idx] === value) delete poses[idx];
    else poses[idx] = value;
    saveToServer(idx);
    checkAutoAdvance(idx);
    renderAll();
}

function setLighting(idx, value) {
    if (lightings[idx] === value) delete lightings[idx];
    else lightings[idx] = value;
    saveToServer(idx);
    checkAutoAdvance(idx);
    renderAll();
}

function setMoment(idx) {
    if (moments[idx] === 'yes') delete moments[idx];
    else moments[idx] = 'yes';
    saveToServer(idx);
    renderAll();
}

function checkAutoAdvance(idx) {
    const lbl = labels[idx];
    const p = poses[idx];
    const lt = lightings[idx];
    const isComplete = lbl && lbl !== '__shoot__' && lbl !== 'cut' && p && lt;
    if (isComplete) setTimeout(() => go(1), 400);
}

function updateCount() {
    const total = Object.keys(labels).length;
    document.getElementById('count').textContent = total;

    const EXPRS = [...EXPRESSIONS, 'cut'];
    const ALL_POSES = [...POSES, ''];
    const poseLabel = p => p || '(none)';

    const counts = {};
    EXPRS.forEach(e => { counts[e] = {}; ALL_POSES.forEach(p => counts[e][p] = 0); });
    let maxCount = 1;

    for (const [idx, lbl] of Object.entries(labels)) {
        if (!lbl || lbl === '__shoot__') continue;
        const p = poses[idx] || '';
        if (counts[lbl]) counts[lbl][p] = (counts[lbl][p]||0) + 1;
    }
    EXPRS.forEach(e => ALL_POSES.forEach(p => { if (counts[e][p] > maxCount) maxCount = counts[e][p]; }));

    const unlabeled = FRAME_COUNT - Object.keys(labels).length;
    const bar = document.getElementById('bucketBar');

    let html = `<div class="bucket-grid" style="grid-template-columns:50px repeat(${EXPRS.length}, 1fr) 36px;max-width:${60 + EXPRS.length * 48 + 40}px">`;
    html += '<div></div>';
    EXPRS.forEach(e => html += `<div style="text-align:center;font-size:9px;font-weight:600;color:${getColor(e)}">${e}</div>`);
    html += '<div></div>';
    for (const p of ALL_POSES) {
        html += `<div style="display:flex;align-items:center;justify-content:flex-end;padding-right:4px;font-size:10px;font-weight:600;color:${getColor(p)||'#666'}">${poseLabel(p)}</div>`;
        EXPRS.forEach(e => {
            const v = counts[e][p] || 0;
            const t = v > 0 ? Math.min(v/maxCount,1) : 0;
            const opacity = v > 0 ? 0.15 + 0.85*t : 0;
            const textColor = opacity > 0.45 ? '#fff' : v > 0 ? '#444' : '#d0d0d0';
            html += `<div class="bucket-cell"><div class="bc-fill" style="background:${getColor(e)};opacity:${opacity}"></div><span class="bc-num" style="color:${textColor}">${v||'-'}</span></div>`;
        });
        const rowTotal = ALL_POSES.length > 0 ? EXPRS.reduce((s,e) => s+(counts[e][p]||0), 0) : 0;
        html += `<div style="display:flex;align-items:center;justify-content:center;font-size:10px;color:#555">${rowTotal}</div>`;
    }
    html += '</div>';
    if (unlabeled > 0) html += `<span style="font-size:11px;color:#e94560;margin-left:12px">${unlabeled} unlabeled</span>`;

    bar.innerHTML = html;
}

const TL_EXPR_COLORS = { '':'#e0e0e0', '__shoot__':'#bbb', cut:'#d32f2f', occluded:'#795548',
    cheese:'#4CAF50', goofy:'#E91E63', chill:'#2196F3', edge:'#FF5722', hype:'#9C27B0' };
const TL_POSE_COLORS = { front:'#00BCD4', angle:'#FF9800', side:'#795548' };

function renderTimeline() {
    const wrap = document.getElementById('timelineWrap');
    const currentIdx = filteredList[currentPos];
    const isDuo = videoMeta && videoMeta.scene === 'duo';

    function makeRow(cls, colorFn) {
        const row = document.createElement('div');
        row.className = 'timeline-row ' + cls;
        for (let i = 0; i < FRAME_COUNT; i++) {
            const el = document.createElement('div');
            el.className = 'tl-frame';
            el.style.background = colorFn(i);
            if (i === currentIdx) { el.style.outline = '1px solid #e94560'; el.style.zIndex = '1'; }
            el.onclick = () => {
                const pos = filteredList.indexOf(i);
                if (pos >= 0) { currentPos = pos; manualStep = null; renderAll(); }
            };
            row.appendChild(el);
        }
        return row;
    }

    wrap.innerHTML = '';
    wrap.appendChild(makeRow('tl-expression', i => TL_EXPR_COLORS[labels[i] || ''] || '#e0e0e0'));
    wrap.appendChild(makeRow('tl-pose', i => TL_POSE_COLORS[poses[i]] || '#eee'));
    if (isDuo) {
        wrap.appendChild(makeRow('tl-moment', i => moments[i] === 'yes' ? '#FFD700' : '#f0f0f0'));
    }
}

// --- Meta bar ---
function setMeta(field, value) {
    if (!videoMeta) videoMeta = { workflow_id: VIDEO_STEM };
    videoMeta[field] = value;
    if (field === 'scene' && value === 'solo') {
        delete videoMeta.passenger_gender;
        delete videoMeta.passenger_ethnicity;
    }
    saveMetaToServer();
    renderMetaBar();
}

function setMetaText(field, value) {
    if (!videoMeta) videoMeta = { workflow_id: VIDEO_STEM };
    videoMeta[field] = value || undefined;
    if (!value) delete videoMeta[field];
    saveMetaToServer();
}

function renderMetaBar() {
    const el = document.getElementById('metaBar');
    const m = videoMeta || {};
    const isDuo = m.scene === 'duo';
    const opts = {
        scene: ['solo','duo'], main_gender: ['male','female'],
        main_ethnicity: ['asian','western','other'],
        passenger_gender: ['male','female'], passenger_ethnicity: ['asian','western','other'],
    };
    const fields = isDuo
        ? ['scene','main_gender','main_ethnicity','passenger_gender','passenger_ethnicity']
        : ['scene','main_gender','main_ethnicity'];

    let html = '';
    for (const f of fields) {
        html += `<span style="color:#999;font-size:11px">${f.replace(/_/g,' ')}:</span>`;
        opts[f].forEach(o => {
            const sel = m[f] === o;
            const bg = sel ? 'background:' + getColor(o) + ';color:#fff;' : '';
            html += `<button class="edit-btn${sel?' active':''}" style="${bg}font-size:11px" onclick="setMeta('${f}','${o}')">${o}</button>`;
        });
        html += '&nbsp;';
    }
    html += `<span style="color:#999;font-size:11px">member_id:</span>`;
    html += `<input type="text" value="${m.member_id||''}" style="background:#fff;border:1px solid #ccc;color:#333;padding:1px 4px;border-radius:3px;width:70px;font-size:11px" onchange="setMetaText('member_id',this.value)">`;

    if (!m.scene) html += `<span style="color:#e94560;font-size:11px;margin-left:8px">\u2190 set scene to start</span>`;
    el.innerHTML = html;
}

function updateMetaIndicator() { renderMetaBar(); }

// --- Confirm merge ---
async function showConfirm() {
    document.getElementById('confirmOverlay').style.display = 'flex';
    // Reset buttons to default state
    document.querySelector('.confirm-btns').innerHTML = '<button style="background:#e0e0e0;color:#555;flex:1;padding:12px;border:none;border-radius:6px;font-size:15px;cursor:pointer" onclick="hideConfirm()">Back to Edit</button><button style="background:#4CAF50;color:#fff;flex:1;padding:12px;border:none;border-radius:6px;font-size:15px;cursor:pointer" id="confirmBtn" onclick="doMerge()">Confirm Merge</button>';
    const s = document.getElementById('confirmSummary');
    s.innerHTML = 'Loading...';
    const preview = await (await fetch('/api/preview')).json();
    if (preview.total === 0) {
        s.innerHTML = '<p style="color:#e94560">No completed labels to merge.</p>';
        document.getElementById('confirmBtn').disabled = true;
        return;
    }
    document.getElementById('confirmBtn').disabled = false;

    let html = `<p>Merge <b>${preview.total}</b> labeled frames into <code>${preview.dataset_dir}</code></p>`;
    html += '<table style="border-collapse:collapse;font-size:12px;margin:12px 0">';
    html += '<tr><th style="padding:4px 8px;text-align:left;color:#888">Bucket</th><th style="padding:4px 8px;color:#888">Count</th></tr>';
    const sorted = Object.entries(preview.buckets).sort((a,b) => b[1]-a[1]);
    for (const [key, count] of sorted) {
        const [expr, pose] = key.split('|');
        const color = getColor(expr);
        html += `<tr><td style="padding:4px 8px"><span style="color:${color}">${expr}</span> \u00d7 ${pose}</td><td style="padding:4px 8px;text-align:center">${count}</td></tr>`;
    }
    html += '</table>';
    if (preview.video_meta) {
        const m = preview.video_meta;
        html += `<p style="font-size:12px;color:#888">Video: ${m.workflow_id} (${m.scene}, ${m.main_gender}/${m.main_ethnicity})</p>`;
    }
    s.innerHTML = html;
}

function hideConfirm() { document.getElementById('confirmOverlay').style.display = 'none'; }

async function afterMerge() {
    hideConfirm();
    // Reload staging from server (synced from dataset)
    const data = await (await fetch('/api/staging')).json();
    labels = data.labels || {};
    poses = data.poses || {};
    lightings = data.lightings || {};
    moments = data.moments || {};
    videoMeta = data.video_meta || null;
    buildFilteredList();
    renderAll();
    renderMetaBar();
    showStatus('Ready for more labeling');
}

async function doMerge() {
    const btn = document.getElementById('confirmBtn');
    btn.textContent = 'Merging...';
    btn.disabled = true;
    try {
        const res = await (await fetch('/api/confirm_merge', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' })).json();
        if (res.ok) {
            const s = document.getElementById('confirmSummary');
            let detail = `${res.merged_images} images saved, ${res.new_labels} new labels added`;
            if (res.updated_labels > 0) detail += `, ${res.updated_labels} labels updated`;
            if (res.video_saved) detail += `<br>Video saved: <code>${res.video_path}</code>`;
            else detail += '<br><span style="color:#FF9800">Video not saved (ffmpeg not found)</span>';
            s.innerHTML = `<div style="color:#4CAF50;font-size:18px;margin:20px 0">Merge complete!</div><p>${detail}</p><p style="color:#888;font-size:12px">You can close this tab or continue labeling.</p>`;
            document.querySelector('.confirm-btns').innerHTML = '<button style="background:#e0e0e0;color:#555;flex:1;padding:12px;border:none;border-radius:6px;font-size:15px;cursor:pointer" onclick="afterMerge()">Close</button>';
        } else {
            alert('Merge failed: ' + (res.error || 'unknown error'));
        }
    } catch(e) {
        alert('Merge failed: ' + e.message);
    }
    btn.textContent = 'Confirm Merge';
    btn.disabled = false;
}

function resetLabels() {
    if (!confirm('Reset all labels?')) return;
    labels = {};
    poses = {};
    moments = {};
    fetch('/api/save_label', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({index:-1,expression:'__reset__'}) });
    buildFilteredList();
    currentPos = 0;
    renderAll();
}

// --- Keyboard ---
const STEP_OPTIONS = {
    '-1': [['__shoot__','setLabel']],
    '0': [['__shoot__','setLabel'], ['cut','setLabel']],
    '2': [['cheese','setLabel'], ['goofy','setLabel'], ['chill','setLabel'], ['edge','setLabel'], ['hype','setLabel'], ['occluded','setLabel']],
    '3': [['front','setPose'], ['angle','setPose'], ['side','setPose']],
    '4': [['dramatic','setLighting'], ['natural','setLighting'], ['backlit','setLighting']],
};

function isModalOpen() {
    return document.getElementById('confirmOverlay').style.display !== 'none';
}

document.addEventListener('keydown', e => {
    if (isModalOpen()) {
        if (e.key === 'Escape') hideConfirm();
        return;
    }

    const idx = filteredList[currentPos];
    if (idx === undefined) return;
    const autoStep = getStep(idx);
    const step = (manualStep !== null) ? manualStep : autoStep;
    if (e.key === 'h') { go(-1); return; }
    if (e.key === 'l') { go(1); return; }
    if (e.key === 'j') { if (manualStep === null) manualStep = step; manualStep = Math.min(4, manualStep + 1); renderAll(); return; }
    if (e.key === 'k') { if (manualStep === null) manualStep = step; manualStep = Math.max(-1, manualStep - 1); renderAll(); return; }
    if (e.key === 'm') { const isDuo = videoMeta && videoMeta.scene === 'duo'; if (isDuo) setMoment(idx); return; }

    const QWER = {'q':1,'w':2,'e':3,'r':4,'t':5,'y':6};
    const n = QWER[e.key];
    const activeStep = String(step);
    if (n && STEP_OPTIONS[activeStep]) {
        const opts = STEP_OPTIONS[activeStep];
        if (n <= opts.length) {
            const [value, fn] = opts[n - 1];
            if (fn === 'setLabel') setLabel(idx, value);
            else if (fn === 'setPose') setPose(idx, value);
            manualStep = null;
        }
    }
});

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        buildFilteredList();
        currentPos = 0;
        renderAll();
    });
});

// --- Init: load config then staging ---
async function init() {
    const cfg = window.__CONFIG__ || {};
    FRAME_COUNT = cfg.frame_count || 0;
    VIDEO_NAME = cfg.video_name || '';
    VIDEO_STEM = VIDEO_NAME.replace(/\.[^.]+$/, '');
    DATASET_DIR = cfg.dataset_dir || '';
    document.getElementById('total').textContent = FRAME_COUNT;
    document.getElementById('videoNameLabel').textContent = VIDEO_NAME;
    document.getElementById('datasetDirLabel').textContent = DATASET_DIR;
    document.title = 'Label Tool \u2014 ' + VIDEO_NAME;
    loadStaging();
}

init();
