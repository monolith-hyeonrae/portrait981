/* common.js — shared constants and utilities for review and label tools */

const COLORS = {
    cheese:'#4CAF50', goofy:'#E91E63', chill:'#2196F3', edge:'#FF5722', hype:'#9C27B0',
    cut:'#d32f2f', occluded:'#795548', front:'#00BCD4', angle:'#FF9800', side:'#795548',
    dramatic:'#FF6F00', natural:'#43A047', backlit:'#5C6BC0',
    moment:'#FFD700', solo:'#607D8B', duo:'#E91E63',
};

const DESC = {
    '__shoot__': 'SHOOT', 'cut': 'CUT',
    cheese:'얼굴이 주인공 — 프로필 사진', goofy:'장난스러운 표정 — 혀 내밀기, 윙크',
    chill:'쿨하고 여유로운', edge:'날카롭고 강렬한',
    hype:'순간이 주인공 — 에너지 폭발', occluded:'얼굴 가려짐',
    front:'정면', angle:'3/4', side:'측면',
    dramatic:'강한 방향광 — Rembrandt, 직사광', natural:'자연광 — 부드러운 빛, 균일', backlit:'역광 — 얼굴 어두움',
    moment:'두 사람 다 잘 나온 추억 사진 — 고객이 구매할 만한 장면',
};

const EXPRESSIONS = ['cheese','goofy','chill','edge','hype','occluded'];
const POSES = ['front','angle','side'];
const LIGHTINGS = ['dramatic','natural','backlit'];

function getColor(c) { return COLORS[c] || '#666'; }

function showStatus(msg) {
    const el = document.getElementById('status');
    if (el) { el.textContent = msg; setTimeout(() => el.textContent = '', 2000); }
}
