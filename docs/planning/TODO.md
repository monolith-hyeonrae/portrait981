# TODO — 미결 작업 목록

## Scoring Formula 재검토 (우선순위 높음)

**문제**: `final_score = quality × impact` 곱 구조에서
- Impact Score가 0.8+ 인 순간에도 Final Score가 0.5를 넘지 않음
- 원인: quality 전형값 0.55~0.65 × impact 0.8 = 0.44~0.52

**세부 원인 분석**:
1. `blur_normed`, `face_size_normed`가 영상 내 min-max라 평균이 ~0.5로 수렴
2. 고감정 순간 (강한 미소, 머리 움직임) = 기술 품질과 역상관 경향
3. EMA smoothing이 gate_mask=0인 프레임(final=0)을 포함해 계산 → 피크 과도 감쇠

**검토할 개선 방향**:

### 옵션 A: 비대칭 지수 가중
```python
final = quality ** 0.4 * impact ** 0.6  # impact에 더 높은 가중
```

### 옵션 B: 가산 방식
```python
final = 0.35 * quality + 0.65 * impact
```

### 옵션 C: Gate-pass only smoothing
```python
# gate=0 프레임을 0으로 채우지 않고 보간 (혹은 gate 통과 프레임만 EMA)
smoothed = EMA(final_scores, alpha=cfg.smoothing_alpha, mask=gate_mask)
```

### 옵션 D: Quality gate를 softgate로 전환
- hard gate (mask=0/1) 대신 quality penalty (-20%~-50%)로 전환
- gate 실패 프레임이 EMA에 0으로 포함되는 문제 해소

**검증 기준**: report.html Score Pipeline에서 high-impact 순간이
smoothed score 0.5 이상으로 표시되어야 함.

---

## head_aesthetic ROI 튜닝 검토

**현황**: `aesthetic_expand=2.5` (어깨 포함 포트레이트 크롭)으로 변경

**확인 필요**:
- 실제 영상에서 head_aes 값 분산이 증가했는지 report.html로 확인
- 크롭이 너무 넓으면 배경 비중이 높아져 얼굴 미학 평가가 희석될 수 있음
- 필요 시 `y_shift` 파라미터 추가로 크롭 중심 하향 이동 (어깨 비중 추가)

**튜닝 파라미터 위치**: `ShotQualityAnalyzer(aesthetic_expand=2.5)` in
`libs/vpx/plugins/vision-embed/src/vpx/vision_embed/analyzer.py`

---

## 메모

- Scoring 공식 변경 시 `test_batch_highlight.py`의 score expectation 확인
- `highlight_rules.md §6` 함께 업데이트 필요
