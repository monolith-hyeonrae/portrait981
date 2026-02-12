# highlight_rules — 수치/통계 기반 하이라이트 엔진

> **momentscan Phase 1** 상세 설계. 마스터 플랜: [momentscan.md](momentscan.md)
>
> 임베딩 없이 수치 feature + 통계 정규화 + peak detection으로 하이라이트 구간을 잡는 시스템.
> 해석 가능한 베이스라인이자, highlight_vector(Phase 2)와 병렬 비교 대상.

## 1. 목적

- **임베딩 사용 금지** — 순수 수치/통계 접근
- 해석 가능성: 모든 highlight에 "왜?" 설명 첨부
- 낮은 compute 요구 — GPU 임베딩 모델 없이 동작
- highlight_vector의 비교 베이스라인

## 2. 입출력

### Input

- 라이드 비디오 (~2분), post-ride 배치 처리

### Output

```
output/{video_id}/highlight_rules/
├── windows.json          # 하이라이트 구간 목록
├── timeseries.csv        # 프레임별 feature 덤프 (비교/디버그용)
└── frames/               # 구간별 best frame 1-3장
    ├── window_001_f1234.jpg
    └── window_001_f1234_meta.json
```

### windows.json schema

```json
[
  {
    "window_id": 1,
    "start_ms": 34200,
    "end_ms": 36100,
    "peak_ms": 35150,
    "score": 0.83,
    "reason": {
      "mouth_open_delta": 0.9,
      "head_velocity": 0.6,
      "wrist_raise": 0.7
    },
    "selected_frames": [
      {
        "frame_idx": 1234,
        "timestamp_ms": 35150,
        "frame_score": 0.91,
        "path": "frames/window_001_f1234.jpg"
      }
    ]
  }
]
```

## 3. 파이프라인

```
Video
  │ decode (visualbase)
  ▼
Face detect + Body pose (vpx plugins)
  │
  ▼
Numeric Feature Extraction
  │ face features + pose features + quality features
  ▼
Per-video Signal Normalization
  │ z-score (MAD) 또는 percentile
  ▼
Composite Impact Score
  │ weighted sum of deltas
  ▼
Temporal Smoothing + Peak Detection
  │ EMA/median filter → local maxima
  ▼
Window Generation + Best Frame Selection
  │ peak ± 1.0s, frame quality scoring
  ▼
Export (windows.json + timeseries.csv + frames/)
```

## 4. Feature 정의

### A. Face 기반

| Feature | 소스 | 설명 |
|---------|------|------|
| `face_bbox_area` | vpx-face-detect | 프레임 대비 얼굴 크기 비율 |
| `face_center_distance` | vpx-face-detect | 프레임 중앙까지 거리 (정규화) |
| `mouth_open_ratio` | vpx-face-expression 또는 landmark | 입 벌림 정도 |
| `eye_open_ratio` | vpx-face-expression 또는 landmark | 눈 뜸 정도 |
| `head_yaw` / `pitch` / `roll` | vpx-face-detect (5-point) | 머리 각도 |
| `head_velocity` | delta(yaw, pitch) / dt | 머리 회전 속도 (deg/sec) |

### B. 상반신 포즈

| Feature | 소스 | 설명 |
|---------|------|------|
| `wrist_raise` | vpx-body-pose | wrist_y - shoulder_y (정규화) |
| `elbow_angle_change` | vpx-body-pose | 팔꿈치 각도 변화율 |
| `torso_rotation` | vpx-body-pose | 어깨 라인 회전 |

### C. 프레임 품질 (gate 용도)

| Feature | 계산 | 용도 |
|---------|------|------|
| `blur` | Laplacian variance | **gate only** — score에 미포함 |
| `exposure` | gray mean + percentile | delta만 score에 반영 |

### D. Temporal Delta (핵심)

모든 feature에 대해:
```python
delta(t) = |feature(t) - EMA(feature, alpha=0.1)|
```

**절대값이 아닌 변화량**이 score의 주 입력.

## 5. 정규화 (Per-video, 필수)

절대 threshold 금지. 비디오마다 정규화:

```python
# 방법 1: MAD 기반 z-score
z = (feature - median(feature)) / MAD(feature)

# 방법 2: Percentile 정규화
z = (feature - p50) / (p95 - p50)
```

MAD = Median Absolute Deviation = `median(|x - median(x)|)`

## 6. Composite Impact Score

```python
impact(t) = (
    0.30 * mouth_open_delta(t) +
    0.20 * head_velocity(t) +
    0.15 * wrist_raise(t) +
    0.15 * torso_rotation(t) +
    0.10 * face_size_change(t) +
    0.10 * exposure_change(t)
)
```

가중치는 초기값이며 데이터 기반 튜닝 대상.

**blur는 score에 포함하지 않음** — gate로만 사용.

## 7. Temporal 처리

### Smoothing

```python
# EMA (추천)
smoothed(t) = alpha * raw(t) + (1 - alpha) * smoothed(t-1)
# alpha = 0.2~0.3

# 또는 median filter
smoothed = medfilt(raw, kernel_size=5~9)
```

수치 signal의 spike noise 제거 필수.

### Peak Detection

```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    smoothed_impact,
    distance=int(2.5 * fps),          # 최소 2.5초 간격
    prominence=np.percentile(smoothed_impact, 90),  # per-video 상대적
)
```

### Window 생성

```python
for peak in peaks:
    window = (peak_time - 1.0s, peak_time + 1.0s)
```

## 8. Best Frame Selection

구간 내 프레임 중 가장 사진으로 적합한 1-3장:

```python
frame_score = (
    0.4 * face_size_norm +
    0.3 * eye_open_ratio +
    0.3 * (1.0 / blur)
)
```

Hard gate 적용 후 frame_score 상위 선택.

## 9. Explainability Log (필수)

`timeseries.csv`:

```csv
frame_idx,timestamp_ms,mouth_open,eye_open,head_yaw,head_pitch,head_velocity,wrist_raise,torso_rotation,blur,exposure,impact_score
0,0.0,0.12,0.85,3.2,-1.1,5.3,0.0,0.02,180.5,128.3,0.15
1,100.0,0.15,0.83,4.1,-0.8,9.0,0.0,0.03,175.2,130.1,0.22
...
```

모든 실행에서 생성. highlight_vector와의 비교 분석에 사용.

## 10. momentscan과의 관계

### 재사용 가능한 부품

| momentscan 부품 | highlight_rules 활용 |
|----------------|---------------------|
| `algorithm/analyzers/highlight/` | Gate 로직 참고 (hysteresis, consecutive counting) |
| `algorithm/monitoring/` | PipelineMonitor 패턴 재사용 가능 |
| `algorithm/analyzers/quality/` | blur/exposure 계산 |

### 차이점

- momentscan: 실시간 스트리밍 (FlowGraph + Pathway backend)
- highlight_rules: **배치 후처리** (비디오 전체를 한 번에 처리)
- momentscan의 trigger 로직(expression_spike, head_turn 등)을 per-video 정규화 + peak detection으로 대체

## 11. vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-face-detect | 얼굴 검출, 추적, head pose |
| vpx-face-expression | 표정 proxy (mouth_open, eye_open) |
| vpx-body-pose | 상반신 포즈 (wrist, elbow, torso) |
| vpx-sdk | QualityGate, Observation 타입 |
| visualbase | 비디오 디코딩 |

## 12. 구현 계획

### Phase 1: Feature 추출 파이프라인

- [ ] 비디오 → 프레임 디코딩 (visualbase)
- [ ] vpx 플러그인으로 face/pose 추출
- [ ] numeric feature 계산 + timeseries.csv 출력
- [ ] per-video 정규화 구현

### Phase 2: Scoring + Peak Detection

- [ ] composite impact score 계산
- [ ] temporal smoothing (EMA)
- [ ] peak detection (scipy.signal.find_peaks)
- [ ] window 생성

### Phase 3: Frame Selection + Export

- [ ] window 내 best frame selection
- [ ] QualityGate 적용 (shared contracts)
- [ ] windows.json + frames/ 출력
- [ ] explainability reason 생성

### Phase 4: 평가 + 튜닝

- [ ] highlight_vector와 구간 겹침률 비교
- [ ] 가중치 튜닝 (운영 데이터 기반)
- [ ] feature importance 분석 → highlight_vector에 수치 feature 추가 여부 판단
