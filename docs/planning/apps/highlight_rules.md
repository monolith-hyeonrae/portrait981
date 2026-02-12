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
QualityGate (hard filter)
  │ blur, exposure, occlusion_by_hand → 불통과 시 score=0
  ▼
Scoring: quality_score × impact_score
  │ gate 통과 프레임만 최종 점수 산출
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
| `hand_near_face` | vpx-body-pose | wrist와 nose 사이 거리 (정규화). 얼굴 가림 감지용 |

### C. 프레임 품질 (gate + quality score)

| Feature | 계산 | 용도 |
|---------|------|------|
| `blur` | Laplacian variance | **gate + quality_score** — impact에는 미포함 |
| `exposure` | gray mean + percentile | gate + delta만 impact에 반영 |
| `occlusion_by_hand` | `hand_near_face < τ` | **gate** — 손이 얼굴 가리면 차단 |

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

## 6. Scoring: Gate × Impact (곱 구조)

weighted sum만 쓰면 "감정이 좋은데 흐린 프레임"이 뽑히고,
quality만 좋으면 "사진은 괜찮은데 임팩트 없는 장면"이 뽑힌다.
이를 방지하기 위해 **gate와 impact를 곱 구조**로 결합한다.

### Step 1: QualityGate (hard filter)

gate를 통과하지 못하면 해당 프레임의 최종 점수는 0.

```python
quality_gate(t) = (
    face_confidence(t) >= 0.7
    and face_area_ratio(t) >= 0.01
    and blur(t) >= tau_blur          # 높을수록 선명
    and 40 <= exposure_mean(t) <= 220
    and not occlusion_by_hand(t)     # 손 제스처로 인한 얼굴 가림 체크
)
```

### Step 2: Quality Score (연속 품질)

gate 통과 프레임에 대해 연속적 품질 점수를 계산.
최종 점수에 곱으로 반영되어, 같은 impact라도 quality가 높은 프레임이 우선된다.

```python
quality_score(t) = (
    0.4 * blur_norm(t) +             # 선명도 (per-video 정규화)
    0.3 * face_size_norm(t) +         # 얼굴 크기
    0.3 * frontalness(t)              # 정면 근접도 (1 - |yaw|/45)
)
```

### Step 3: Impact Score (감정/동작 변화)

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

### Step 4: 최종 점수

```python
final_score(t) = quality_score(t) * impact(t) if quality_gate(t) else 0
```

**blur는 impact에 미포함** — quality_score와 gate에서만 사용.

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

## 10. 테마파크 라이드 특화 주의사항

온라이드 촬영 환경에서 흔히 발생하는 문제와 대응:

| 문제 | 원인 | 대응 |
|------|------|------|
| 표정 검출 불안정 | 조명 변화, 그림자, 모션블러 | intensity 절대값 대신 **상대 변화(delta)**만 사용. 이미 §4.D로 반영 |
| 헤드포즈 노이즈 | ride 중 지속적 흔들림 | raw yaw/pitch를 impact에 직접 쓰지 않음. **head_velocity(변화율)**와 **정면 근접도(gate/quality)**로만 활용 |
| 손 올림 = 얼굴 가림 | wrist_raise와 occlusion 동시 발생 | `hand_near_face` gate로 **얼굴 가림 시 차단**. wrist_raise가 높아도 사진이 망하면 제외 |
| blur가 가장 치명적 | 카메라/탑승객 모두 움직임 | blur 나쁘면 impact 무관하게 무조건 제외 (hard gate). quality_score에도 blur 반영 |

## 11. Phase 2 (highlight_vector)와의 연결

Phase 1과 Phase 2는 **즉시 통합하지 않고 병렬 비교** 후 단계적으로 연결한다.

### 단계 1: 병렬 비교 (현재 목표)
- 동일 비디오에서 Phase 1 window와 Phase 2 window를 각각 생성
- overlap rate, 선택 프레임 품질 평균으로 비교

### 단계 2: 후보 → 리랭크 파이프라인
- Phase 1이 Top-K window 후보를 생성
- Phase 2 embedding reranker가 후보를 재정렬
- 규칙 기반의 해석 가능성은 유지하면서 학습형 모델로 정밀도 향상

### 단계 3: Hybrid (장기)
- Phase 1의 feature importance 분석 결과를 기반으로
- Phase 2 모델 input에 유의미한 수치 feature를 추가
- 최종적으로 hybrid highlight 시스템으로 통합

**지금은 단계 1에만 집중. 단계 2 이후는 Phase 2 구현 + 비교 데이터 확보 후 결정.**

## 12. momentscan과의 관계

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

## 13. vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-face-detect | 얼굴 검출, 추적, head pose |
| vpx-face-expression | 표정 proxy (mouth_open, eye_open) |
| vpx-body-pose | 상반신 포즈 (wrist, elbow, torso) |
| vpx-sdk | QualityGate, Observation 타입 |
| visualbase | 비디오 디코딩 |

## 14. 구현 계획

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
