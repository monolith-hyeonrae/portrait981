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
output/{video_id}/highlight/
├── windows.json          # 하이라이트 구간 목록
├── timeseries.csv        # 프레임별 feature + scoring 덤프
├── score_curve.png       # 시간축 점수 그래프 + peak 마커
├── report.html           # Plotly 인터랙티브 리포트
└── frames/               # 구간별 best frame 1-3장
    ├── w1_peak_35150ms.jpg
    └── w1_best_35200ms.jpg
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
      "mouth_open_ratio": 0.9,
      "head_velocity": 0.6,
      "wrist_raise": 0.7
    },
    "selected_frames": [
      {
        "frame_idx": 1234,
        "timestamp_ms": 35150,
        "frame_score": 0.91
      }
    ]
  }
]
```

## 3. 파이프라인

`field_mapping.py`의 `PIPELINE_FIELD_MAPPINGS`이 feature → scoring role 매핑의 single source of truth.
highlight.py와 info.py가 같은 레지스트리를 읽어 일관성을 보장한다.

```
Video
  │ decode (visualbase)
  ▼
Face detect + Expression + Body pose (vpx plugins)
  │
  ▼
Numeric Feature Extraction (extract.py → FrameRecord)
  │ field_mapping.py가 모듈 출력 → record 필드 매핑 정의
  ▼
Temporal Delta (EMA baseline) + Derived Fields
  │ PIPELINE_DELTA_SPECS, PIPELINE_DERIVED_FIELDS
  ▼
Per-video Signal Normalization (MAD z-score)
  │
  ▼
QualityGate (hard filter)
  │ face_detected, face_confidence, blur_score, brightness
  │ 미측정(=0) → 통과 (blur, brightness)
  ▼
Scoring: quality_score × impact_score
  │ gate 통과 프레임만 최종 점수 산출
  ▼
Temporal Smoothing (EMA) + Peak Detection
  │ scipy.signal.find_peaks
  ▼
Window Generation + Best Frame Selection
  │ peak ± 1.0s, final_scores 상위 N개
  ▼
Export (windows.json + timeseries.csv + score_curve.png + report.html + frames/)
```

## 4. Feature 정의

모든 필드는 `field_mapping.py`의 `PIPELINE_FIELD_MAPPINGS`에 선언.
`scoring_role`과 `rationale`이 각 필드의 역할과 비즈니스 의사결정 이유를 기술한다.

| record_field | 소스 | scoring_role | rationale |
|---|---|---|---|
| `face_detected` | face.detect | gate | 얼굴이 없는 프레임은 인물 사진으로 사용 불가 |
| `face_confidence` | face.detect | gate | 오검출된 얼굴로 선별하면 의미 없는 사진이 뽑힘. 0.7 이상만 신뢰 |
| `face_area_ratio` | face.detect | quality | 얼굴이 너무 작으면 인쇄/SNS 사진으로 부적합. 화면의 1% 이상 |
| `face_center_distance` | face.detect | info | 프레임 중심 거리 참고용. 현재 scoring 미사용 |
| `head_yaw` | face.detect | quality | 정면에 가까운 사진이 고객 만족도 높음. 45도 이상 옆모습은 감점 |
| `head_pitch` | face.detect | info | 상하 회전 참고. head_velocity 파생 필드의 소스 |
| `head_roll` | face.detect | info | 머리 기울기 참고. 현재 scoring 미사용 |
| `mouth_open_ratio` | face.expression | impact | 환호/놀람 등 감정 표현이 큰 순간이 라이드 하이라이트 |
| `eye_open_ratio` | face.expression | gate | 눈 감은 사진은 인물 사진으로 부적합. 0.15 미만이면 탈락 |
| `smile_intensity` | face.expression | impact | 미소 피크는 감정적으로 좋은 순간. 평소 무표정/부정 표정 대비 미소 급등이 특히 의미 있음 |
| `wrist_raise` | body.pose | impact | 손을 올리는 동작은 라이드 즐기는 대표적 제스처 |
| `torso_rotation` | body.pose | impact | 상체 움직임이 큰 순간 = 활발한 반응 구간 |
| `hand_near_face` | body.pose | info | 손으로 얼굴 가리는 순간 감지용. 향후 gate 추가 후보 |
| `elbow_angle_change` | body.pose | info | 팔 동작 크기. wrist_raise와 중복도 높아 현재 미사용 |
| `blur_score` | frame.quality | quality | 모션블러가 심한 프레임은 사진 품질 열화. Laplacian 50 미만 탈락 |
| `brightness` | frame.quality | gate | 너무 밝거나 너무 어두우면 후보정으로도 복구 어려움. 40-220 범위만 통과 |
| `contrast` | frame.quality | info | 대비 정보. 현재 brightness와 blur로 충분하여 미사용 |
| `main_face_confidence` | face.classify | info | 주탑승자 분류 신뢰도 참고용 |
| `frame_score` | frame.scoring | info | 종합 프레임 점수 참고용 |

### D. Temporal Delta (핵심)

delta 대상 필드 (`PIPELINE_DELTA_SPECS`):
```python
delta(t) = |feature(t) - EMA(feature, alpha=0.1)|
```

대상: `mouth_open_ratio`, `smile_intensity`, `head_yaw`, `head_pitch`, `wrist_raise`, `torso_rotation`, `face_area_ratio`, `brightness`

**절대값이 아닌 변화량**이 score의 주 입력.

### E. 파생 필드 (PIPELINE_DERIVED_FIELDS)

| 필드명 | 소스 | 계산 |
|---|---|---|
| `head_velocity` | head_yaw, head_pitch | `sqrt(delta_yaw^2 + delta_pitch^2) / dt` (deg/sec) |
| `frontalness` | head_yaw | `1 - |yaw| / max_yaw`, clamped [0, 1] |

## 5. 정규화 (Per-video, 필수)

절대 threshold 금지. 비디오마다 정규화.

현재 구현: **MAD 기반 z-score**

```python
z = (feature - median(feature)) / MAD(feature)
# MAD = Median Absolute Deviation = median(|x - median(x)|)
# MAD < 1e-8 이면 z = 0 (상수 신호)
```

대안 (미사용): Percentile 정규화
```python
z = (feature - p50) / (p95 - p50)
```

## 6. Scoring: Gate × Impact (곱 구조)

weighted sum만 쓰면 "감정이 좋은데 흐린 프레임"이 뽑히고,
quality만 좋으면 "사진은 괜찮은데 임팩트 없는 장면"이 뽑힌다.
이를 방지하기 위해 **gate와 impact를 곱 구조**로 결합한다.

### Step 1: QualityGate (hard filter)

gate를 통과하지 못하면 해당 프레임의 최종 점수는 0.
미측정 값(=0)은 통과 처리 (blur_score, brightness, eye_open_ratio).

```python
quality_gate(t) = (
    face_detected(t)
    and face_confidence(t) >= 0.7
    and face_area_ratio(t) >= 0.01
    and (blur_score(t) == 0 or blur_score(t) >= 50)        # 미측정 → 통과
    and (brightness(t) == 0 or 40 <= brightness(t) <= 220)  # 미측정 → 통과
    and (eye_open_ratio(t) == 0 or eye_open_ratio(t) >= 0.15)  # 미측정 → 통과
)
```

### Step 2: Quality Score (연속 품질)

gate 통과 프레임에 대해 연속적 품질 점수를 계산.
최종 점수에 곱으로 반영되어, 같은 impact라도 quality가 높은 프레임이 우선된다.

```python
quality_score(t) = (
    0.4 * blur_norm(t) +             # 선명도 (per-video min-max 정규화)
    0.3 * face_size_norm(t) +         # 얼굴 크기 (per-video min-max 정규화)
    0.3 * frontalness(t)              # 정면 근접도 (1 - |yaw|/45)
)
```

### Step 3: Impact Score (감정/동작 변화)

MAD z-score 정규화된 delta에 **ReLU** 적용 후 가중합. 평균 이상 변화만 기여한다.

```python
relu = lambda x: max(x, 0)

impact(t) = (
    0.35 * relu(normed_smile_intensity(t)) +
    0.15 * relu(normed_head_yaw_delta(t)) +
    0.12 * relu(normed_mouth_open_ratio(t)) +
    0.10 * relu(normed_head_velocity(t)) +
    0.08 * relu(normed_wrist_raise(t)) +
    0.08 * relu(normed_torso_rotation(t)) +
    0.06 * relu(normed_face_area_ratio(t)) +
    0.06 * relu(normed_brightness(t))
)
```

가중치는 초기값이며 데이터 기반 튜닝 대상. 합계 = 1.00.

### Step 4: 최종 점수

```python
final_score(t) = quality_score(t) * impact(t) if quality_gate(t) else 0
```

**blur는 impact에 미포함** — quality_score와 gate에서만 사용.

## 7. Temporal 처리

### Smoothing

```python
smoothed(t) = alpha * raw(t) + (1 - alpha) * smoothed(t-1)
# alpha = 0.25
```

EMA로 spike noise 제거.

### Peak Detection

```python
from scipy.signal import find_peaks

positive_scores = smoothed[smoothed > 0]
prominence = np.percentile(positive_scores, 90)  # per-video 상대적

peaks, _ = find_peaks(
    smoothed,
    distance=int(2.5 * fps),          # 최소 2.5초 간격
    prominence=prominence,
)
```

### Window 생성

```python
for peak in peaks:
    window = (peak_time - 1.0s, peak_time + 1.0s)
```

## 8. Best Frame Selection

구간 내 프레임 중 `final_scores` 상위 N개 (기본 3장) 선택.
`final_scores > 0`인 프레임만 대상.

```python
window_scores = final_scores[start_idx:end_idx + 1]
window_indices = np.argsort(window_scores)[::-1][:best_frame_count]
# final_scores <= 0인 프레임 제외
```

## 9. Explainability Log (필수)

`timeseries.csv`:

```csv
frame_idx,timestamp_ms,gate_pass,quality_score,impact_score,final_score,smoothed_score,is_peak,face_detected,face_confidence,face_area_ratio,head_yaw,head_pitch,mouth_open_ratio,eye_open_ratio,smile_intensity,wrist_raise,torso_rotation,blur_score,brightness
0,0.0,1,0.6543,0.1523,0.0997,0.0997,0,1,0.920,0.0400,3.2,-1.1,0.120,0.850,0.300,0.000,0.020,180.5,128.3
...
```

모든 실행에서 생성. highlight_vector와의 비교 분석에 사용.

## 10. 테마파크 라이드 특화 주의사항

온라이드 촬영 환경에서 흔히 발생하는 문제와 대응:

| 문제 | 원인 | 대응 |
|------|------|------|
| 표정 검출 불안정 | 조명 변화, 그림자, 모션블러 | intensity 절대값 대신 **상대 변화(delta)**만 사용. 이미 §4.D로 반영 |
| 헤드포즈 노이즈 | ride 중 지속적 흔들림 | raw yaw/pitch를 impact에 직접 쓰지 않음. **head_velocity(변화율)**와 **정면 근접도(gate/quality)**로만 활용 |
| 손 올림 = 얼굴 가림 | wrist_raise와 occlusion 동시 발생 | `hand_near_face`는 info로 기록. 향후 gate 추가 후보 |
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
| vpx-face-expression | 표정 proxy (mouth_open, eye_open, smile_intensity) |
| vpx-body-pose | 상반신 포즈 (wrist, elbow, torso) |
| vpx-sdk | QualityGate, Observation 타입 |
| visualbase | 비디오 디코딩 |

## 14. 구현 계획

### Phase 1: Feature 추출 파이프라인

- [x] 비디오 → 프레임 디코딩 (visualbase)
- [x] vpx 플러그인으로 face/pose 추출
- [x] numeric feature 계산 + timeseries.csv 출력
- [x] per-video 정규화 구현

### Phase 2: Scoring + Peak Detection

- [x] composite impact score 계산
- [x] temporal smoothing (EMA)
- [x] peak detection (scipy.signal.find_peaks)
- [x] window 생성

### Phase 3: Frame Selection + Export

- [x] window 내 best frame selection
- [x] QualityGate 적용
- [x] windows.json + frames/ + report.html + score_curve.png 출력
- [x] explainability reason 생성

### Phase 4: 평가 + 튜닝

- [ ] highlight_vector와 구간 겹침률 비교
- [ ] 가중치 튜닝 (운영 데이터 기반)
- [ ] feature importance 분석 → highlight_vector에 수치 feature 추가 여부 판단
