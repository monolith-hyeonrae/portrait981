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
      "smile_intensity": 0.9,
      "portrait_best": 0.7,
      "head_yaw": 0.4
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
Analyzers (DAG):
  face.detect → face.classify → face.baseline (Welford online stats)
       │ → face.expression, face.quality, face.parse, portrait.score, face.au, head.pose
       └→ face.gate (depends: detect+classify, optional: quality+frame.quality+head.pose)
  body.pose, hand.gesture, frame.quality (independent)
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
face.gate (DAG에서 per-frame 판정 완료 → gate_passed 읽기)
  │ confidence, blur, exposure, contrast, parsing_coverage
  ▼
Scoring: 0.35×Quality + 0.65×Impact (가산)
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
| `face_confidence` | face.detect | gate | 오검출된 얼굴로 선별하면 의미 없는 사진이 뽑힘. ≥0.7만 신뢰 |
| `face_area_ratio` | face.detect | gate+quality | 얼굴이 너무 작으면 인쇄/SNS 사진으로 부적합. ≥2% |
| `face_center_distance` | face.detect | info | 프레임 중심 거리 참고용. 현재 scoring 미사용 |
| `head_yaw` | face.detect / head.pose | impact | delta → MAD z-score → relu. 급격한 회전이 하이라이트 |
| `head_pitch` | face.detect / head.pose | info | 상하 회전 참고 |
| `head_roll` | face.detect / head.pose | info | 머리 기울기 참고. 현재 scoring 미사용 |
| `head_blur` | face.quality | gate+quality | 얼굴 크롭 Laplacian 분산. gate: ≥5.0 (main), ≥20.0 (passenger) |
| `head_exposure` | face.quality | info | 얼굴 크롭 평균 밝기 [40-220] |
| `head_contrast` | face.quality | gate | CV=std/mean. ≥0.05 (flat/washed-out 배제) |
| `clipped_ratio` | face.quality | gate | 과노출 픽셀(>250) 비율. ≤30% |
| `crushed_ratio` | face.quality | gate | 저노출 픽셀(<5) 비율. ≤30% |
| `parsing_coverage` | face.quality | gate | BiSeNet 마스크 커버리지. ≥15% |
| `seg_face/eye/mouth/hair` | face.quality | info | 시맨틱 세그멘테이션 비율 |
| `smile_intensity` | face.expression | impact | 미소 피크. per-video min-max 절대값. 이 영상에서 가장 웃는 순간 포착 |
| `portrait_best` | portrait.score | impact | CLIP 4축(disney_smile, charisma, wild_roar, playful_cute) 중 프레임별 max |
| `head_aesthetic` | portrait.score | info | CLIP aggregate score. 미학적 구분 근거 부족으로 scoring 제외 |
| `clip_disney_smile/charisma/wild_roar/playful_cute` | portrait.score | info | CLIP 4축 개별 점수. portrait_best의 소스 |
| `AU6/AU12/AU25/AU26` | face.au | info | Action Units: cheek_raiser, lip_corner, lips_part, jaw_drop |
| `duchenne_smile/wild_intensity/chill_score` | composites | info | 크로스-analyzer 복합 지표. timeline 표시용 |
| `blur_score` | frame.quality | gate | 프레임 전체 Laplacian. face.quality 없을 때 fallback (≥50) |
| `brightness` | frame.quality | gate | 프레임 전체 평균 밝기. face.quality 없을 때 fallback [40-220] |
| `contrast` | frame.quality | info | 프레임 전체 대비 정보 참고용 |
| `main_face_confidence` | face.classify | info | 주탑승자 분류 신뢰도 참고용 |
| `gate_passed` | face.gate | gate | per-frame main face gate 판정 결과 |

### D. Temporal Delta (핵심)

delta 대상 필드 (`PIPELINE_DELTA_SPECS`):
```python
delta(t) = |feature(t) - EMA(feature, alpha=0.1)|
```

대상: `smile_intensity`, `head_yaw`, `face_area_ratio`, `brightness`, `duchenne_smile`, `wild_intensity`

**절대값이 아닌 변화량**이 score의 주 입력 (단, smile_intensity는 예외 — per-video min-max 절대값 사용).

### E. 파생 필드 (PIPELINE_DERIVED_FIELDS)

| 필드명 | 소스 | 계산 |
|---|---|---|
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

## 6. Face Gate — `face.gate` analyzer

> 상세 문서: **[gate_policy.md](gate_policy.md)** (방법론, 임계값 근거, 역할별 전략)

DAG에서 per-frame 판정하는 전용 analyzer.
`depends: [face.detect, face.classify]`, `optional_depends: [face.quality, frame.quality, head.pose]`.

**주탑승자(main)**: binary gate (PASS/FAIL). 미통과 프레임은 peak/best frame 후보에서 완전 배제.
**동승자(passenger)**: suitability score (0.0~1.0). 차단 없음, scoring에서 additive bonus로 반영.

## 7. Quality Score (연속 품질)

gate 통과 프레임에 대해 연속적 품질 점수를 계산.

```python
quality_score(t) = (
    0.30 * head_blur_norm(t) +        # 얼굴 크롭 선명도 (per-video min-max)
    0.20 * face_size_norm(t) +         # 얼굴 크기 (per-video min-max)
    0.30 * face_identity(t)            # ArcFace anchor similarity
    # face_identity 없으면 → 0.25 * frontalness(t) (1 - |yaw|/max_yaw)
)
```

face_identity (ArcFace) 사용 시 가중치 합 = 0.80, frontalness fallback 시 = 0.75.
정규화: `quality / sum(weights)` → [0, 1].

## 8. Impact Score (Top-K 가중 평균)

3채널 중 상위 K개(기본 K=3) 시그널의 가중 평균.

```python
channels = {
    "smile_intensity": (0.25, per_video_minmax(smile_intensity)),  # 절대값!
    "head_yaw":        (0.15, relu(mad_zscore(delta_head_yaw))),
    "portrait_best":   (0.25, max(clip_disney, clip_charisma, clip_wild, clip_playful)),
}

# Top-K: weight 기준 상위 K개 채널 선택
top_k = sorted(channels, key=weight, reverse=True)[:3]
max_achievable = sum(w for w in top_k_weights)

impact(t) = sum(w_i * v_i for top-K) / max_achievable  # [0, 1]
```

**smile_intensity는 예외**: delta가 아닌 per-video min-max 절대값 사용.
절대값이 높은 프레임이 좋은 사진이므로 '이 영상에서 가장 웃는 순간' 포착.

**삭제된 채널**: mouth_open, head_velocity, wrist_raise, torso_rotation, face_area_ratio, brightness.

## 9. 최종 점수 (가산 구조)

```python
base(t) = 0.35 * quality_score(t) + 0.65 * impact(t)
final_score(t) = base(t) + 0.30 * passenger_suitability(t)
# passenger_suitability: 동승자 적합 시 0.30 가산, 미감지/부적합 시 0
# gate_passed=False 프레임은 peak/best frame 후보에서 배제 (score 자체는 산출)
```

**곱이 아닌 덧셈**. Quality가 0이어도 Impact가 높으면 점수가 나온다.
동승자 보너스도 가산 구조 — 동승자 부적합 시 차감 없음.
Gate는 peak detection / best frame 선택에서만 필터링 (final score 계산은 전 프레임).

## 10. Temporal 처리

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

## 11. Best Frame Selection

구간 내 프레임 중 `final_scores` 상위 N개 (기본 3장) 선택.
`final_scores > 0`인 프레임만 대상.

```python
window_scores = final_scores[start_idx:end_idx + 1]
window_indices = np.argsort(window_scores)[::-1][:best_frame_count]
# final_scores <= 0인 프레임 제외
```

## 12. Explainability Log (필수)

`timeseries.csv`:

```csv
frame_idx,timestamp_ms,gate_pass,quality_score,impact_score,final_score,smoothed_score,is_peak,face_detected,face_confidence,face_area_ratio,head_yaw,head_pitch,smile_intensity,head_blur,head_exposure,parsing_coverage,portrait_best,blur_score,brightness
0,0.0,1,0.6543,0.1523,0.152,0.152,0,1,0.920,0.0400,3.2,-1.1,0.300,45.2,128.3,0.82,0.31,180.5,128.3
...
```

모든 실행에서 생성. highlight_vector와의 비교 분석에 사용.

## 13. 테마파크 라이드 특화 주의사항

온라이드 촬영 환경에서 흔히 발생하는 문제와 대응:

| 문제 | 원인 | 대응 |
|------|------|------|
| 표정 검출 불안정 | 조명 변화, 그림자, 모션블러 | smile_intensity만 per-video min-max 절대값. head_yaw는 delta 사용 |
| 헤드포즈 노이즈 | ride 중 지속적 흔들림 | head_yaw delta → MAD z-score → relu. 급격한 변화만 impact에 기여 |
| 손 올림 = 얼굴 가림 | wrist_raise와 occlusion 동시 발생 | `hand_near_face`는 info로 기록. 향후 gate 추가 후보 |
| blur가 가장 치명적 | 카메라/탑승객 모두 움직임 | face.gate에서 head_blur ≥5.0 hard gate. quality_score에도 head_blur(0.30) 반영 |
| 조명 불균일 | 터널/실외 전환 | face.quality의 local contrast(CV), clipped/crushed ratio로 gate 판정 |

## 14. Phase 2 (highlight_vector)와의 연결

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

## 15. momentscan과의 관계

### 재사용 가능한 부품

| momentscan 부품 | highlight_rules 활용 |
|----------------|---------------------|
| `algorithm/analyzers/frame_gate/` | face.gate analyzer (per-frame gate 판정) |
| `algorithm/analyzers/face_quality/` | face crop blur/exposure + BiSeNet seg ratios |
| `algorithm/analyzers/face_baseline/` | Welford online stats (face.baseline) |
| `algorithm/batch/` | BatchHighlightEngine (scoring + peak detection) |
| `algorithm/monitoring/` | PipelineMonitor 패턴 재사용 가능 |

### 차이점

- momentscan: 실시간 스트리밍 (FlowGraph + Pathway backend)
- highlight_rules: **배치 후처리** (비디오 전체를 한 번에 처리)
- momentscan의 trigger 로직(expression_spike, head_turn 등)을 per-video 정규화 + peak detection으로 대체

## 16. vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-face-detect | 얼굴 검출, 추적 |
| vpx-face-expression | 표정 (smile_intensity 등) |
| vpx-face-parse | BiSeNet 19-class segmentation |
| vpx-portrait-score | CLIP 4축 aesthetic scoring |
| vpx-face-au | Action Unit 분석 |
| vpx-head-pose | 6DoF head pose |
| vpx-body-pose | 상반신 포즈 |
| vpx-hand-gesture | 제스처 감지 |
| vpx-sdk | Module, Observation 타입, crop 유틸리티 |
| visualbase | 비디오 디코딩 |

## 17. 구현 계획

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
