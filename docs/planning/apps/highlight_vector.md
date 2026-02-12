# highlight_vector — 임베딩 기반 하이라이트 엔진

> **momentscan Phase 2** 상세 설계. 마스터 플랜: [momentscan.md](momentscan.md)
>
> DINOv2/SigLIP temporal delta로 semantic saliency를 감지하고,
> 선택적으로 GRU-lite reranker를 학습하여 highlight 구간을 잡는 시스템.
> Phase 1(highlight_rules)과 병렬 비교하여 효과를 검증하는 실험적 접근.

## 1. 목적

- **임베딩 temporal delta**로 "의미적 변화"를 감지
- Expression classifier에 의존하지 않음 — 범용 visual change 감지
- Pairwise ranking loss로 학습 가능한 reranker (shadow mode부터)
- highlight_rules와 병렬 운영 + 비교 평가

## 2. 입출력

### Input

- 라이드 비디오 (~2분), post-ride 배치 처리

### Output

```
output/{video_id}/highlight_vector/
├── windows.json          # 하이라이트 구간 목록
├── frames/               # 구간별 best frame 1-3장
│   ├── window_001_f1234.jpg
│   └── window_001_f1234_meta.json
└── debug/
    ├── score_curve.json   # 시간별 score 곡선
    ├── peak_candidates.json
    └── filtering_log.json
```

### windows.json schema

```json
[
  {
    "window_id": 1,
    "start_ms": 34200,
    "end_ms": 36100,
    "peak_ms": 35150,
    "score_rule": 0.83,
    "score_model": null,
    "key_features": {
      "d_face": 0.42,
      "d_body": 0.31,
      "motion": 0.55,
      "audio": 0.0
    },
    "selected_frames": [
      {
        "frame_idx": 1234,
        "timestamp_ms": 35150,
        "quality_score": 0.91,
        "face_bbox": [120, 80, 320, 380],
        "crop_box": [100, 60, 340, 400],
        "blur": 180.5,
        "exposure_mean": 128.3
      }
    ]
  }
]
```

## 3. 파이프라인

```
Video
  │ decode + clip indexing (1.0s clip, 0.5s stride)
  ▼
Face detect + tracking + main person select (1-2명)
  │
  ▼
Embedding Extraction
  │ face crop → e_face (DINOv2/SigLIP)
  │ upper-body crop → e_body (DINOv2/SigLIP)
  │ + cheap signals (blur, exposure, motion)
  ▼
Temporal Delta Scoring
  │ d(t) = ||e(t) - EMA(e)||
  ▼
Candidate Proposal (peak detection)
  │ per-video 상대 threshold
  ▼
[Optional] Temporal Reranker (GRU-lite / TCN-lite)
  │ shadow mode → canary → full
  ▼
Best Frame Selection (photo-quality gate)
  │
  ▼
Export (windows.json + frames/ + debug/)
```

## 4. 임베딩 모델 선택

| 모델 | 특징 | 적합 시나리오 |
|------|------|--------------|
| **DINOv2** | 픽셀/텍스처 변화에 민감, 표정 미세 변화 잘 감지 | emotion peak / surprise 중심 |
| **SigLIP** | semantic 변화에 민감, identity invariance 적음 | portrait + impact 균형 |
| **OpenCLIP** | 안정적, 검색 확장성 | 범용 MVP |

추천 순위: `SigLIP ≈ DINOv2 > OpenCLIP >>> Face-ID models`

Face-ID 임베딩(ArcFace)은 identity invariance가 강해서 표정/포즈 변화에 둔감 → highlight에 부적합.

**DINOv2 주의사항**: 배경 변화에도 민감 → crop stabilization 필수 (bbox EMA smoothing).

## 5. Temporal Delta (핵심 scoring)

### 임베딩 추출

```python
# 프레임별 or clip pooling (3-5 frame average)
e_face(t) = embed_model(face_crop(t))      # L2 normalized
e_body(t) = embed_model(body_crop(t))      # L2 normalized
```

Pooling 강력 추천: 인접 3-5프레임 평균으로 noise 감소.

### Delta 계산

```python
# EMA baseline
ema_face(t) = alpha * e_face(t) + (1 - alpha) * ema_face(t-1)

# Temporal delta
d_face(t) = || e_face(t) - ema_face(t) ||
d_body(t) = || e_body(t) - ema_body(t) ||

# Combined
d(t) = max(d_face(t), d_body(t))   # 또는 weighted sum
```

### Composite score

```python
s(t) = w1 * d(t) + w2 * motion(t) + w3 * audio(t)
# w1=0.6, w2=0.3, w3=0.1 (초기값)
```

- `motion`: optical flow magnitude 또는 pose velocity
- `audio`: RMS energy peak (optional, 없으면 w3=0)

### Smoothing + Peak Detection

```python
# EMA smoothing
smoothed_s(t) = 0.3 * s(t) + 0.7 * smoothed_s(t-1)

# Peak detection
peaks = find_peaks(
    smoothed_s,
    distance=int(3.0 * fps),      # 최소 3초 간격
    prominence=percentile(smoothed_s, 95),
)

# Window: peak ± 1.0s
```

## 6. Temporal Reranker (Optional, Phase 2)

후보 구간(candidate windows)을 재정렬하는 작은 시퀀스 모델.

### 모델 옵션

| 모델 | 구조 | 파라미터 | 특징 |
|------|------|---------|------|
| GRU-lite | 64-128 hidden, 1-2 layers | ~50K | 순서 의존성 학습 |
| TCN-lite | depthwise separable conv, dilation 1/2/4 | ~30K | 병렬 추론, edge-friendly |

### Input

각 candidate window의:
- clip embedding sequence (5-10 clips)
- cheap signal features (blur, exposure, motion, ...)
- rule score (feature로만 사용, teacher mimic 금지)

### Training

```python
# Pairwise ranking loss
L = max(0, margin - score(positive) + score(negative))

# positive: 구매/저장/공유된 구간 (없으면 rule top-k를 weak positive로)
# negative: 같은 비디오의 다른 candidate 구간
```

**Rule score를 teacher로 mimic하는 것은 금지** — feature로만 입력.

### 배포 전략

1. **Shadow mode**: rule pick과 model pick 모두 저장, 비교만
2. **Canary**: 1-5% 트래픽에 model pick 적용
3. **Full**: 메트릭 확인 후 전환
4. 메트릭 하락 시 자동 rollback

## 7. Best Frame Selection

### Hard Gate (shared QualityGate)

```python
gate.check(frame, face_bbox, level=GateLevel.STRICT)
```

- blur, face_size, exposure, occlusion 기준

### Soft Preference

```python
frame_quality = (
    0.3 * face_size_norm +
    0.3 * eye_open_ratio +
    0.2 * frontality +      # |yaw| + |pitch| 낮을수록 좋음
    0.2 * (1.0 / blur)
)
```

### Duplicate Suppression

MMR(Maximal Marginal Relevance) 스타일:
```python
# 이미 선택된 프레임과 cosine similarity 높으면 제외
if max(cos(e_new, e_selected)) > 0.9:
    skip
```

## 8. Clip Indexing

```python
clip_len = 1.0    # seconds
clip_stride = 0.5  # seconds (50% overlap)

# clip 내 representative frame: center frame
# clip 내 pooling frames: 3-5 등간격 샘플
```

## 9. vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-face-detect | 얼굴 검출, 추적, head pose, face crop |
| vpx-body-pose | upper-body crop |
| vpx-sdk | QualityGate, Observation |
| visualbase | 비디오 디코딩 |

### 신규 vpx 플러그인 필요

| 플러그인 | 역할 | 비고 |
|---------|------|------|
| **vpx-vision-embed** | DINOv2/SigLIP/OpenCLIP 임베딩 추출 | identity_builder와 공유 |

## 10. highlight_rules와의 비교

| 항목 | highlight_rules | highlight_vector |
|------|----------------|-----------------|
| Score 기반 | 수치 feature delta | 임베딩 temporal delta |
| 정규화 | per-video z-score | per-video percentile |
| Peak detection | 동일 (find_peaks) | 동일 |
| 해석성 | 높음 (feature별 기여도) | 낮음 (delta 크기만) |
| 학습 | 없음 | pairwise reranker (optional) |
| Compute | CPU only | GPU (임베딩 추출) |

### 병렬 비교 메트릭

- 구간 겹침률 (IoU)
- 선택 프레임 품질 평균
- 구매/저장 상관성 (운영 데이터 기반)

## 11. 구현 계획

### Phase 1: 임베딩 추출 파이프라인

- [ ] vpx-vision-embed 플러그인 구현 (DINOv2 우선)
- [ ] face crop + body crop 추출
- [ ] clip indexing + pooling
- [ ] temporal delta 계산 + score curve 출력

### Phase 2: Candidate Proposal + Frame Selection

- [ ] peak detection (per-video 상대 threshold)
- [ ] window 생성
- [ ] QualityGate 기반 best frame selection
- [ ] windows.json + frames/ 출력
- [ ] highlight_rules와 겹침률 비교 도구

### Phase 3: Reranker (Shadow Mode)

- [ ] GRU-lite 또는 TCN-lite 모델 구현
- [ ] pairwise ranking loss 학습 파이프라인
- [ ] shadow mode: rule pick vs model pick 동시 저장
- [ ] 비교 리포트 자동 생성

### Phase 4: 배포 + 고도화

- [ ] canary 배포
- [ ] highlight_rules feature를 reranker input에 추가 (hybrid)
- [ ] audio signal 통합 (optional)
