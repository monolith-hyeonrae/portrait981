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

## 2. 왜 Emotion Classifier가 아닌 Embedding인가

하이라이트는 **identity problem이 아닌 state-change problem**이다.
글로벌 video AI 팀들이 face landmark/emotion classifier를 중심에 두지 않는 이유:

### Emotion Classifier의 구조적 한계

| 문제 | 설명 |
|------|------|
| **라벨 불안정** | "놀람 / 공포 / 즐거움" 경계가 모호. 문화/개인/ride마다 다름 |
| **거짓 확신** | 모델의 confidence가 높아 보이지만 실제 오판 빈번 (blur + extreme pose) |
| **temporal context 부재** | 감정은 "순간 값"이 아니라 **변화 과정**. classifier는 프레임 단위 독립 판정 |
| **domain shift** | 학습 데이터(정면, 정적)와 온라이드 환경(동적, 그림자, 모션블러)의 괴리 |

### 대안: State Change + Context

```
emotion = label ❌
emotion = emergent pattern ⭕
```

감정을 직접 분류하지 않고, **embedding temporal delta + quality gate**로 "이전 대비 얼마나 튀는가"를 측정.
놀람, 웃음 burst, 몸 반응 같은 순간은 delta가 커지는 패턴으로 자연스럽게 감지된다.

### Face Landmark의 역할 축소

Face landmark와 head pose는 **crop 안정화와 tracking 보조**로만 사용.
정면도 판정도 landmark 대신 embedding 기반 방법을 사용한다 (§7 참조).

## 3. 입출력

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

## 4. 파이프라인

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

## 5. 임베딩 모델 선택

추천 순위 (온라이드 하이라이트 기준): `SigLIP ≈ DINOv2 > OpenCLIP >>> Face-ID models`

### 모델별 특성

| 모델 | 장점 | 온라이드 한계 | 적합 시나리오 |
|------|------|--------------|--------------|
| **DINOv2** | self-supervised, 픽셀/텍스처 미세 변화에 민감. temporal delta와 조합 시 특히 강함 | 배경 변화에도 민감 → crop 안정화 필수 | emotion peak / surprise 중심 |
| **SigLIP** | contrastive 강도가 CLIP보다 강함. identity invariance 적어 상태 변화가 잘 남음. 얼굴+포즈+제스처 동시 반영 | 실전 레퍼런스가 OpenCLIP보다 적음. weight 선택 실수 시 noise 증가 | portrait + impact 균형 (추천) |
| **OpenCLIP** | 안정적, 문서/도구 풍부. 배경+사람+포즈를 고르게 담음 | identity invariance가 의외로 강함. 감정 피크 민감도 중간. motion blur에서 변화 감도 저하 | 범용 MVP / 텍스트 검색 확장 |
| **Face-ID (ArcFace)** | 동일인 판정에 매우 강함 | 표정/포즈 변화를 **의도적으로 무시**하도록 학습됨 (invariance) | highlight에 부적합. tracking/sanity check 용도로만 |

### 실무 추천 조합

1. Face crop → **DINOv2 or SigLIP** (상태 변화 민감)
2. Upper-body crop → **동일 모델** (전신 context 포함)
3. InsightFace(ArcFace)는 **tracking 안정화 + crop quality proxy**로만 사용

### Crop Stabilization (DINOv2 필수)

DINOv2는 배경 변화에도 민감하므로 crop 안정화가 필수:

```python
# bbox EMA smoothing
smoothed_bbox(t) = alpha * bbox(t) + (1 - alpha) * smoothed_bbox(t-1)
# alpha = 0.3~0.5
```

## 6. Temporal Delta (핵심 scoring)

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

## 7. Embedding 기반 정면도 (Frontality without Landmarks)

Face landmark의 yaw/pitch 추정은 가림/블러에 취약하고 모델 변경 시 기준이 흔들린다.
Embedding 기반 정면도 판정이 더 robust한 대안:

### 방법 A: Embedding Self-Consistency

정면 얼굴은 프레임 간 embedding 변화가 작고 안정적인 특성을 이용.

```python
# 인접 k 프레임의 embedding variance
score_front(t) = -Var(e(t-k : t+k))
# variance 낮을수록 = 정면 + 안정적
```

### 방법 B: Canonical Distance

영상에서 "가장 안정적인 구간"을 canonical(기준)으로 잡고, 다른 프레임과의 거리를 측정.

```python
# canonical = 가장 embedding variance가 낮은 구간의 평균 embedding
score_front(t) = -|| e(t) - e(canonical) ||
# canonical에 가까울수록 = 정면 + 잘 나온 얼굴
```

### 방법 C: Gate + Embedding (하이브리드)

blur/occlusion만 rule로 gate, 나머지는 embedding이 판단.
Landmark 없이도 "정면도" 판정이 가능.

### 적용

Best frame selection의 `frontality` 항목에 위 방법 중 하나를 적용.
초기에는 landmark 기반(Phase 1과 공유)으로 시작하고, 점진적으로 embedding 기반으로 전환.

## 8. Temporal Reranker

후보 구간(candidate windows)을 재정렬하는 작은 시퀀스 모델.

### 아키텍처 옵션

| 모델 | 구조 | 파라미터 | 특징 |
|------|------|---------|------|
| **GRU-lite** | 64-128 hidden, 1-2 layers | ~50K | 순서 의존성 학습, 가장 무난 |
| **TCN-lite** | depthwise separable conv, dilation 1/2/4 | ~30K | 병렬 추론, edge-friendly |
| Tiny Transformer | 2-4 head, 2 layers, d=64 | ~80K | self-attention, 더 큰 context |

### Temporal 처리 패턴

단순 GRU/TCN 외에 고려할 수 있는 아키텍처 패턴:

| 패턴 | 설명 | 적합 상황 |
|------|------|----------|
| **Hierarchical** | frame → clip → video 단계적 인코딩 | 긴 영상, 다중 스케일 |
| **Event-driven** | delta가 threshold 초과할 때만 처리 (sparse) | edge 배포, 전력 절약 |
| **Sliding window + memory** | 고정 window + 이전 context 요약 토큰 | 스트리밍 확장 시 |

초기 구현은 **GRU-lite**로 시작. 성능 한계가 보이면 패턴 변경.

### Input

각 candidate window의:
- clip embedding sequence (5-10 clips)
- cheap signal features (blur, exposure, motion, ...)
- rule score (feature로만 사용, teacher mimic 금지)

### Training: Pairwise Ranking Loss

```python
L = max(0, margin - score(positive) + score(negative))

# positive: 구매/저장/공유된 구간 (없으면 rule top-k를 weak positive로)
# negative: 같은 비디오의 다른 candidate 구간
```

**Rule score를 teacher로 mimic하는 것은 금지** — feature로만 입력.

### 라벨링 전략

하이라이트는 주관적이므로 절대 라벨이 아닌 **상대 비교** 기반 라벨링:

1. **Self-relative labeling**: 같은 비디오 내에서 "어느 구간이 더 나은가?" 쌍 비교
2. **Behavioral pseudo-labels**: 운영 데이터의 간접 신호 활용
   - 구매/저장/공유 → positive
   - 스킵 → negative
3. **Cross-video normalization**: 영상 간 점수 비교가 필요하면 Elo/TrueSkill 방식 적용
4. **Weak positive bootstrap**: 운영 데이터 전에는 rule top-k를 weak positive로 사용

### 배포 전략

1. **Shadow mode**: rule pick과 model pick 모두 저장, 비교만
2. **Canary**: 1-5% 트래픽에 model pick 적용
3. **Full**: 메트릭 확인 후 전환
4. 메트릭 하락 시 자동 rollback

## 9. Rule → Learned 전환 로드맵

highlight_rules(Phase 1)에서 highlight_vector(Phase 2)로의 점진적 전환:

| 단계 | 이름 | 내용 |
|------|------|------|
| **Phase 0** | Logging | 모든 signal/embedding을 기록만. 판정은 rule이 100% 담당 |
| **Phase 1** | Rule + Label 수집 | Rule이 후보 생성. 구매/스킵 behavioral label 축적 |
| **Phase 2** | Reranker Shadow | 학습된 reranker가 shadow mode로 동시 판정. 메트릭 비교 |
| **Phase 3** | Gated Rollout | Canary(1-5%) → 확대. 메트릭 하락 시 자동 rollback |
| **Phase 4** | Rules as Auxiliary | Reranker가 주, rule feature는 input의 일부로만 활용 |

**핵심 원칙**: 각 단계에서 이전 단계로 즉시 rollback 가능해야 한다.
Phase 0(logging)은 구현 첫 날부터 시작.

## 10. Best Frame Selection

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

## 11. Clip Indexing

```python
clip_len = 1.0    # seconds
clip_stride = 0.5  # seconds (50% overlap)

# clip 내 representative frame: center frame
# clip 내 pooling frames: 3-5 등간격 샘플
```

## 12. vpx 의존성

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

## 13. highlight_rules와의 비교

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

## 14. 구현 계획

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
