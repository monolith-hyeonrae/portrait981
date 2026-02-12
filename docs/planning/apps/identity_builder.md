# identity_builder — Identity-aware Generative Dataset Builder

> **momentscan Phase 3** 상세 설계. 마스터 플랜: [momentscan.md](momentscan.md)
>
> 라이드 비디오에서 인물별로 디퓨전 생성에 최적화된 이미지 세트를 구성하는 시스템.
> Anchor(정면 고품질) + Coverage(각도/표정 다양성) + Challenge(극단 조건)의 3세트 출력.
> 핵심은 **Face-ID와 General Vision 임베딩의 분리 사용**.

## 1. 목적

- Highlight detection이 아님 — **다양한 얼굴 이미지 수집**이 목표
- 디퓨전 모델(InstantID, IP-Adapter, PuLID)의 reference image로 사용
- 인물 ID 유지 + 상태 다양성 확보를 동시에 달성
- identity_memory의 입력 데이터 생성

## 2. 입출력

### Input

- 라이드 비디오 (~2분), 탑승자 1-2명, post-ride 배치

### Output (인물별)

```
output/{video_id}/identity_builder/
├── person_0/
│   ├── anchors/              # 3-5장, 정면 고품질
│   │   ├── anchor_001.jpg
│   │   └── anchor_001_meta.json
│   ├── coverage/             # 버킷별 best, 가변 수량
│   │   ├── cov_yaw30_pitch0_neutral.jpg
│   │   └── cov_yaw30_pitch0_neutral_meta.json
│   ├── challenge/            # 5-8장, 극단 조건
│   │   ├── chal_dark_001.jpg
│   │   └── chal_dark_001_meta.json
│   └── meta.json             # 인물 요약 (prototype, coverage 현황)
├── person_1/
│   └── ...
└── pair/                     # (optional) 2인 구도 anchor
    └── pair_001.jpg
```

### meta.json schema (인물별)

```json
{
  "person_id": 0,
  "track_score": 0.85,
  "prototype_type": "medoid",
  "anchor_count": 4,
  "coverage_buckets_filled": {
    "yaw": "6/7",
    "pitch": "4/5",
    "expression": "3/4"
  },
  "challenge_count": 6,
  "total_saved": 28,
  "stop_reason": "coverage_satisfied"
}
```

## 3. 핵심 설계 원칙

### 듀얼 임베딩 분리

| 임베딩 | 모델 | 용도 | 금지 용도 |
|--------|------|------|----------|
| **Face-ID** (`e_id`) | InsightFace ArcFace | 동일인 판정, ID stability | 표정/포즈 다양성 측정 |
| **General Vision** (`e_vis`) | DINOv2/SigLIP | 다양성, 노벨티 측정 | 동일인 판정 |

Face-ID로 다양성을 측정하면 identity invariance 때문에 표정/포즈 변화가 무시됨.
General Vision으로 ID를 판정하면 조명/배경 변화를 동일인 변화로 오인함.

### ComfyUI 연동 방식

- 임베딩 직접 주입 금지 — **이미지 경로** 전달
- InstantID / IP-Adapter / PuLID 노드에 reference image로 사용

## 4. 파이프라인

```
Video
  │ decode (visualbase)
  ▼
Face detect + tracking (vpx-face-detect)
  │
  ▼
Main Person Selection (1-2명)
  │ track_score = 0.4*avg_face_size + 0.3*center_proximity + 0.3*track_length
  ▼
Per-frame Processing:
  │
  ├── QualityGate check (strict / loose)
  ├── Crop stabilization (bbox EMA smoothing)
  ├── Dual embedding extraction
  │     e_id = ArcFace(aligned_face_crop)       # L2 norm
  │     e_vis_face = DINOv2(face_crop_224)      # L2 norm
  │     e_vis_body = DINOv2(body_crop_224)      # L2 norm (optional)
  ├── Bucket classification
  │     yaw_bin, pitch_bin, expression_bin, occlusion_bin
  ├── ID stability check
  │     stable(t) = cos(e_id, prototype)
  └── Novelty check
        novel(t) = min_j(1 - cos(e_vis, saved_j))
  │
  ▼
Selection Logic
  │ bucket empty → save
  │ bucket filled + higher quality → replace
  │ bucket filled + high novelty → add
  ▼
Set Construction
  │ Anchor (strict gate, frontal, top quality)
  │ Coverage (bucket-best)
  │ Challenge (extreme conditions + stable ID)
  ▼
Coverage Stop Check
  │ yaw >= 6/7, pitch >= 4/5, expression >= 3/4
  │ anchors >= 3, novelty plateau
  ▼
Export
```

## 5. 버킷 설계

### Yaw (7 bins)

```
[-45,-30]  [-30,-15]  [-15,-5]  [-5,5]  [5,15]  [15,30]  [30,45]
```

### Pitch (5 bins)

```
down  slight-down  neutral  slight-up  up
```

### Expression (4 bins, cheap proxy)

| Bin | 판정 기준 |
|-----|----------|
| neutral | mouth_open < 0.3, smile < 0.3, eyes_open |
| smile | smile_intensity >= 0.3 |
| mouth_open | mouth_open_ratio >= 0.3 |
| eyes_closed | eye_open_ratio < 0.3 |

vpx-face-expression 또는 landmark 기반 cheap proxy 사용. Emotion classifier는 사용하지 않음.

### Occlusion (3 bins)

```
none (ratio < 0.1)  partial (0.1-0.4)  heavy (> 0.4)
```

### Lighting (optional, 2-3 bins)

```
normal  dark (mean < 60)  overexposed (mean > 200)
```

### 버킷 키

```python
bucket_key = f"yaw{yaw_bin}_pitch{pitch_bin}_{expression_bin}"
```

## 6. ID Stability

### Face-ID 임베딩이 연속 프레임에서 흔들리는 원인

Face-ID(ArcFace)는 표정/각도에 불변을 목표로 하지만, 실전에서 연속 프레임 간 임베딩이 크게 흔들리는 경우가 있다.
원인은 **표정 변화가 아니라 품질/정렬 문제**인 경우가 대부분:

| 원인 | 설명 |
|------|------|
| Motion blur | detector crop이 미세하게 흔들림 |
| Alignment 실패 | 부분 가림(손/안전바)으로 5-point alignment 오류 → 임베딩 급변 |
| Extreme pose | roll/pitch가 모델의 안정 영역 밖 |
| Small/jittery bbox | 작은 얼굴 or bbox 불안정 → 품질 저하가 임베딩에 직접 반영 |

**대응 원칙**: "프레임 vs 프레임" 비교 대신 **"프레임 vs 프로토타입"** 비교.
프로토타입은 품질 좋은 프레임으로만 구성하여 outlier에 강하게 만든다.

### Short-term Tracker + Long-term ID 분리

| 계층 | 방법 | 역할 |
|------|------|------|
| **단기 (1-3초)** | bbox/velocity 기반 tracker | 임베딩이 잠깐 튀어도 연속성 유지 |
| **장기** | Face-ID prototype (medoid) | 사람 ID 안정적 유지 |

단기 tracker가 프레임 연속성을 담당하므로, 임베딩 spike가 발생해도 tracking이 버텨준다.

### Medoid Prototype

```python
# strict gate를 통과한 프레임의 face-ID 임베딩으로 medoid 계산
prototype = argmin_i sum_j distance(e_id_i, e_id_j)

# 프레임별 stability
stable(t) = cos(e_id(t), prototype)

# threshold
if stable(t) < tau_id:  # tau_id ≈ 0.3~0.4
    exclude from save candidates
```

Medoid가 mean보다 나은 이유: blur/alignment 실패 outlier에 강함.

### Prototype 업데이트 규칙

prototype 업데이트에는 **품질 게이트를 통과한 프레임만** 사용:
- blur 낮고, 가림 적고, alignment 양호한 프레임만 반영
- 품질 불량 프레임이 prototype을 오염시키는 것을 방지

### 후기 버전: Memory Bank (identity_memory)

```python
stable(t) = max_i cos(e_id(t), memory_bank[i].vec_id)
```

## 7. Novelty 계산

### 단일 임베딩

```python
# 이미 저장된 이미지들과 비교
novel(t) = min_j(1 - cos(e_vis_face(t), saved[j].e_vis_face))
```

### 듀얼 Novelty (고급)

```python
face_novelty = min_j(1 - cos(e_vis_face(t), saved[j].e_vis_face))   # DINOv2
body_novelty = min_j(1 - cos(e_vis_body(t), saved[j].e_vis_body))   # CLIP/SigLIP

final_novelty = max(face_novelty, body_novelty)
```

DINOv2: 표정/각도 미세 변화에 민감 (얼굴 다양성).
CLIP/SigLIP: 포즈/장면 semantic 변화에 민감 (상반신 다양성).

## 8. Set 구성 규칙

### Anchor Set (3-5장)

- Strict gate 통과
- Yaw near 0 (정면 선호)
- Blur 최소, occlusion 최소
- Quality 상위 5장

### Coverage Set (가변 수량)

- 버킷별 best quality 1장 (최대 2장)
- 수량은 고정하지 않음 — coverage 기준 충족 시 중단

### Challenge Set (5-8장)

- 극단 yaw/pitch OR 어두운 조명 OR heavy occlusion
- 조건: `stable(t) >= tau_id` + blur floor 통과
- **Constraint cues for diffusion**: 단순 극단 조건이 아니라 "이런 조건에서도 이 사람" 맥락을 제공
  - 디퓨전 모델의 robustness 향상
  - 네거티브 프롬프트 설계에 참고 (어떤 조건을 피해야 하는지)

## 9. 중단 조건 (Coverage-driven)

수량이 아닌 **커버리지 기준**으로 중단:

```python
stop_conditions = (
    yaw_coverage >= 6/7 and
    pitch_coverage >= 4/5 and
    expression_coverage >= 3/4 and
    anchor_count >= 3 and
    recent_novelty_mean < 0.08  # 최근 20프레임의 novelty 평균
)
```

`recent_novelty_mean < epsilon` = diminishing returns → 더 이상 새로운 이미지 없음.

## 10. 디퓨전 Conditioning 비율

```
Anchor : Coverage : Challenge = 20 : 70 : 10  (기본값)
```

| 상황 | 조정 |
|------|------|
| ID drift 잦음 (흔들림/가림) | 30 : 60 : 10 |
| 표정/각도 다양성 부족 | 15 : 75 : 10 |
| 어두운 조건 빈번 | 20 : 65 : 15 |

## 11. 이미지 메타 스키마 (필수)

```json
{
  "type": "anchor",
  "person_id": 0,
  "frame_idx": 1234,
  "timestamp_ms": 41133.3,
  "yaw_bin": "[-5,5]",
  "pitch_bin": "neutral",
  "expression_bin": "smile",
  "occlusion_bin": "none",
  "lighting_bin": "normal",
  "quality_score": 0.92,
  "novelty_score": 0.45,
  "stable_score": 0.88,
  "face_bbox": [120, 80, 320, 380],
  "crop_box": [100, 60, 340, 400],
  "embed_model_version": "dinov2-vits14-v1",
  "gate_version": "v1"
}
```

## 12. 2인 팩 (Optional)

비디오에 2명 탑승 시:
- 각각 독립 세트 생성
- 추가로 pair/ 디렉토리에 2인 구도 anchor 저장
  - 조건: 두 명 모두 quality gate 통과, 두 명 모두 stable, 상호 occlusion 최소
  - 메타에 상대 위치, 스케일 정보 포함

### Joint Diversity (2인 특화)

디퓨전에서 두 인물 관계/구도 가이드에 유용한 단서:
- **상대 위치**: 두 사람의 bbox 관계 (좌우, 거리)
- **동시 표정**: 함께 웃는/놀라는 동시성 프레임이 특히 가치 있음
- **상호 가림 최소**: 두 명의 bbox overlap이 낮은 프레임 우선

## 13. vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-face-detect | 얼굴 검출, 추적, face crop, head pose |
| vpx-face-expression | expression proxy (mouth_open, eye_open, smile) |
| vpx-body-pose | upper-body crop (body novelty용) |
| vpx-sdk | QualityGate, Observation |
| visualbase | 비디오 디코딩 |

### 신규 vpx 플러그인 필요

| 플러그인 | 역할 | 비고 |
|---------|------|------|
| **vpx-vision-embed** | DINOv2/SigLIP 임베딩 추출 | highlight_vector와 공유 |

Face-ID 임베딩은 vpx-face-detect의 InsightFace recognition model에서 추출.

## 14. Embedding Stability 트러블슈팅 체크리스트

Face-ID 임베딩이 불안정할 때 확인할 항목:

1. **Alignment 실패 프레임 제외**: roll이 큰 프레임 → save/update 후보에서 제외
2. **bbox jitter 감소**: tracker 기반 smoothing (bbox EMA, alpha=0.3~0.5)
3. **Quality-gated prototype update**: blur/occlusion 불량 프레임은 prototype 업데이트 금지
4. **Medoid/trimmed mean 사용**: 단순 mean 대신 outlier-robust 대표값
5. **"Frame vs prototype" 비교**: 연속 프레임 간 비교 금지 → 항상 prototype과 비교

이 조합으로 대부분의 임베딩 불안정 문제를 해결할 수 있다.

## 15. 구현 계획

### Phase 1: 추출 파이프라인

- [ ] 비디오 디코딩 + face tracking
- [ ] main person selection (track_score)
- [ ] QualityGate (strict/loose) 적용
- [ ] bbox EMA smoothing (crop stabilization)
- [ ] Face-ID embedding 추출 (ArcFace)
- [ ] medoid prototype 계산

### Phase 2: 버킷 + Novelty

- [ ] 버킷 분류 (yaw/pitch/expression/occlusion)
- [ ] General vision embedding 추출 (vpx-vision-embed)
- [ ] novelty 계산
- [ ] selection logic (empty → save, replace, add)

### Phase 3: Set 구성 + Export

- [ ] Anchor / Coverage / Challenge 세트 분리
- [ ] coverage stop condition 구현
- [ ] meta.json 생성
- [ ] 이미지 + 메타 export

### Phase 4: 고도화

- [ ] 듀얼 novelty (DINOv2 face + CLIP body)
- [ ] identity_memory 연동 (prototype → memory bank)
- [ ] 2인 팩 지원
- [ ] highlight window sampling priority 연동
