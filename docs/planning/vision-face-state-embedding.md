# Face State Embedding — 인물 상태 통합 임베딩 모델

> 기존 얼굴 임베딩(identity)과 다른 새로운 공간:
> "이 사람이 **누구**인가"가 아니라 "이 사람이 **지금 어떤 상태**인가"를 표현.

---

## 문제: 기존 임베딩의 빈 자리

### Identity Embedding (ArcFace, AdaFace)

같은 사람이면 같은 벡터. 표정이 바뀌어도, 포즈가 달라져도 임베딩 거리가 거의 동일.
**"누구인가"에 최적화 → "어떤 상태인가"는 무시.**

### Scene Embedding (CLIP, DINO, SigLIP)

장면 전체의 구조, 상황, 분위기를 잘 표현.
"골프카트를 타고 도로를 달린다"는 잘 보지만
"Duchenne smile이다", "3/4 측면이다"는 구분 못함.
**"무슨 장면인가"에 최적화 → 얼굴 미세 변화에 둔감.**

### 단일 task 모델 (HSEmotion, LibreFace, 6DRepNet)

각각 emotion(8D), AU(12D), pose(3D)만 출력.
풍부하지만 **분리되어 있고, 통합된 표현이 없음.**

### 비어있는 곳

```
                Identity    State
              ┌───────────┬───────────┐
얼굴 특화     │ ArcFace   │ (비어있음) │
              │ AdaFace   │           │
              ├───────────┼───────────┤
범용          │           │ CLIP/DINO │
              │           │ (둔감)    │
              └───────────┴───────────┘

Face State Embedding = 얼굴 특화 × 상태 표현
  expression + AU + pose + quality + gaze → 단일 벡터
  "이 사람이 지금 어떤 상태인가"
```

---

## 아이디어: Frozen Specialist Crowds → Face State Embedding

### 핵심

14개 frozen face analysis 모델이 각각 부분적 관측을 한다.
이들의 출력을 **crowds consensus로 통합하여 단일 모델을 학습**하면,
어떤 개별 모델보다 풍부한 face state 표현을 가진 임베딩이 만들어진다.

```
14 Frozen Specialists (각각 부분적):
  face.detect    → confidence, bbox
  face.au        → AU12D (근육 수축)
  face.expression → emotion 8D (감정)
  head.pose      → yaw/pitch/roll 3D
  face.quality   → blur, exposure, contrast
  face.parse     → segmentation ratios
  portrait.score → CLIP aesthetic axes
  ...

     ↓ crowds consensus (annotation-free)

Face State Embedding Model:
  Raw face image → single vector (e.g. 256D)
  이 벡터가 위 14개 모델의 지식을 통합
  = expression + AU + pose + quality가 하나의 공간에
```

### 왜 가능한가

- **데이터**: 매일 10M+ 프레임, 야외/차량 환경의 다양한 얼굴
- **Supervision**: 14개 frozen 모델의 crowds consensus가 pseudo-label 제공
- **선행 사례**: Florence-2 (다양한 모델 출력 → unified model), EmoNet-Face (SigLIP2 + 40D emotion head)

### Identity Embedding과의 차이

```
ArcFace:
  같은 사람, 웃는 표정  → embedding A
  같은 사람, 무표정    → embedding A (거의 동일)
  → 표정 변화가 노이즈로 취급됨

Face State Embedding:
  같은 사람, 웃는 표정  → embedding A
  같은 사람, 무표정    → embedding B (다름!)
  같은 사람, 측면 포즈  → embedding C (다름!)
  → 표정/포즈 변화가 핵심 정보
```

---

## 활용: Video-LLM에 장착

### 현재 Video-LLM의 한계

```
Time-R1 / Qwen2.5-VL:
  Video → SigLIP vision encoder → visual tokens → LLM → 응답

SigLIP이 보는 것:   "골프카트, 도로, 건물, 사람" (장면)
SigLIP이 못 보는 것: "Duchenne smile, 인상 찡그림, 3/4 측면" (인물 상태)
```

### 개선된 구조

```
Video → SigLIP (장면 encoder)          → scene tokens  ─┐
      → Face State Embedding (인물 encoder) → face tokens ──┤→ LLM → 응답
                                                          ─┘

LLM이 두 관점을 동시에 봄:
  "94초에 여성이 Duchenne smile로 카메라를 정면으로 바라보며,
   조명이 자연스럽고 얼굴이 선명한 순간 — portrait에 최적"
```

### Temporal Grounding 개선

Time-R1 스타일 RL fine-tuning에 두 가지 reward:
- **temporal IoU**: 모델이 찾은 구간 vs 정답 구간
- **crowds consensus**: 14개 specialist 합의도

이렇게 하면 "portrait worthy 순간"을 정확한 timestamp로 찾을 수 있음.

---

## 야외 강인 얼굴 인식

### 범용 가치

981파크의 데이터 특성:
- 야외/차량 환경 (실내 학습 모델의 약점)
- 조명 변화 극심 (역광, 그림자, 시간대별)
- 진동, 움직임
- 다양한 고객 (연령, 성별, 인종)
- 매일 3000건 × 365일 축적

이 환경에서 학습된 face state embedding은
**야외 환경에 강인한 범용 모델**로도 가치가 있다.

```
실내 학습 모델 (ArcFace, HSEmotion 등):
  → 실내/정면/균일 조명 데이터로 학습
  → 야외 성능 저하 (domain gap)

우리 모델:
  → 야외/차량 환경 매일 10M+ 프레임으로 학습
  → 자연스러운 조명 변화, 다양한 얼굴
  → 야외 face analysis benchmark에서 경쟁력
```

---

## 학술적 포지셔닝

### 논문 제목 후보

- "Face State Embedding: Annotation-Free Multi-Specialist Distillation for Unified Facial Analysis"
- "Beyond Identity: Learning Face State Representations from Frozen Specialist Crowds"

### Contribution

1. **Face State Embedding**: identity가 아닌 state 공간 정의 — expression, AU, pose, quality 통합
2. **Annotation-free learning**: 14개 frozen specialist의 crowds consensus로 학습, 수동 라벨 불필요
3. **Video-LLM integration**: face encoder를 Video-LLM에 장착, face event temporal grounding 개선
4. **야외 강인성**: 실환경 데이터에서 학습된 모델의 범용 face analysis 성능

### 필요한 실증

| 실험 | 목적 |
|------|------|
| Face retrieval: state embedding vs CLIP/DINO | 표정/포즈 변화에 대한 변별력 비교 |
| Face clustering: state별 그룹핑 품질 | 같은 상태끼리 모이는지 |
| Video-LLM + face encoder: face event QA | temporal grounding 정확도 개선 |
| 공개 벤치마크 (DISFA, BP4D, AffectNet) | 범용 성능 검증 |
| Teacher ablation: 어떤 조합이 최적인지 | N개 teacher subset 실험 |

### 대상 학회

| 수준 | 학회 | 필요 완성도 |
|------|------|-----------|
| Embedding + 내부 데이터 | FG / ECCV Workshop | embedding 학습 + face retrieval |
| + 공개 벤치마크 | CVPR / ECCV Main | + BP4D/DISFA 평가 |
| + Video-LLM integration | NeurIPS / ICML | + temporal grounding 실증 |

---

## 실현 경로

### visualgrow를 통한 단계적 실현

```
현재 (Level 1):
  14 frozen 모델 → 45D signal → XGBoost
  = face state를 45D 수치로 이미 표현 중

Level 3 (Vision Student):
  Raw face image → MobileNetV3 → feature → bucket prediction
  + 14개 teacher 출력 재구성 (soft loss)
  → 이 feature가 face state embedding의 초기 버전

Level 3.5 (Embedding 추출):
  학습된 Vision Student의 중간 feature (576D 또는 줄인 256D)
  = Face State Embedding
  → face retrieval, clustering에 활용 가능
  → 공개 벤치마크 평가

Level 4 (Video-LLM 장착):
  Face State Embedding을 Qwen2.5-VL의 추가 encoder로 장착
  → RL fine-tuning (crowds consensus as reward)
  → portrait moment temporal grounding
```

### 핵심 의존성

```
visualgrow Level 3 (Vision Student) 성공이 전제 조건
→ Vision Student의 feature quality가 embedding quality를 결정
→ Level 1-2 (XGBoost + crowds pseudo-label)가 충분히 작동해야
→ pseudo-label 품질이 Vision Student 학습의 천장
```

---

## Qwen2.5-VL 테스트 결과 (2026-03-17)

8GB GPU에서 3B INT4로 테스트:

```
장면 이해:    ✅ "여성이 골프카트를 운전하며 미소, 바람에 머리 날림"
표정 변화:    ✅ "중립 → 가끔 미소 → 카메라 보며 미소"
portrait 추천: ✅ "미소 + 자연 배경이 portrait에 좋다"
timestamp:    ⚠️ 부정확 (pretrained 한계, RL fine-tuning 필요)
```

pretrained VLM은 장면 이해에 탁월하지만 정밀한 temporal grounding은 부족.
Face State Embedding이 추가되면 인물 분석 정밀도가 향상될 것으로 기대.

---

## 연관 문서

| 문서 | 관계 |
|------|------|
| `why-visualbind.md` | 이론적 기반 (crowds, 축별 독립 근사) |
| `how-visualbind.md` | 현재 판단 시스템 (XGBoost, 45D) |
| `how-visualgrow.md` | 성장 시스템 (Level 3에서 embedding 생산) |
| `projected-crowds.md` | 상세 이론 |

## 참고 연구

| 연구 | 관련성 |
|------|--------|
| EmoNet-Face (NeurIPS 2025) | SigLIP2 + 40D emotion, face-specialized embedding 사례 |
| Time-R1 (Xiaomi) | RL temporal grounding, crowds consensus as reward 적용 가능 |
| Video-R1 (Tsinghua) | T-GRPO, video reasoning + RL |
| DANCE (NeurIPS 2025 Spotlight) | 독립 공간 분해 → 결합, 구조적 대칭 |
| Florence-2 (CVPR 2024) | multi-model output → unified model, 원리적 근거 |
| Head Pursuit (NeurIPS 2025 Spotlight) | VLM attention head specialization, 이론 참고 |
