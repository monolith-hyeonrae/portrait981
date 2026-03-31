# Face State Embedding — 인물의 모든 것을 하나의 모델에

> identity, state, pose를 conditional하게 분리/결합할 수 있는 통합 임베딩.
> "누구인가" + "어떤 표정인가" + "어떤 포즈인가"를 하나의 모델이 담는다.

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

Face State Embedding = 얼굴 특화 × 통합 표현
  identity + expression + AU + pose + quality → 단일 벡터
  conditional하게 모드별 접근 가능
```

### 완전한 형태: Conditional Multi-Mode Embedding

기존 접근은 identity와 state를 별개 모델로 다룬다.
**하나의 모델이 conditional하게 모든 모드를 담을 수 있다.**

```
Raw Image → Shared Backbone → z (rich feature)
                                ↓
                z = [z_id | z_state | z_pose | z_quality]
                      ↓        ↓         ↓         ↓
               ID head  State head  Pose head  Quality head
               "누구"    "어떤 표정"  "어떤 각도"  "얼마나 선명"
```

쿼리 모드에 따라 다른 subspace를 활용:

```
"이 사람과 같은 사람 찾기"         → z_id로 검색
"이 표정과 비슷한 프레임 찾기"     → z_state로 검색
"이 포즈와 비슷한 프레임 찾기"     → z_pose로 검색
"이 사람의 이 표정과 비슷한"       → z_id + z_state 조합 검색
"이 사람의 정면 미소"             → z_id + z_state + z_pose 조합
```

### 수동 라벨 없이 학습 가능한 이유

14개 frozen 모델이 각 모드의 supervision을 이미 제공한다:

```
ID supervision:      face.detect ArcFace embedding → z_id pseudo-label
                     같은 비디오 내 = 같은 사람 (비디오 상수)
                     다른 비디오 간 = ArcFace similarity로 자동 매칭

State supervision:   face.expression 8D + face.au 12D → z_state pseudo-label
                     crowds consensus로 noise 필터링

Pose supervision:    head.pose yaw/pitch/roll → z_pose pseudo-label

Quality supervision: face.quality blur/exposure/contrast → z_quality pseudo-label
                     face.parse segmentation ratios

→ member_id 수동 라벨 불필요
→ ArcFace similarity = pseudo identity label
→ crowds가 모든 모드의 supervision 제공
```

### Identity vs State 분리의 핵심

```
ArcFace (기존):
  같은 사람, 웃는 표정  → embedding A
  같은 사람, 무표정    → embedding A (거의 동일)
  → 표정이 noise로 취급됨

Conditional Embedding (제안):
  같은 사람, 웃는 표정  → z_id=A, z_state=smile
  같은 사람, 무표정    → z_id=A, z_state=neutral
  → z_id는 같고 z_state가 다름 → 분리 성공

  다른 사람, 웃는 표정  → z_id=B, z_state=smile
  → z_state는 같고 z_id가 다름 → identity 무관하게 상태 검색 가능
```

### 비디오 상수의 활용 — ArcFace보다 강인한 identity supervision

탑승 비디오의 물리적 제약이 identity 학습의 핵심 이점:

```
ArcFace (프레임 단위, 불안정):
  Frame 100: face_id=A (confidence 0.95)
  Frame 101: face_id=B (confidence 0.72) ← 흔들림으로 ID switch
  Frame 102: face_id=A (confidence 0.88)
  Frame 103: face 미검출              ← 고개 돌림
  → 야외 환경에서 face_id가 계속 변함

비디오 상수 (100% 정확):
  "이 비디오의 이 좌석은 처음부터 끝까지 같은 사람"
  → 물리적 제약에서 오는 자연적 ground truth
  → ArcFace 불필요, 수동 라벨 불필요
```

**ArcFace가 실패하는 프레임이 곧 학습 데이터:**
```
ArcFace: "Frame 101은 다른 사람" (ID switch 오류)
비디오 상수: "같은 사람임을 보장"
→ Student가 "흔들려도, 고개 돌려도, 표정 바뀌어도 같은 사람"을 학습
→ ArcFace보다 야외 환경에 강인한 identity embedding 생산 가능
```

비디오 내 / 비디오 간 학습:
```
비디오 내 (물리적 보장, 자동):
  같은 비디오의 프레임 → z_id 가까워야 함 (contrastive positive)
  같은 비디오의 다른 표정 → z_state 달라야 함 (state diversity)

비디오 간 (member_id로 연결):
  같은 member_id의 다른 비디오 → z_id 가까워야 함 (cross-session positive)
  → 다른 날, 다른 조명, 다른 컨디션에서도 같은 identity
  → 조명/날짜/환경에 강인한 z_id 학습

  다른 member_id의 비디오 → z_id 달라야 함 (negative)
  같은 표정의 다른 사람 → z_state 가까워야 함 (identity-invariant state)
```

member_id는 videos.csv에 비디오당 한번만 기록. 모르면 비워두면 됨.
비디오 내 identity는 물리적 보장으로 자동, 비디오 간 identity만 member_id 필요.

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

- "Conditional Face Embedding: Identity, State, and Pose in One Model via Frozen Specialist Crowds"
- "Beyond Identity: Disentangled Face Representation from Annotation-Free Multi-Teacher Distillation"

### Contribution

1. **Conditional Multi-Mode Embedding**: identity/state/pose/quality를 하나의 모델에서 conditional하게 분리/결합. 모드별 subspace로 다양한 쿼리 지원.
2. **Annotation-free learning**: 14개 frozen specialist가 각 모드의 pseudo-label 제공 — ArcFace→identity, expression/AU→state, head.pose→pose. 수동 라벨 0건.
3. **비디오 상수 활용**: 탑승 비디오의 identity 상수를 contrastive learning에 활용 — 같은 비디오 = 같은 사람의 다양한 상태.
4. **Video-LLM integration**: face encoder를 Video-LLM에 장착, face event temporal grounding 개선.
5. **야외 강인성**: 실환경 데이터에서 학습된 모델의 범용 face analysis 성능.

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
  14 frozen 모델 → 43D signal → XGBoost
  = face state를 43D 수치로 이미 표현 중

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
| `how-visualbind.md` | 현재 판단 시스템 (XGBoost, 43D) |
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
