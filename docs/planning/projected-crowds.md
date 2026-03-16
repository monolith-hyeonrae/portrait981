# Projected Crowds — 이종 Observer 사영을 통한 Weak Supervision 프레임워크

> 2026-03-16, visualbind의 학술적 novelty를 강화하기 위한 이론적 프레이밍.
> `why-visualbind.md`, `visualbind-reframe.md`, `visualbind-expert-review-unified.md`의 논의를 기반으로,
> **Projected Crowds**라는 형식적 프레임워크로 발전시킨 문서.

---

# Part 1: 기존 연구 대비 Novelty 진단

## 구조적 동치 문제

14명의 전문가 리뷰(Round 1-4)에서 일관되게 지적된 핵심:

```
CrossCheck = Snorkel의 Labeling Function
AgreementEngine = Snorkel의 Label Model
Threshold 적용된 모듈 = Dawid-Skene의 Annotator
```

개별 메커니즘이 모두 기존 연구에 존재한다면, "새로운 조합"만으로 top-tier contribution이 되기 어렵다.

## 그러나 — 정말 동치인가?

"구조적으로 비슷하다"와 "동치다"는 다르다. 기존 프레임워크가 **다루지 않는 설정**이 존재한다.

### Snorkel과의 진짜 차이

Snorkel의 Labeling Function은 사람이 설계한 규칙이다:

```python
# Snorkel LF: 텍스트에서 "married"가 있으면 배우자 관계
def lf_married(x):
    return SPOUSE if "married" in x.text else ABSTAIN
```

VisualBind의 "LF"는 수백만 파라미터를 가진 독립 학습된 ML 모델의 출력이다:

```python
# VisualBind: HSEmotion이 학습한 표정 분포 → threshold → vote
vote = expression_model(face_crop)["happy"] > 0.6
```

이 차이가 만드는 구조적 결과:

| | Snorkel LF | VisualBind Observer |
|---|---|---|
| 출력 | binary/discrete | **연속 다차원** (threshold 전) |
| 정보량 | ~1 bit/LF | **수십 bit/observer** (24D continuous) |
| 오류 구조 | 규칙 설계 실수 (random-like) | **학습 경계 기반 체계적 bias** |
| 상관 | LF 간 독립 가정 가능 | observer 간 **구조적 의존성** (shared input) |
| 개수 | 수십~수백 개 | 소수 (5-14개), 각각 고비용 |

**Snorkel은 "많은 약한 신호"를 다루고, VisualBind는 "소수의 강하지만 편향된 신호"를 다룬다.**
이것은 수학적으로 다른 regime이다.

### Learning from Crowds와의 진짜 차이

Dawid-Skene의 annotator는 같은 질문에 같은 형식으로 답한다:

```
"이 이미지가 고양이인가?" → Annotator 1: Yes, Annotator 2: No, Annotator 3: Yes
```

VisualBind의 observer는 **다른 질문에 다른 형식으로 답하는데, threshold를 통해 같은 질문으로 사영된다**:

```
AU 모델: "AU12 intensity = 3.2" (근육 수축량, 0-5 스케일)
Expression 모델: "happy = 0.8" (감정 확률, 0-1 스케일)
Pose 모델: "yaw = 12°" (각도, ±90° 스케일)
    ↓ threshold projection
모두: "이 프레임이 warm_smile 버킷에 적합한가?" → ✓, ✓, ✓
```

**Threshold는 단순한 이진화가 아니라, 이종 공간을 공통 결정 공간으로 사영하는 operator이다.**
이 사영 과정에서 정보가 손실되는데, dual-mode loss가 그 손실을 복구한다.

이것은 crowds 문헌에서 다뤄지지 않은 설정이다. Crowds는 annotator가 이미 같은 공간에 있다고 가정한다.

## 기존 연구 지형의 빈 자리

```
          이산 출력          연속 출력
        ┌─────────────┬─────────────┐
같은    │ Dawid-Skene  │ (비어있음)   │
task    │ Snorkel      │             │
        ├─────────────┼─────────────┤
다른    │ (비어있음)    │ Multi-modal │
task    │              │ fusion      │
        └─────────────┴─────────────┘

Projected Crowds는 사영(τ)으로 네 사분면을 연결:
  - 다른 task의 연속 출력 → 사영 → 같은 task의 이산 투표
  - 동시에 연속 출력도 활용
```

**사영이 발생하는 순간과 그 정보 손실을 명시적으로 다루는 프레임워크가 없다.** 이것이 빈 자리이다.

---

# Part 2: Projected Crowds — 핵심 개념

## 사영의 본질

현재 시스템에서 이미 일어나고 있는 일을 형식화한다.

### Observer와 사영

```
N개의 observer: f₁, f₂, ..., f_N
각 observer의 출력 공간: Y₁, Y₂, ..., Y_N (이종, 다차원)
공통 결정 공간: {0, 1}  (버킷 적합 여부)
사영 연산자: τᵢ: Yᵢ → {0, 1}  (threshold operator)

입력 x에 대해:
  연속 출력:  yᵢ = fᵢ(x) ∈ Yᵢ
  사영된 투표: vᵢ = τᵢ(yᵢ) ∈ {0, 1}
```

### 사영에서 일어나는 두 번의 정보 손실

```
Observer 출력: y_expr = [happy=0.82, neutral=0.10, surprise=0.05, ...] ∈ ℝ⁸

Step 1 — 축 선택: "happy" 차원만 취함
         ℝ⁸ → ℝ¹  (8차원에서 1차원으로, 7차원 버림)

Step 2 — 절단: happy > 0.6?
         ℝ¹ → {0,1}  (연속에서 이산으로)
```

Step 1에서 축을 하나만 고르면서 나머지를 버리고,
Step 2에서 연속값을 이진화하면서 "얼마나"의 정보를 버린다.

이것을 observer마다 반복한 뒤 AND로 결합하면,
24차원 joint space에서 **축 정렬 직사각형(axis-aligned box)**을 정의하는 것과 같다.

### Threshold = 이종 공간 간의 통약 가능성을 만드는 연산

```
τ₁: ℝ⁸  → {0,1}    (expression space → binary vote)
τ₂: ℝ¹² → {0,1}    (AU space → binary vote)
τ₃: ℝ³  → {0,1}    (pose space → binary vote)
τ₄: ℝ¹  → {0,1}    (confidence space → binary vote)
```

사영된 후에야 비로소 모든 observer가 **같은 공간**에 있게 되고,
비교와 합의가 가능해진다.

### 기존 프레임워크의 사영에 대한 태도

| 프레임워크 | 사영에 대한 태도 |
|-----------|----------------|
| Learning from Crowds | 사영이 이미 끝났다고 가정 (annotator는 이미 같은 공간) |
| Snorkel | LF 설계 시 사람이 사영을 내장 (규칙 자체가 binary) |
| Multi-modal fusion | 사영 없이 연속 공간을 직접 결합 (attention, concatenation) |
| Multi-teacher KD | 사영 불필요 (teacher가 같은 task, 같은 출력 공간) |

---

# Part 3: 세 가지 학습 모드

## Mode A — Crowds-only (사영 후 정보만 사용)

```
입력: {v₁, v₂, ..., v_N}  (binary votes)
방법: Dawid-Skene / majority vote
정보량: N bits (최대)
```

사영의 정보 손실이 그대로 학습에 반영된다.

```
happy = 0.82 → τ(0.82, threshold=0.6) → 1
happy = 0.61 → τ(0.61, threshold=0.6) → 1   ← 0.82와 구분 불가
happy = 0.59 → τ(0.59, threshold=0.6) → 0   ← 0.61과 극적으로 다른 취급

사영의 정보 손실: H(Y) - H(V) = H(Y|V)
8차원 연속 분포 → 1 bit. 손실이 막대함.
```

## Mode B — Fusion-only (사영 전 정보만 사용)

```
입력: {y₁, y₂, ..., y_N}  (연속 출력, concatenated)
방법: MLP / attention으로 직접 학습
정보량: Σ dim(Yᵢ) × precision (최대)
문제: ground truth label이 없음 → self-supervised만 가능
```

정보는 풍부하지만 **방향이 없다**. "이 프레임이 좋은 초상화인가?"에 대한 답이 없다.

## Mode C — Projected Crowds (사영 전 + 후 동시 사용)

```
입력: {y₁, ..., y_N} + {v₁, ..., v_N}
방법: dual-mode loss
  Hard: BCE(student(y), crowds_consensus(v))  ← 방향성 (crowds)
  Soft: MSE(teacher_head(z), yᵢ)             ← 정보 복구 (fusion)
정보량: Mode A + Mode B의 결합
```

### Mode C가 두 문제를 동시에 해결

```
사영된 vote → crowds 합의 → pseudo-label (방향성 제공)
연속 출력 → soft reconstruction → 사영 전 정보 복구

두 경로가 shared encoder의 z를 통해 만남.
z는 crowds가 가리키는 방향으로 이동하되,
연속 정보의 풍부함을 유지함.
```

### 이론적 질문

**Q1: 사영의 정보 손실은 얼마인가?**
```
I_loss = I(Y; T) - I(V; T)
Y = 연속 출력, V = 사영된 vote, T = true label
```

**Q2: Dual-mode는 손실을 얼마나 복구하는가?**
```
Mode A 성능: Perf(V)        — crowds-only
Mode C 성능: Perf(V, Y)     — projected crowds
복구량:      Perf(V, Y) - Perf(V)
```

Exp 1 (hard-only) vs Exp 3 (hybrid)가 정확히 이 비교이다.

**Q3: Observer 수가 증가하면 사영 손실이 보상되는가?**
```
N이 충분히 크면, 개별 사영의 정보 손실을
다수의 독립적 vote가 보상할 수 있는가?
→ N-observer scaling law의 정보 이론적 버전
```

---

# Part 4: 정보 흐름의 기하학적 해석

## 사영의 기하학

각 observer의 출력 공간에서 threshold는 초평면을 정의한다:

```
Observer 1 (Expression, ℝ⁸):
  τ₁은 happy 축에서 0.6에 수직인 초평면
  → 공간을 "적합" 반공간과 "부적합" 반공간으로 분할

Observer 2 (AU, ℝ¹²):
  τ₂는 AU12 축에서 2.0에 수직인 초평면

Observer 3 (Pose, ℝ³):
  τ₃는 |yaw| < 30을 만족하는 slab (두 초평면 사이 영역)
```

Joint space에서 이들의 교차:

```
τ₁의 적합 영역 ∩ τ₂의 적합 영역 ∩ τ₃의 적합 영역
= axis-aligned rectangular region (축 정렬 직사각형)
```

**이것이 catalog_scoring의 기하학적 본질이다** — 축 정렬 직사각형 경계.

Student MLP는 이 직사각형을 임의의 비선형 경계로 대체한다:

```
축 정렬 직사각형:          MLP 결정 경계:
┌──────────┐              ╭──────────╮
│          │              │    ╭─╮   │
│  ✓ 영역  │     →        │ ✓  │ ✗│  │
│          │              │    ╰─╯   │
└──────────┘              ╰──────────╯

"웃을 때 고개가 약간 기울어야 자연스럽다"
= expression × pose의 비선형 상호작용
= 축 정렬로 표현 불가, MLP로 표현 가능
```

## 논문의 핵심 그림 (Figure 1 후보)

```
          Observer Spaces              Common Space
          (Heterogeneous)              (Projected)

    Y₁ (ℝ⁸)  ──── τ₁ ────→  v₁ ∈ {0,1} ──┐
         │                                   │
    Y₂ (ℝ¹²) ──── τ₂ ────→  v₂ ∈ {0,1} ──┤→ Crowds → pseudo-label
         │                                   │        (Hard loss)
    Y₃ (ℝ³)  ──── τ₃ ────→  v₃ ∈ {0,1} ──┘
         │
         └──── concatenate ──→ [y₁;y₂;y₃] (24D)
                                    │
                              Student Encoder
                                    │
                                    z ←── Hard loss (crowds 방향)
                                    │ ←── Soft loss (정보 복구)
                                    │ ←── VICReg (붕괴 방지)
                                    ↓
                              Bucket 판단

    ═══════════════════════════════════════════════
    왼쪽 경로: Fusion (연속, 풍부, 방향 없음)
    위쪽 경로: Crowds (이산, 손실, 방향 있음)
    Student:   두 경로의 교차점
```

Fusion과 Crowds가 사영을 매개로 연결되고, Student가 그 교차점에서 학습한다.

---

# Part 5: 사영 축의 정의 — 방향과 스케일

## 사영의 구성 요소

하나의 버킷에 대한 사영의 완전한 정의:

```
Projection π for bucket k:
  (1) 축 선택:    어떤 observer의 어떤 차원을 사용할 것인가
  (2) 정규화:    각 축의 스케일을 어떻게 통일할 것인가
  (3) 방향 정의:  통일된 공간에서 어떤 방향이 "이 버킷에 적합함"인가
  (4) 절단점:    그 방향에서 어디를 경계로 삼을 것인가

π_k(y) = 𝟙[wₖ · normalize(y) > θₖ]
```

## 사영 방법의 스펙트럼

### 방법 1: 수동 축 선택 + 수동 threshold (현재)

```python
projection = {
    "face.expression": {"signal": "happy", "threshold": 0.6},
    "face.au":         {"signal": "AU12",  "threshold": 2.0},
    "head.pose":       {"signal": "yaw",   "range": (-15, 15)},
}
```

- 장점: 해석 가능, 디버깅 쉬움, 도메인 전문가 직관
- 단점: 축 정렬만 가능, 비선형 관계 표현 불가

### 방법 2: 정규화된 축 + 가중 결합 (catalog_scoring)

```python
normalize(AU12, range=(0,5))     → [0, 1]
normalize(happy, range=(0,1))    → [0, 1]
normalize(yaw, range=(-90,90))   → [0, 1]

weight_i = fisher_ratio(signal_i)
score = Σ wᵢ × (xᵢ - centroidᵢ)²
```

- 장점: 스케일 자동 보정, 데이터 기반
- 단점: 여전히 축 정렬, 비선형 불가, centroid + 참조 이미지 필요

### 방법 3: 임의 방향의 선형 사영 (빈 자리)

```
기존: 각 축을 독립적으로 절단
     happy > 0.6?  AU12 > 2.0?  (각각 독립)

제안: joint space에서 임의 방향을 정의
     0.5 × happy + 0.3 × norm(AU12) + 0.2 × norm(AU6) > θ

     방향 벡터 w가 "이 버킷에서 무엇이 중요한가" 정의
     절단점 θ가 "얼마나 강해야 하는가" 정의
```

기하학적으로 축 정렬이 아닌 **기울어진 경계**:

```
    AU12
     ↑
   5 │   ╱
     │  ╱  ✓ 영역
     │ ╱
   2 │╱............  ← 기울어진 경계: w·x > θ
     │ ✗ 영역
   0 └────────────→ happy
     0             1.0
```

방향 벡터 w를 정하는 방법:

```
(a) 수동: 도메인 전문가가 가중치 지정
    w = [0.5, 0.3, 0.2, 0, 0, ...]

(b) Fisher LDA: 데이터에서 최적 방향 계산
    w = Σ_w⁻¹(μ₁ - μ₂)
    → catalog_scoring의 Fisher ratio가 이것의 축 정렬 근사

(c) 소량 예시에서 학습:
    50건의 예시 → logistic regression → w, θ 학습
```

### 방법 4: 비선형 사영 (Student MLP)

```
y (24D) → MLP → z (16D) → Bucket Head → score ∈ [0,1] → threshold → vote
```

임의의 비선형 경계를 학습한다:

```
    AU12
     ↑
   5 │   ╭───╮
     │  ╱  ✓  ╲
     │ │       │  ← 비선형 경계
   2 │  ╲     ╱
     │   ╰───╯
   0 └────────────→ happy
     0             1.0
```

### 4가지 방법의 관계

```
표현력 ─────────────────────────────────────────→
해석력 ←─────────────────────────────────────────

방법 1          방법 2          방법 3          방법 4
축 정렬         축 정렬 +       임의 방향        비선형
threshold       스케일 보정      선형 사영        MLP
                Fisher          w·x > θ         f(x) > θ
│               │               │               │
│  현재 수동     │ catalog_      │  빈 자리       │ Student
│  threshold    │ scoring       │               │ MLP
```

방법 3이 빈 자리이다. 해석 가능하면서 축 정렬보다 표현력이 좋은 중간 지점.

방법 3(선형 사영)이 crowds의 pseudo-label을 만들고,
방법 4(Student MLP)가 그 pseudo-label을 학습하면,
**MLP는 방법 3이 정의한 경계를 비선형으로 정제(refine)하는 것**이다.
사영이 초기 방향을 잡아주고, Student가 그 방향을 따라가면서 더 정교한 경계를 학습하는 구조.

## 스케일 문제: 이종 축의 통약

방향을 정의하려면 먼저 스케일이 통일되어야 한다.

```
원시 스케일:
  AU:         0-5 (FACS intensity, 지각적 로그 스케일)
  Expression: 0-1 (softmax 확률)
  Pose:       -90°~+90°
  Confidence: 0-1
```

정규화 전략별 효과:

| 전략 | 방법 | 장단점 |
|------|------|--------|
| MinMax | `(x - lo) / (hi - lo)` | 범위 설정 자의적 |
| Percentile | 데이터 분포 기반 rank | 적응적, 분포 변화 시 의미 변화 |
| Sigmoid | `σ((x - center) × scale)` | 관심 구간에 해상도 집중, 도메인 지식 필요 |
| LayerNorm | Student MLP 첫 레이어 | 자동, 해석 어려움 |

**정규화 = 각 축의 "의미 있는 단위"를 정의하는 것.**
AU12가 2→3으로 변하는 것과 happy가 0.5→0.6으로 변하는 것이
"같은 정도의 변화"라고 정의하는 것이 정규화이다.

---

# Part 6: 사영이 곧 개념 — Intentional Concept Drift

## 핵심 통찰

Observer는 세계를 있는 그대로 관측하는 감각 기관이다.
같은 프레임에서 같은 observer는 항상 같은 값을 출력한다.
**달라지는 것은 사영이다.**

```
Observer = 감각 (고정, what IS)
Projection = 의도 (가변, what I WANT)
Student = 감각 + 의도를 학습한 판단 주체
```

### 사영이 개념을 정의한다

같은 observer, 같은 출력, 같은 프레임인데, 사영만 바꾸면 전혀 다른 판단이 나온다:

```
사영 A — "warm_smile":
  τ_expr: happy > 0.6         → ✓
  τ_au:   AU12 > 2.0          → ✓
  τ_pose: |yaw| < 15          → ✓
  → "정면 따뜻한 미소를 원한다"

사영 B — "dramatic_profile":
  τ_expr: (happy > 0.7) OR (surprise > 0.5)
  τ_au:   AU6 > 2.0
  τ_pose: |yaw| > 30
  → "측면 극적인 표정을 원한다"

사영 C — "serene_close":
  τ_expr: neutral > 0.5
  τ_au:   max(AU) < 1.5       (근육 이완)
  τ_pose: |pitch| < 10
  → "차분한 정면을 원한다"
```

### Concept Drift의 재정의

```
기존 ML에서 concept drift:
  "환경이 변해서 모델이 맞지 않게 된다" → 방어해야 할 문제

Projected Crowds에서 concept drift:
  "운용자가 사영을 바꿔서 개념을 재정의한다" → 시스템의 핵심 기능
```

**Concept drift는 사영 공간의 변화이다.**
Observer 공간은 고정이고, 사영이 바뀔 때 개념이 바뀐다.

### 구체적 시나리오

```
시즌 1 (겨울):
  Π₁ = {warm_smile, cool_gaze, lateral}
  → Student가 이 3개 사영에 대해 학습

디자인팀: "봄 시즌은 눈감은 장면이 필요해"

시즌 2 (봄):
  Π₂ = {warm_smile, cool_gaze, lateral, eyes_closed}
  → 새 사영 τ_eyes_closed 추가
  → 기존 데이터에 즉시 적용 가능 (observer 출력은 이미 있음)
  → 50건 pseudo-label → 새 Bucket Head만 학습

디자인팀: "따뜻한 미소 대신 밝은 웃음으로 바꿔줘"

시즌 3 (여름):
  warm_smile의 τ_expr를 happy > 0.6 → happy > 0.8로 변경
  warm_smile의 τ_au를 AU12 > 2.0 → AU12 > 3.0 AND AU6 > 2.0으로 변경
  → 같은 이름의 버킷이지만 개념이 바뀜
  → pseudo-label이 자동으로 갱신됨
  → Student가 새 개념에 재학습
```

## 3층 분리 아키텍처

```
┌─────────────────────────────────────────────┐
│  Observer Layer (고정, infrastructure)        │
│  f₁: X → Y₁, f₂: X → Y₂, ..., f_N: X → Y_N │
│  "세계가 어떤가"                              │
│  변경 주기: 거의 없음 (모델 교체 시에만)        │
└──────────────────┬──────────────────────────┘
                   │ 연속 출력 {y₁, ..., y_N}
                   │
┌──────────────────▼──────────────────────────┐
│  Projection Layer (가변, configuration)       │
│  Π = {τ₁, τ₂, ..., τ_N} per bucket          │
│  "무엇을 원하는가"                            │
│  변경 주기: 시즌별, 요구사항 변경 시            │
│  변경 비용: threshold 숫자 몇 개 수정          │
└──────────────────┬──────────────────────────┘
                   │ binary votes + 연속 출력
                   │
┌──────────────────▼──────────────────────────┐
│  Student Layer (학습, adaptation)             │
│  Encoder → z → Bucket Heads                  │
│  "감각과 의도를 결합한 판단"                   │
│  변경 주기: 자동 (데이터 축적에 따라)           │
└─────────────────────────────────────────────┘
```

변경 비용 비교:

| 변경 사항 | 영향 범위 | 비용 |
|-----------|----------|------|
| 새 observer 추가 | 전체 재학습 | 높음 |
| **사영 변경 (threshold 수정)** | **pseudo-label 재생성 + 재학습** | **낮음** |
| **새 사영 추가 (새 버킷)** | **새 Head만 학습** | **매우 낮음** |
| 데이터 축적 | Student 자동 개선 | 없음 (자동) |

**사영 변경이 가장 빈번하고, 가장 저비용이고, 가장 큰 의미 변화를 만든다.**
이것이 시스템의 주요 조작점(control point)이다.

## 사영의 조작

운용자가 개념을 바꾸고 싶을 때 조작하는 것:

| 조작 | 의미 | 효과 |
|------|------|------|
| w의 원소를 0으로 | "이 신호는 이 버킷에 무관" | 축 제거 |
| w의 비율 변경 | "이 신호가 더 중요" | 가중치 조정 |
| w의 방향 변경 | "다른 조합이 이 개념을 정의" | 개념 자체 변경 |
| θ 이동 | "기준을 높여/낮춰" | 엄격도 조정 |
| 정규화 변경 | "이 범위에서의 변화가 중요" | 민감도 조정 |

**사영의 전체가 "개념의 조작적 정의(operational definition)"이다.**
개념은 추상적인 것이 아니라, w와 θ로 구체화된 수학적 객체.

### 사영의 표현력

```
Level 0 — Single threshold:
  happy > 0.6 → ✓/✗
  표현력: 축 하나에 대한 반공간

Level 1 — Range:
  yaw ∈ [30°, 60°] → ✓/✗
  표현력: 축 하나에 대한 구간

Level 2 — Conjunction (현재 AND):
  happy > 0.6 AND AU12 > 2.0 AND |yaw| < 15 → ✓/✗
  표현력: 축 정렬 직사각형

Level 3 — Disjunction:
  (happy > 0.6 AND frontal) OR (surprise > 0.5 AND lateral) → ✓/✗
  표현력: 직사각형의 합집합

Level 4 — Derived signal:
  duchenne = min(AU6, AU12) / 5.0; duchenne > 0.5 → ✓/✗
  표현력: 비선형 결합에 대한 threshold

Level 5 — Learned projection:
  small_classifier(y₁, y₂, y₃) > 0.5 → ✓/✗
  표현력: 임의의 비선형 경계
```

사영의 복잡도가 올라갈수록 crowds가 더 좋은 pseudo-label을 생성하지만,
crowds의 단순함(해석 가능성, 디버깅 용이성)을 잃는다.
이 trade-off 자체가 연구 질문이다.

## Few-Shot과의 연결

새 버킷 = 새 사영 정의. 성공 조건은:

```
Case 1 — 기존 차원의 새 조합:
  "eyes_closed" = neutral > 0.5 AND AU43 > 2.0
  → 기존 z에 이미 관련 정보 인코딩됨
  → Head만 학습, 50건이면 충분

Case 2 — 기존 threshold의 이동:
  warm_smile의 happy > 0.6 → happy > 0.8
  → pseudo-label만 갱신, 기존 Head 재학습

Case 3 — 새로운 observer 조합:
  body.pose "양팔 벌린" + expression "surprise"
  → Teacher Head에 body.pose가 있었다면 z에 정보 있음
```

**Few-shot 성공 조건 = 새 사영이 요구하는 정보가 기존 z에 이미 인코딩되어 있는가.**
Soft loss(Teacher Head)가 이를 보장 — z가 모든 observer의 연속 출력을 재구성할 수 있으므로,
observer가 관측한 모든 정보가 z에 들어있다.

---

# Part 7: Novelty 강화를 위한 추가 방향

## Competence Boundary 기반 Bias 모델링

기존 crowds에서 annotator bias는 상태 무관 confusion matrix로 모델링한다 (Dawid-Skene):

```
P(annotator says "yes" | truth = "yes") = 0.9
P(annotator says "yes" | truth = "no") = 0.2
```

VisualBind의 observer bias는 입력 의존적이다:

```
HSEmotion:  정면 조명 좋을 때 reliable, 측면/모션블러에서 unreliable
LibreFace:  AU12 SNR 좋음, AU6 조명 민감
6DRepNet:   정면 ±30° reliable, 측면에서 MAE 급증
```

Observer의 학습 데이터 분포가 정의하는 신뢰 영역에 따라 bias가 달라진다:

```
기존:  P(vote_i | truth)           — 상수 confusion matrix
제안:  P(vote_i | truth, x)        — 입력 의존 reliability
     = P(vote_i | truth) · c_i(x)  — c_i(x)는 competence function
```

Competence function을 데이터에서 학습하면 Dawid-Skene의 자연스러운 확장이 된다.
Crowds 문헌에서 instance-dependent annotator quality는 일부 다뤄졌지만,
ML model의 학습 분포 기반 competence boundary와 연결한 연구는 거의 없다.

## Cross-Task Distillation의 Information-Theoretic 분석

```
N개의 이종 task observer가 있을 때,
합의로 생성한 pseudo-label의 품질은
observer 수 N, task 다양성 D, 사영 품질 Q의 함수:

Pseudo-label quality ≥ f(N, D, Q)
```

기존 multi-teacher KD에서 teacher는 같은 task. Student가 teacher를 넘는 이유는 ensemble 효과뿐.
VisualBind에서 teacher는 다른 task. Student가 넘을 수 있는 이유:

1. **Cross-modal 관계를 학습** (AU6+AU12 conjunction = Duchenne smile)
2. **Bias를 상호 보정** (Expression이 틀릴 때 AU가 맞는 상보적 구조)
3. **Threshold 너머의 연속 정보를 활용** (happy=0.59와 0.61의 구분)

이 세 가지가 합쳐져서 student > any single teacher가 가능해지는 조건을 형식화할 수 있다면,
이것은 genuine theoretical contribution이다.

---

# Part 8: 학술적 포지셔닝

## 논문 제목 후보

- "Projected Crowds: Unifying Multi-Modal Fusion and Learning from Crowds via Threshold Projection"
- "From Observers to Annotators: Threshold Projection Bridges Multi-Modal Fusion and Crowd Learning"
- "When Frozen Models Vote: Learning from Projected Heterogeneous Observers without Labels"

## Contribution 정리

1. **Projected Crowds 프레임워크**: 이종 연속 공간의 observer를 threshold로 공통 결정 공간에 사영하고,
   사영 전 연속 정보와 사영 후 이산 투표를 dual-mode로 동시 활용하는 형식적 프레임워크 제안.
   이것은 multi-modal fusion과 learning from crowds를 하나로 통합한다.

2. **사영 = 개념 정의**: Concept drift를 사영 공간의 변화로 재해석.
   운용자가 사영을 변경하여 의도적으로 개념을 재정의할 수 있고,
   Student가 자동으로 새 개념에 적응하는 메커니즘.

3. **입력 의존 Competence Function**: Observer의 학습 경계를 고려한
   instance-aware reliability 모델링. 기존 crowds의 상수 confusion matrix를 확장.

4. **정보 손실과 복구의 정량적 분석**: 사영에 의한 MI 손실량과
   dual-mode loss에 의한 복구량을 정량화. Exp 1 vs Exp 3 ablation으로 검증.

## Reviewer 공격 예상과 방어

| 예상 공격 | 방어 |
|-----------|------|
| "Snorkel에 ML model을 LF로 넣으면 되는 것 아닌가" | Snorkel은 연속 출력을 다루지 못함. Dual-mode ablation으로 연속 정보의 가치 정량화 |
| "Multi-teacher KD와 뭐가 다른가" | KD는 same-task. Cross-task에서 bias 구조가 다름 (competence boundary). KD baseline 대비 비교 |
| "Threshold 설정이 자의적" | 기존 운용 시스템의 threshold 재활용. Threshold sensitivity 분석 추가 |
| "Observer 독립성 가정 위반" | 정면 인정 + N_eff 계산. 의존성 하에서도 개선됨을 실증 |

## 게재 전략

| 범위 | 학회 | 필요 완성도 |
|------|------|-----------|
| Mode A vs C ablation + 내부 데이터 | FG / ECCV Workshop | 4-6주 |
| + Few-shot transfer + BP4D/DISFA | FG / ECCV Main | 2-3개월 |
| + 이론적 bound + scaling law | NeurIPS / ICML | 6-9개월 |

---

# Part 9: 다음 단계

## 열린 연구 질문

1. **사영의 표현력과 Student 성능의 관계** — Level 0~5 사영에서 실험적으로 어떤 차이가 나는가
2. **사영 간 전이** — 기존 버킷의 사영이 새 버킷의 사영 학습에 어떻게 도움이 되는가
3. **사영의 자동 학습** — 소량의 예시에서 사영(w, θ) 자체를 학습할 수 있는가
4. **사영 복잡도 vs pseudo-label 품질 trade-off** — 해석 가능성과 성능의 최적 균형점
5. **N-observer scaling law의 정보 이론적 유도** — observer 수에 따른 사영 손실 보상 조건

## 실행 우선순위

```
즉시:    Day 0 — N_eff + Exp 0 (crowds 프레이밍 검증)
Week 1:  새 코드 (model.py, trainer.py) + Anchor Set 500건
Week 2:  Exp 1 vs 3 (Mode A vs Mode C) — 핵심 ablation
Week 3:  Exp 4 (few-shot) — 사영 변경으로 새 버킷
Week 4:  정보 손실/복구 정량화 — MI 측정
```
