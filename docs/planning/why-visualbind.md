# 왜 visualbind인가

> "우리만의 AI 모델을 만들 리소스가 없다. 하지만 매일 2-3000건의 데이터가 쏟아진다."

이 문서는 왜 VisualBind가 필요한지, 어떤 문제를 풀려는 것인지,
그리고 그것이 학술적으로 어떤 위치에 있는지를 설명합니다.

---

## 반복되는 루프

AI 시스템을 실환경에 적용할 때 반복되는 문제:

```
자체 AI 모델을 개발할 여유와 리소스가 없다
    → 파운데이션 모델이나 범용 모델을 적용한다
    → 우리 문제에 적합한 결과를 내놓지 않는다
    → 모델을 교체해본다
    → 또 안 맞는다
    → 반복...
```

벤치마크에서 99% 정확도를 주장하는 모델이 실환경에서는 95%에 그치는 경우가 흔하다.
95%는 논문에서는 좋은 수치이지만, 실시간 시스템에서는 5%의 오류가 연쇄적 문제를 만든다.

### 범용 임베딩 모델의 한계

CLIP, DINO 같은 범용 비전 임베딩 모델에 기대를 걸고 적용해보았으나,
**얼굴 표정 변화나 포즈 차이에 대해 거의 일관된 출력**을 보여주었다.
같은 사람이 웃든 무표정이든, 정면이든 측면이든 임베딩 거리가 거의 동일.

얼굴 분석에 적합한 모델을 찾기 위해 **수십 가지 모델을 평가**했다.
결론: 완벽한 단일 모델은 없고, 도메인 특화 모듈의 조합이 현실적 해법이었다.

```
범용 모델 (CLIP/DINO) → 표정/포즈 변별력 부족
단일 특화 모델         → 우리 도메인에 완전히 맞지 않음
특화 모듈 조합         → 작동하지만 수동 threshold 조합이 비효율적
→ VisualBind           → 조합의 학습 자동화
```

**작은 조직은 모델을 직접 학습시킬 리소스가 없고, 범용 모델은 우리 도메인에 완전히 맞지 않는다.**

---

## 우리가 가진 것과 없는 것

| 자산 | 설명 |
|------|------|
| **N개의 frozen 모델** | 14개 pretrained 모듈 (face detection, expression, AU, pose 등) |
| **매일 생성되는 데이터** | 하루 2-3000건 × 2분 × 30fps ≈ 일 1000만+ 프레임. 도메인 특화 데이터 |
| **기존 파이프라인** | visualpath 기반 14개 모듈 조합 + catalog_scoring (Fisher-weighted 25D matching) |

| 부재 | 영향 |
|------|------|
| **Annotation 리소스** | 매일 수천 건의 데이터를 라벨링할 인력/비용 없음 |
| **학습 인프라** | GPU 1개 보유. 대규모 모델 학습을 위한 클러스터는 없음 |
| **대규모 팀** | ML 전문가 팀이 모델을 계속 개선할 구조 아님 |

---

## 목표

> **Annotation 없이, N개의 불완전한 frozen 모델의 조합을 통해,
> 어떤 단일 SOTA 모델보다 우리 도메인에서 나은 결과를 만들어내는 시스템.**

유일한 무기는 매일 쏟아지는 도메인 특화 데이터다.
시스템이 자율적으로 이 데이터를 활용할 수 있다면 이야기가 달라진다.

더 정확하게는: **annotation-free training, annotation-efficient validation**.
학습에는 annotation이 필요 없고, 검증에만 소량의 annotation을 사용한다.

---

## 핵심 통찰: Threshold는 추상 개념 경계의 역사상이다

### 현재 시스템이 이미 하고 있는 것

momentscan에서 "3/4 각도 happy" 프레임을 수집할 때:

```
face.detect confidence > 0.8  → ✓/✗
pose yaw ∈ [30°, 60°]        → ✓/✗
expression happy > 0.6        → ✓/✗
brightness ∈ [0.3, 0.7]      → ✓/✗

ALL 만족 → "이 버킷에 적합한 프레임"
```

각 모듈은 원래 다른 것을 측정한다 (AU는 근육, Pose는 각도, Expression은 감정).
하지만 **threshold를 적용하는 순간, 모두 같은 질문("이 프레임이 이 버킷에 적합한가?")에 대한 weak binary vote를 던지는 구조**가 된다.

### Threshold는 방향이 반대다

Threshold를 "연속값을 0/1로 자르는 것"으로만 보면 핵심을 놓친다.

"warm_smile"이라는 추상 개념은 어딘가에 존재하는 복잡한 영역이다.
각 frozen 모델의 출력 공간에 개별적으로 threshold를 설정하는 것은,
이 추상 개념의 경계를 각 feature space에 **역사상(back-project)**한 것이다.

```
올바른 방향:
  추상 개념 ("warm_smile") → 개념 공간에서 경계 정의 → 각 feature space로 역사상

실제로 하는 것:
  각 feature space에서 독립적으로 threshold 설정 → AND 결합 → 개념의 근사
```

즉, `happy > 0.6`은 expression 공간을 decision 공간으로 사영하는 것이 아니라,
**"warm_smile" 개념의 경계가 expression 공간의 happy 축에 역사상된 근사값**이다.

### Frozen 모델 = Feature Extractor (고정된 매핑)

이 구조는 kernel methods와 개념적으로 대응한다:

```
Kernel method:  φ(x) → feature space   (미리 정의된 매핑)
                hyperplane in feature space (학습된 경계)

VisualBind:     fᵢ(x) → output space   (frozen 모델, 남이 학습한 매핑)
                threshold in each space (수동 설정된 경계)
```

차이점: kernel은 해당 task를 위해 설계되었지만, frozen 모델은 **다른 task를 위해 학습**되었다.
그래서 각 출력 공간에서의 threshold가 우리 task에 최적이 아니다.

### 독립 threshold의 두 가지 정보 손실

```
Observer 출력: y_expr = [happy=0.82, neutral=0.10, surprise=0.05, ...] ∈ ℝ⁸

Step 1 — 축 선택: "happy" 차원만 취함
         ℝ⁸ → ℝ¹  (8차원에서 1차원으로, 7차원 버림)

Step 2 — 절단: happy > 0.6?
         ℝ¹ → {0,1}  (연속에서 이산으로)
```

Step 1에서 축을 하나만 고르면서 나머지를 버리고,
Step 2에서 연속값을 이진화하면서 "얼마나"의 정보를 버린다.

이것을 observer마다 **독립적으로** 반복한 뒤 AND로 결합하면,
24차원 joint space에서 **축 정렬 직사각형(axis-aligned box)**을 정의하는 것과 같다.

### 추상 개념이 독립 모델 간 상관을 만든다

여기서 핵심적인 문제가 드러난다.
각 frozen 모델은 서로 독립적인 task를 위해 개발되었지만,
**추상 개념 자체가 이들의 출력 사이에 상관을 만들어낸다.**

```
"밝게 웃을수록 눈이 감긴다"

Expression 모델: happy ↑
AU 모델:         AU6 (cheek raiser) ↑, AU43 (eye closure) ↑
Head Pose 모델:  pitch 약간 ↑ (고개가 살짝 뒤로)
```

이 상관은 모델들이 같은 데이터로 학습되어서가 아니다.
각 모델은 완전히 독립적으로 개발되었다.
**하지만 "밝은 미소"라는 개념 자체가 얼굴 근육, 감정, 포즈 사이에 물리적 상관을 내포하고 있다.**

추상 개념의 경계를 각 축에 독립적으로 역사상하면, 이 상관이 소실된다:

```
모델 개발 관점:   Expression ⊥ AU ⊥ Pose  (독립적 task)
추상 개념 관점:   Expression ↔ AU ↔ Pose  (개념이 연결)

Independent thresholds:  happy > 0.6 AND AU12 > 2.0
  → 이 둘을 독립으로 취급
  → happy = 0.9 + AU12 = 1.5 (매우 밝은 미소, AU는 약간 부족) → ✗
  → happy = 0.61 + AU12 = 2.1 (약한 미소, AU는 겨우 넘음) → ✓
  → 직관적으로 첫 번째가 더 "warm_smile"인데 탈락
```

### 축 정렬 직사각형의 기하학

```
축 정렬 직사각형                 실제 개념의 경계
(독립 threshold AND):           (cross-modal 상관 포함):
    AU12                            AU12
     ↑                               ↑
   5 │  ┌────────┐                  5 │   ╭────────╮
     │  │        │                    │  ╱          ╲
     │  │  ✓     │                    │ │  ✓         │
   2 │──┘        │                  2 │  ╲          ╱
     │  ✗        │                    │   ╰────────╯
   0 └───────────┘→ happy           0 └──────────────→ happy
     0   0.6     1.0                  0              1.0

왼쪽: happy > 0.6 AND AU12 > 2.0 (독립, 축 정렬)
오른쪽: happy와 AU12이 상보적 관계 (하나가 높으면 다른 하나 기준 완화)
```

### Observer 독립성 위반의 근본 원인

이 관점은 N_eff 논의에도 새로운 시각을 제공한다.

```
표면적 원인:  face.detect → face crop → 여러 모듈이 같은 입력 공유
              → 기술적 의존성 (모듈 설계로 일부 완화 가능)

근본적 원인:  추상 개념 자체가 모듈 출력 간 상관을 만듦
              → 물리적/의미적 의존성 (환원 불가능)
              → 모델을 아무리 다양화해도 이 상관은 남음
```

얼굴 근육과 감정과 포즈는 물리적으로 연결되어 있다.
이 연결은 모델의 문제가 아니라 세계의 구조이며,
**crowds의 독립성 가정이 근본적으로 위반되는 이유**이기도 하다.

### 현재 시스템의 한계

수동 threshold 기반 AND 결합이 가진 문제:

1. **방향이 반대다** — 추상 개념에서 threshold로 가야 하는데, threshold에서 개념을 조립
2. **축 간 독립 가정** — 개념이 만든 cross-modal 상관을 구조적으로 무시
3. **축 정렬 직사각형에 갇힘** — 비선형 관계를 표현 불가
4. **정보 손실** — happy = 0.59와 0.61이 threshold 0.6에 의해 완전히 다른 결과가 됨
5. **스케일하지 않음** — 14개 모듈 × 5개 버킷 = 70개 threshold를 수동 디자인

**VisualBind는 추상 개념의 경계를 joint feature space에서 직접 학습하여,
독립 threshold의 역사상 근사를 대체한다.**

### 개념 시드 — 추상 개념을 어떻게 정의하는가

추상 개념은 무에서 발견되지 않는다. **최소한의 시드(seed)**가 필요하다.
"Annotation-free"는 학습 데이터에 annotation이 불필요하다는 것이지,
개념 정의 자체가 자동이라는 뜻이 아니다.

```
개념 시드: 참조 이미지 (또는 텍스트 프롬프트)
     ↓
Frozen 모델이 독립적으로 관찰
     ↓
  Expression: [0.82, 0.85, 0.79, ...]  (참조 이미지들의 happy 분포)
  AU:         [3.1, 2.8, 3.5, ...]      (참조 이미지들의 AU12 분포)
  Pose:       [5°, -3°, 8°, ...]        (참조 이미지들의 yaw 분포)
     ↓
Joint 25D space에서 "이 참조들이 공유하는 영역"의 경계를 발견
     ↓
= 추상 개념의 결정 경계 (자동 도출)
     ↓
역사상: 학습된 경계를 각 축으로 투영하면 threshold가 자동 도출됨
        (해석/디버깅/운용자 조정용)
```

**Threshold는 입력이 아니라, 학습된 결정 경계의 역사상된 부산물이다.**

현실적으로 catalog에 이미 참조 이미지가 존재한다
(warm_smile 25장, cool_expression 27장, lateral 31장).
이 이미지들에 frozen 모델을 실행하면 25D 분포가 나오고,
분포에서 threshold를 자동 추정할 수 있다 (percentile 기반 등).

이 방식의 양방향 인터페이스:

| 방향 | 동작 | 효과 |
|------|------|------|
| **Forward** | 참조 이미지 → frozen 모델 관찰 → joint 경계 발견 | 개념 자동 정의 |
| **Backward** | 학습된 경계 → 각 축으로 역사상 → threshold 도출 | 해석 가능한 설명 |
| **Adjustment** | 운용자가 역사상된 threshold를 조정 → 재학습 | 개념 미세 조정 |

---

## 세 가지 학습 모드

독립 threshold가 추상 개념의 crude marginal 근사라면, 가능한 학습 전략은 세 가지로 나뉜다.

### Mode A — Crowds-only (사영 후 정보만 사용)

```
입력: {v₁, v₂, ..., v_N}  (binary votes)
방법: Dawid-Skene / majority vote
정보량: N bits (최대)
```

사영의 정보 손실이 그대로 학습에 반영된다.
happy = 0.82와 0.61이 구분되지 않고, 0.61과 0.59는 극적으로 다르게 취급된다.

### Mode B — Fusion-only (사영 전 정보만 사용)

```
입력: {y₁, y₂, ..., y_N}  (연속 출력, concatenated)
방법: MLP / attention으로 직접 학습
정보량: Σ dim(Yᵢ) × precision (최대)
문제: ground truth label이 없음 → self-supervised만 가능
```

정보는 풍부하지만 **방향이 없다**. "이 프레임이 좋은 초상화인가?"에 대한 답이 없다.

### Mode C — Projected Crowds (사영 전 + 후 동시 사용)

```
입력: {y₁, ..., y_N} + {v₁, ..., v_N}
방법: dual-mode loss
  Hard: BCE(student(y), crowds_consensus(v))  ← 방향성 (crowds)
  Soft: MSE(teacher_head(z), yᵢ)             ← 정보 복구 (fusion)
정보량: Mode A + Mode B의 결합
```

사영된 vote가 crowds 합의를 통해 방향을 제공하고,
연속 출력이 soft reconstruction을 통해 사영 전 정보를 복구한다.
두 경로가 shared encoder의 z를 통해 만나면서,
z는 crowds가 가리키는 방향으로 이동하되 연속 정보의 풍부함을 유지한다.

**VisualBind는 Mode C를 구현한다.** Exp 1 (hard-only) vs Exp 3 (hybrid)가 Mode A vs Mode C 비교이다.

---

## Projected Crowds — 이론적 프레이밍

### "동치"가 아니라 "확장"

VisualBind의 구조는 Learning from Crowds와 비슷해 보이지만, **동치가 아니라 확장**이다.
기존 프레임워크가 다루지 않는 설정이 존재한다.

기존 연구 지형에는 빈 자리가 있다:

```
          이산 출력          연속 출력
        ┌─────────────┬─────────────┐
같은    │ Dawid-Skene  │ (비어있음)   │
task    │ Snorkel      │             │
        ├─────────────┼─────────────┤
다른    │ (비어있음)    │ Multi-modal │
task    │              │ fusion      │
        └─────────────┴─────────────┘

VisualBind는 네 사분면을 연결:
  - 다른 task의 연속 출력 (frozen 모델)
  → 추상 개념 경계의 역사상 (threshold)
  → 같은 task의 이산 투표
  - 동시에 연속 출력도 활용 (dual-mode)
```

**추상 개념의 역사상 과정과 그에 따른 정보 손실 · cross-modal 상관 소실을
명시적으로 다루는 프레임워크가 없다.** 이것이 빈 자리이다.

### Snorkel과의 진짜 차이

Snorkel의 Labeling Function은 사람이 설계한 규칙이다.
VisualBind의 "LF"는 수백만 파라미터를 가진 독립 학습된 ML 모델의 출력이다.

| | Snorkel LF | VisualBind Observer |
|---|---|---|
| 출력 | binary/discrete | **연속 다차원** (threshold 전) |
| 정보량 | ~1 bit/LF | **수십 bit/observer** (25D continuous) |
| 오류 구조 | 규칙 설계 실수 (random-like) | **학습 경계 기반 체계적 bias** |
| 상관 원인 | LF 설계 중복 | **추상 개념이 만드는 물리적 상관** |
| 개수 | 수십~수백 개 | 소수 (5-14개), 각각 고비용 |

**Snorkel은 "많은 약한 신호"를 다루고, VisualBind는 "소수의 강하지만 편향된 신호"를 다룬다.**

### Learning from Crowds와의 진짜 차이

Dawid-Skene의 annotator는 같은 질문에 같은 형식으로 답한다.
VisualBind의 observer는 **다른 질문에 다른 형식으로 답하는데,
추상 개념의 경계가 각 출력 공간에 역사상되면서 비로소 같은 질문이 된다.**

Crowds는 annotator가 이미 같은 공간에 있다고 가정한다.
또한 annotator 간 독립성을 가정하지만, VisualBind에서는
**추상 개념 자체가 observer 간 환원 불가능한 상관을 만든다** — 이것은 모델을 다양화해도 제거할 수 없다.

### Rodrigues & Pereira (2018) "Deep Learning from Crowds"와의 차별화

구조적으로 가장 유사한 선행연구. DNN + per-annotator heads로 annotator reliability와 true label을 동시 학습.
VisualBind의 shared encoder + teacher heads + bucket heads 구조와 닮아있다.

| | Rodrigues & Pereira | VisualBind |
|---|---|---|
| Annotator | 사람 (같은 task, 같은 출력 형식) | **frozen 모델 (다른 task, 이종 출력)** |
| Annotator 출력 | 이산 label만 | **연속 다차원 + 이산 vote (dual-mode)** |
| 상관 원인 | annotator 간 독립 가정 | **추상 개념이 만드는 환원 불가능한 상관** |
| Downstream | 단일 task | **multi-bucket + few-shot transfer** |
| 핵심 차이 | annotator가 이미 같은 공간 | **이종 공간의 역사상 + 역사상 시 소실된 cross-modal 상관 복원** |

Rodrigues & Pereira가 다루지 않는 것:
역사상 과정에서 발생하는 정보 손실, 축 간 독립 가정에 의한 cross-modal 상관 소실,
그리고 dual-mode loss를 통한 복원. 이것이 VisualBind의 핵심 확장이다.

상세한 이론적 전개는 `projected-crowds.md` 참조.

---

## Student가 Teacher를 넘는 메커니즘

### 단순 distillation의 한계

```
Student = Teacher 따라하기
→ Student ≤ Teacher (상한이 teacher)
```

### Student가 배우는 것 = 추상 개념이 역사상한 상관 구조

독립 threshold는 추상 개념의 marginal 근사이므로, 축 간 상관을 잃는다.
**Student MLP는 joint feature space에서 학습하므로, 이 상관을 복원할 수 있다.**

```
Teacher_i의 출력 = underlying truth + bias_i

Student가 학습하는 것:
  (1) 모든 teacher의 출력 패턴 뒤에 있는 truth
  (2) 각 teacher의 bias 패턴
  (3) 추상 개념이 만들어낸 teacher 간 cross-modal 상관

→ Teacher₃이 이 상황에서 자주 틀린다는 것을 학습
→ Teacher₁과 Teacher₂가 동의하면 더 신뢰
→ AU6+AU12 conjunction = Duchenne smile (어떤 단일 teacher도 모르는 관계)
→ happy↑이면 AU12 threshold를 완화해도 된다는 상보적 구조
→ 어떤 단일 Teacher보다 나은 판단 가능
```

핵심: Student가 teacher의 **출력**을 따라하는 게 아니라,
teacher들의 **출력 패턴 뒤에 있는 추상 개념의 구조**를 학습한다.
개별 threshold가 독립적으로 역사상하면서 잃어버린 cross-modal 상관을,
Student가 joint space에서 복원하는 것이다.

Student가 teacher를 넘을 수 있는 세 가지 메커니즘:

1. **Cross-modal 상관 복원** — 추상 개념이 만든 축 간 관계를 joint space에서 학습. 독립 threshold는 구조적으로 포착 불가.
2. **Bias 상호 보정** — Expression이 틀릴 때 AU가 맞는 상보적 구조. 독립 threshold에서는 하나가 틀리면 바로 탈락.
3. **Threshold 너머의 연속 정보 활용** — happy=0.59와 0.61의 구분, AU12=1.9와 2.1의 구분을 연속적으로 반영.

### MoE와의 차이

MoE(Mixture of Experts)는 적절한 expert를 선택하는 메커니즘이다.
우리 문제는 다르다: **불완전한 expert들을 데리고 청출어람할 수 있는 모델을 학습시키는 task.**
MoE에서 expert는 이미 유능하고 routing만 필요하지만,
우리의 observer들은 각자의 경계 내에서만 부분적으로 유능하다.

---

## 3층 분리 아키텍처

추상 개념의 역사상 과정을 인식하면, 시스템이 자연스럽게 세 층으로 분리된다.

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

이 분리가 만드는 핵심 속성:

| 변경 사항 | 영향 범위 | 비용 |
|-----------|----------|------|
| 새 observer 추가 | 전체 재학습 | 높음 |
| **사영 변경 (threshold 수정)** | **pseudo-label 재생성 + 재학습** | **낮음** |
| **새 사영 추가 (새 버킷)** | **새 Head만 학습** | **매우 낮음** |
| 데이터 축적 | Student 자동 개선 | 없음 (자동) |

**사영 변경이 가장 빈번하고, 가장 저비용이고, 가장 큰 의미 변화를 만든다.**
이것이 시스템의 주요 조작점(control point)이며, Drift 적응의 열쇠이기도 하다.

---

## 자기 개선 루프 — 매일 데이터로 성장하는 시스템

```
Day 1:  검출기 95% + expression 70% + AU 75% + pose 85%
             ↓
        2-3000건/일 → 수만 프레임
             ↓
        다수 observer의 고신뢰 합의 → pseudo ground truth
             ↓
        Student 모델 점진적 적응
             ↓
Day 30: Student 97%+ (도메인 특화)
        + 더 좋은 검출 → 더 좋은 expression/AU 입력 → 전체 향상
             ↓
        선순환
```

매일 2-3000건의 데이터를 annotation 작업 없이 자율적으로 활용하여
점진적으로 개선되는 모델을 생산하는 시스템.

### 안전장치 (self-degrading 방지)

자기 개선은 자기 악화로도 이어질 수 있다.
Validation gate, rollback, blind spot 감지가 아키텍처에 내장되어야 한다.
(`how-visualbind.md`에서 상세 기술)

---

## Drift 적응 — 365일 운용 시스템의 필수 메커니즘

1년 365일 동작하는 실환경 시스템에서 drift는 피할 수 없다.
VisualBind는 annotation 없이 drift에 적응하는 메커니즘을 내장한다.

### 4가지 Drift

| Drift | 설명 | 예시 |
|-------|------|------|
| **Data drift** | 입력 분포 변화 | 계절별 조명, 고객층 변화, 시간대별 패턴 |
| **Concept drift** | "좋은 프레임" 기준 변화 | 디자인팀 요구: "이번 시즌은 측면 portrait 중심으로" |
| **Teacher drift** | upstream 모듈 교체/업데이트 | 6DRepNet → 다른 모델로 교체 시 출력 범위 변화 |
| **Self-reinforcing** | 학습 결과가 다음 학습을 오염 | Student 편향 → pseudo-label 오염 → 편향 강화 |

### 기존 방식 vs VisualBind

```
기존 (수동 대응):
  환경 변화 → 성능 저하 감지 → annotation 재수집 → 재학습 → 배포
               (수일)         (수주-수개월, 비용↑↑)   (수일)

VisualBind:
  환경 변화 → Teacher 출력 분포 변화 감지 → pseudo-label 자동 갱신 → 재학습
               (자동, 당일)              (자동, 당일)           (분 단위)
```

**Annotation bottleneck을 제거하면서 동시에 drift adaptation을 자동화**한다.
이 두 문제는 실무에서 항상 함께 오지만, 보통 별개로 다뤄진다.

### Concept Drift = 사영 공간의 변경

3층 분리 아키텍처에서 concept drift의 의미가 명확해진다.

```
기존 ML에서 concept drift:
  "환경이 변해서 모델이 맞지 않게 된다" → 방어해야 할 문제

Projected Crowds에서 concept drift:
  "운용자가 사영을 바꿔서 개념을 재정의한다" → 시스템의 핵심 기능
```

**Observer 공간은 고정이고, 사영이 바뀔 때 개념이 바뀐다.**
같은 프레임, 같은 observer 출력인데, 사영만 바꾸면 전혀 다른 판단이 나온다.

```
시즌 1 (겨울):
  Π₁ = {warm_smile, cool_expression, lateral}
  → Student가 이 3개 사영에 대해 학습

디자인팀: "봄 시즌은 눈감은 장면이 필요해"

시즌 2 (봄):
  Π₂ = {warm_smile, cool_expression, lateral, eyes_closed}
  → 새 사영 τ_eyes_closed 추가
  → 기존 데이터에 즉시 적용 가능 (observer 출력은 이미 있음)
  → 50건 pseudo-label → 새 Bucket Head만 학습

디자인팀: "따뜻한 미소 대신 밝은 웃음으로 바꿔줘"

시즌 3 (여름):
  warm_smile의 τ_expr를 happy > 0.6 → happy > 0.8로 변경
  → 같은 이름의 버킷이지만 개념이 바뀜
  → pseudo-label이 자동으로 갱신됨
  → Student가 새 개념에 재학습
```

다소 주관적인 기준이 시즌마다 바뀌는 것이 우리 도메인의 본질이다.
catalog 메커니즘과 momentbank로 다양성을 저장하고,
**VisualBind의 multi-bucket + few-shot이 기준 변경에 빠르게 적응하는 메커니즘**이 된다.

```
momentbank  — 다양성 저장 (모든 가능성을 보존)
catalog     — 현재 시즌의 수집 기준 정의 ("무엇을 찾을지")
visualbind  — 기준 변경에 빠르게 적응하는 학습 ("어떻게 찾을지")
```

### Data Drift와 Teacher Drift

Data drift와 Teacher drift는 Observer Layer나 입력 분포의 변화다.
이들은 concept drift와 달리 의도적이지 않지만,
자기 개선 루프 + drift 감지 메커니즘으로 자동 대응한다.

---

## catalog_scoring과의 관계

catalog_scoring은 사실상 **VisualBind의 수동 버전**이다:

| catalog_scoring | VisualBind |
|----------------|------------|
| 14개 모듈의 25D signal 조합 | N개 모듈의 임의 차원 signal |
| Fisher ratio로 가중치 계산 | DNN이 가중치를 데이터에서 학습 |
| 카테고리별 centroid matching (선형) | Latent space에서 비선형 boundary |
| 7장 참조 이미지 + 수동 threshold | Daily data에서 자동 pseudo-label |
| 새 카테고리 = threshold 정의 + 참조 촬영 | 새 버킷 = 50건 few-shot transfer |
| 정적 (한 번 계산하면 고정) | 매일 데이터로 성장 |

기하학적으로 보면:

```
catalog_scoring = 축 정렬 직사각형 경계 (각 축 독립 threshold의 AND)
Student MLP    = 임의의 비선형 경계 (cross-modal 관계 학습)
```

catalog_scoring의 한계 = VisualBind가 풀어야 할 것:
- `SIGNAL_RANGES` 하드코딩
- Fisher = 선형 분리력만 측정
- 카테고리 사전 정의 필요
- 가중치가 데이터셋 전체에 고정
- axis-aligned rectangular boundary

catalog_scoring은 VisualBind의 **baseline이자 전환 출발점**이다.

---

## 학술적 포지셔닝

### 위치

**"Projected Crowds: Cross-Task Crowd Distillation via Threshold Projection of Heterogeneous Observers"**

### 선행연구 계보

VisualBind는 4개의 연구 흐름이 교차하는 지점에 있다.
그리고 이 교차점에는 기존 연구가 다루지 않은 빈 사분면이 있다.

#### 1. Learning from Crowds (이론적 기반)

여러 noisy annotator의 출력으로부터 underlying truth를 추정하는 연구 흐름.

- **Dawid & Skene (1979)**: EM으로 annotator reliability + true label 동시 추정
- **Raykar et al. (2010)**: classifier + annotator model 동시 학습
- **Rodrigues & Pereira (2018)**: Deep learning from crowds — DNN + crowd layer

> VisualBind에서: threshold 적용된 frozen 모듈 = noisy annotator.
> **핵심 차이**: Crowds는 annotator가 이미 같은 공간에 있다고 가정한다.
> VisualBind의 observer는 이종 공간에 있으며, threshold **사영**을 통해 같은 공간으로 온다.
> 이 사영 과정과 정보 손실을 다루는 것이 기존 crowds에 없는 부분이다.

#### 2. Snorkel / Data Programming (pseudo-label 생성)

Noisy labeling function들의 조합으로 annotation 없이 학습하는 프레임워크.

- **Ratner et al. (2017)**: Snorkel — labeling function의 accuracy/correlation 자동 추정, GT 불필요
- **Ratner et al. (2019)**: Snorkel MeTaL — 계층적 multi-task에 대한 weak supervision 확장

> VisualBind에서: threshold 적용된 모듈 = labeling function.
> **핵심 차이**: Snorkel은 이산 출력만 다루고, LF 수가 많은 regime.
> VisualBind는 소수의 고비용 observer에서 연속 출력을 함께 활용한다 (dual-mode).

#### 3. Multi-Teacher Knowledge Distillation (아키텍처)

여러 teacher 모델의 지식을 단일 student로 압축하는 연구 흐름.

- **Hinton et al. (2015)**: Knowledge Distillation — soft target으로 학습
- **Zuchniak et al. (2023)**: Multi-teacher KD as ensemble compression — student가 개별 teacher를 초과
- **NeurIPS (2024)**: Ensemble-Then-Distill — ensemble pseudo-label → student distillation
- **Kendall et al. (2018)**: Multi-task learning using uncertainty — homoscedastic uncertainty로 task별 loss 가중치 자동 학습

> VisualBind에서: 14개 frozen teacher → student.
> **핵심 차이**: 기존 KD는 같은 task의 homogeneous teacher.
> VisualBind는 다른 task의 heterogeneous teacher. cross-task에서 bias 구조가 근본적으로 다르다.
> Kendall uncertainty weighting으로 이종 loss의 스케일 문제를 해결한다.

#### 4. Foundation Model Distillation (end-to-end 비전)

대형 모델의 지식을 자동 annotation으로 소형 모델에 전이하는 실무적 접근.

- **Roboflow (2023)**: Autodistill — foundation model → pseudo-label → target model
- **Lu et al. (2025)**: Single-model multi-task face analysis — 검출+인식+속성을 단일 모델로

> VisualBind Phase 2에서: 14 Teachers → pseudo-label (annotation 공장) → Student CNN (end-to-end).
> **핵심 차이**: 단일 foundation model이 아닌 **cross-task crowds 합의**로 pseudo-label 생성.
> 대규모 인프라 불필요 (GPU 1개 + 도메인 데이터).

### 빈 사분면 — VisualBind가 차지하는 자리

기존 연구는 네 사분면 중 세 곳만 다뤘다:

- **같은 task + 이산 출력**: Dawid-Skene, Snorkel
- **다른 task + 연속 출력**: Multi-modal fusion
- **같은 task + 연속 출력**, **다른 task + 이산 출력**: 비어있음

Projected Crowds는 사영(τ)으로 네 사분면을 연결한다.
다른 task의 연속 출력이 사영을 거쳐 같은 task의 이산 투표가 되고,
동시에 사영 전 연속 출력도 활용한다.

### 논문 Contribution

1. **Projected Crowds 프레임워크**: 추상 개념의 경계가 각 이종 feature space에 역사상될 때 발생하는 정보 손실(marginal 근사에 의한 cross-modal 상관 소실)을 형식화하고, 역사상 전 연속 정보와 역사상 후 이산 투표를 dual-mode로 동시 활용하여 이를 복구하는 프레임워크. Multi-modal fusion과 learning from crowds를 통합한다.

2. **Cross-task crowd distillation**: 같은 task가 아닌 다른 task의 frozen observer들로부터 annotation 없이 domain-adapted student를 학습. 독립 threshold가 잃어버린 cross-modal 상관을 Student가 joint space에서 복원한다.

3. **추상 개념의 역사상 = 개념 정의**: Concept drift를 사영 공간의 변화로 재해석. 운용자가 threshold를 변경하여 의도적으로 개념을 재정의할 수 있고, Student가 자동으로 새 개념에 적응하는 메커니즘.

4. **정보 손실과 복구의 정량적 분석**: 독립 threshold에 의한 marginal 근사의 정보 손실량과 dual-mode loss에 의한 cross-modal 상관 복구량을 정량화. Mode A vs Mode C ablation으로 검증.

5. **Daily data stream 기반 continual self-improvement + drift adaptation 실증**

**Phase 2 (End-to-End Distillation):**

6. **Cross-Task Multi-Teacher → 단일 End-to-End 모델**: 14개 specialist 파이프라인을 annotation-free pseudo-label로 단일 CNN에 distill.
7. **Teacher를 annotation 공장으로 전환**: 학습 시에만 Teacher 사용, inference에서 제거.

### Novelty에 대한 정직한 포지셔닝

개별 메커니즘은 기존에 알려져 있다. Novelty는 **추상 개념의 역사상 과정을 형식화하고, 그 한계를 극복하는 프레임워크**에 있다:
- 추상 개념 경계의 역사상(threshold)이 만드는 정보 손실과 cross-modal 상관 소실을 형식화
- 역사상 전/후 정보를 dual-mode loss로 동시 활용하여 상관 복원
- 역사상 변경(threshold 조정)으로 concept drift에 대응하는 3층 아키텍처
- cross-task setting에서 Student가 joint space의 상관 복원을 통해 개별 teacher를 넘는 조건

### 상용 가치

| 관점 | 가치 |
|------|------|
| 학술 | Projected Crowds는 빈 사분면을 채우는 프레임워크 — FG/Main 이상 |
| **상용** | **annotation-free drift adaptation — 모든 frozen 모델 기반 시스템에 적용 가능** |

VisualBind의 본질:
> **여러 frozen inference의 noise를 고려한 underlying truth를 학습을 통해
> 통합된 context를 표현하는 단일 inference를 만들어내고,
> 현장 데이터에 적응하는 과정을 유연하게 만들어냄으로써 model drift 문제를 해결한다.**

### 대상 학회

| 범위 | 학회 | 필요 완성도 |
|------|------|-----------|
| Mode A vs C ablation + 내부 데이터 | FG / ECCV Workshop | 4-6주 |
| + Few-shot transfer + BP4D/DISFA | FG / ECCV Main | 2-3개월 |
| + 이론적 bound + scaling law | NeurIPS / ICML | 6-9개월 |

### References

```
[1] Dawid, A.P. & Skene, A.M. (1979). Maximum likelihood estimation of observer
    error-rates using the EM algorithm. JRSS-C.
[2] Raykar, V.C. et al. (2010). Learning from Crowds. JMLR.
[3] Ratner, A. et al. (2017). Snorkel: Rapid Training Data Creation with Weak
    Supervision. VLDB.
[4] Ratner, A. et al. (2019). Snorkel MeTaL: Weak Supervision for Multi-Task
    Learning. DEEM@SIGMOD.
[5] Rodrigues, F. & Pereira, F. (2018). Deep Learning from Crowds. AAAI.
[6] Hinton, G. et al. (2015). Distilling the Knowledge in a Neural Network. NeurIPS
    Workshop.
[7] Zuchniak, K. et al. (2023). Multi-teacher Knowledge Distillation as an Effective
    Method for Compressing Ensembles of Neural Networks. arXiv:2302.07215.
[8] NeurIPS (2024). Ensemble-Then-Distill Approach.
[9] Kendall, A. et al. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics. CVPR 2018.
[10] Roboflow (2023). Autodistill: Images to Inference with No Labeling.
     github.com/autodistill/autodistill.
[11] Lu, X. et al. (2025). A Single-Model Multi-Task Method for Face Recognition and
     Face Attribute Recognition. IET Image Processing.

# 논문 단계에서 검증·정리 필요 (방법론 참고용)
[12] Wang, Y. et al. (2023). FreeMatch: Self-adaptive Thresholding for Semi-supervised
     Learning. ICLR 2023. — adaptive thresholding, "threshold가 최적이 아니다"와 직접 관련
[13] Li, J. et al. (2020). DivideMix: Learning with Noisy Labels as Semi-supervised
     Learning. ICLR 2020. — noisy label에서 semi-supervised 학습
[14] Jiang, L. et al. (2018). MentorNet: Learning Data-Driven Curriculum for Very Deep
     Neural Networks on Corrupted Labels. ICML 2018. — curriculum learning from noisy labels
[15] Ren, M. et al. (2018). Learning to Reweight Examples for Robust Deep Learning.
     NeurIPS 2018. — meta-learning for noisy labels
[16] Fu, D. et al. (2020). Fast and Three-rious: Speeding Up Weak Supervision with
     Triplet Methods (Flyingsquid). ICML 2020. — Snorkel 계열 확장, 빠른 label model
```

---

## 이름에 대하여

### 왜 "bind"인가

여러 Analyzer의 출력(분리된 신호)을
하나의 학습된 표현(통합된 판단)으로
결합(bind)하는 것이 핵심이다.

### visualpath와의 관계

```
visualbase  — Read (미디어 I/O)
visualpath  — Process (분석 파이프라인, 발판)
visualbind  — Bind & Transcend (통합 학습, 발판을 넘어서는 모델)
```

visualpath는 초창기에 어쩔 수 없이 필요한 **발판(scaffolding)**이다.
visualbind는 이 발판이 만들어낸 pseudo-label을 소비하여, 궁극적으로는 visualpath 전체를 단일 모델로 distill한다.
Phase 간 모델 파라미터 전이는 없다 — 전이되는 것은 pseudo-label 생성 파이프라인과 검증 인프라이다.

```
Phase 1: visualpath → 25D signal → Student → 판단 개선 (Teacher 조합 개선)
Phase 2: visualpath → pseudo-label (학습 시만) → Student CNN → 직접 판단 (end-to-end)
Phase 3: Student가 도메인 foundation model → visualpath 퇴역
```

**visualpath 없이는 visualbind가 시작할 수 없고,
visualbind가 성숙하면 visualpath가 필요 없어진다.**

---

## 다음 단계

구체적인 아키텍처, MVP 경로, 실험 설계, 안전장치는 `how-visualbind.md`에서 다룬다.
