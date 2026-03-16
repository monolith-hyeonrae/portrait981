# VisualBind 재정립 — 논의 기록과 방향

> 2026-03-16, VisualBind의 정체성과 방향에 대한 논의를 정리한 문서.
> 기존 `why-visualbind.md`(초안), `how-visualbind.md`(초안)의 전제를 재검토하고,
> 10명의 전문가 리뷰(`visualbind-expert-review-unified.md`)를 반영하여 재프레이밍.

---

# Part 1: 출발점 — 무엇이 문제인가

## 실무의 반복되는 루프

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

**문제의 핵심**: 작은 조직은 모델을 직접 학습시킬 리소스가 없고,
범용 모델은 우리 도메인에 완전히 맞지 않는다.

## 우리가 가진 것

| 자산 | 설명 |
|------|------|
| **N개의 frozen 모델** | 오픈소스 pretrained 모델들 (face detection, expression, AU, pose, quality, CLIP 등 14개) |
| **매일 생성되는 데이터** | 하루 2-3000건의 얼굴 촬영 비디오. 다른 연구 기관이 갖기 어려운 도메인 특화 데이터 |
| **기존 파이프라인** | visualpath 기반 14개 모듈 조합 + catalog_scoring (Fisher-weighted 21D matching) |

## 우리에게 없는 것

| 부재 | 영향 |
|------|------|
| **Annotation 리소스** | 매일 수천 건의 데이터를 라벨링할 인력/비용 없음 |
| **학습 인프라** | 대규모 모델 학습을 위한 GPU 클러스터 없음 |
| **대규모 팀** | ML 전문가 팀이 모델을 계속 개선할 구조 아님 |

## 목표

> **Annotation 없이, N개의 불완전한 frozen 모델의 조합을 통해,
> 어떤 단일 SOTA 모델보다 우리 도메인에서 나은 결과를 만들어내는 시스템.**

유일한 무기는 매일 쏟아지는 도메인 특화 데이터다.
이것을 처리하고 가공하기 어려운 조직 구조와 리소스를 가지고 있지만,
**시스템이 자율적으로 이 데이터를 활용**할 수 있다면 이야기가 달라진다.

---

# Part 2: 핵심 통찰의 전개

## 2.1 불확실한 분포로서의 모델 출력

기존 접근(catalog_scoring 포함)은 모델 출력을 **점 추정**(point estimate)으로 다룬다:
`happy = 0.8`, `AU12 = 3.2`, `yaw = 15°`.

그러나 각 모델의 출력은 본질적으로 **불확실한 분포**다:
- Expression 모델이 `happy = 0.8`이라고 할 때, 이것은 "확실히 happy"가 아니라
  "내 학습 데이터 기준으로, 이 입력이 happy일 likelihood가 0.8"이다.
- 다른 학습 데이터로 학습된 모델은 같은 입력에 다른 값을 낼 수 있다.

**각 모델의 출력 = 자신의 학습 경계(competence boundary) 내에서의 likelihood.**

## 2.2 Drift — 불확실한 분포들의 교차점으로의 이동

여러 모델의 불확실한 분포를 "힌트"로 삼아, **더 맞는 방향으로 drift**시킬 수 있다.

```
모델 A: "내 경험상, 이건 이쪽이다" (불확실, 60-95%)
모델 B: "내 경험상, 이건 저쪽이다" (불확실, 60-95%)
모델 C: "내 경험상, 이건 이쪽이다" (불확실, 60-95%)
                    ↓
    분포들의 교차/중첩 → "여기가 더 맞는 방향"이라는 drift signal
                    ↓
    drift를 따라가면 → 단일 모델로는 도달 못하는 지점에 수렴
```

이것은 단순 앙상블(답들의 평균)과 근본적으로 다르다:
- **앙상블**: 정적, 일회성, 입력의 범위 안에 갇힘
- **Drift**: 동적, 반복적, 방향성, **경계를 넘어서는** 이동

## 2.3 Bayesian Posterior Update로의 프레이밍

통계적으로 이 drift는 Bayesian posterior update로 자연스럽게 표현된다:

```
Prior:      "좋은 프레임"에 대한 초기 믿음
Observer 1: Likelihood₁ 관측 → Posterior₁ (belief 업데이트)
Observer 2: Likelihood₂ 관측 → Posterior₂ (belief 재업데이트)
...
Observer N: Likelihood_N 관측 → Posterior_N (최종 belief)
```

각 observer가 주는 것은 "내 관점에서 이 프레임은 이 정도로 그럴듯하다"는 likelihood이고,
이것들이 순차적으로 belief를 업데이트하면서 drift가 일어난다.

## 2.4 학습 경계(Competence Boundary)의 개념

각 frozen 모델은 자신의 학습 환경이 정의하는 **경계**를 가진다:

```
Wild 학습 얼굴 인식기:    야외 환경에 강함, 미세 특징 약함
Studio 학습 얼굴 인식기:  정밀도 높음, 환경 변화에 취약
981파크 실환경:           두 모델 모두의 학습 분포 밖
```

한 모델의 경계 안에서는 likelihood가 높고 신뢰할 수 있지만,
경계 밖에서는 불확실성이 급격히 증가한다.

**여러 모델의 likelihood 패턴은 3가지 정보를 동시에 제공한다:**

1. **본질에 대한 추정** — 여러 경계에서 공통으로 높으면 본질적으로 확실
2. **경계 자체의 지도** — 어떤 모델이 확신하고 어떤 모델이 불확실한가 → 샘플의 위치
3. **경계 확장의 방향** — 모든 모델이 불확실한 영역 → 아직 미개척, 학습해야 할 곳

```
         Wild 모델 경계
        ┌─────────────────┐
        │   ╔═══════╗     │
        │   ║ 겹침  ║     │    ← 여러 모델이 동의 = 본질에 가까움
        │   ║ 영역  ║     │
   ┌────│───╚═══════╝─────│────┐
   │    │                 │    │
   │    └─────────────────┘    │  Studio 모델 경계
   │                           │
   │     미개척 영역            │    ← 981파크 데이터가 여기를 채움
   │                           │
   └───────────────────────────┘
```

각 모델의 출력은 자기 경계에서의 likelihood로 다루어지고,
경계를 넓혀가는 과정에 활용된다.

## 2.5 Evidence Accumulation — 실시간 시스템에서의 경험

실시간 프레임에서 검출 모델을 적용하면, 환경이 변하면서 단일 모델의 결과가
일관되지 않을 때가 있다. 이때 **여러 번의 관측을 통해 evidence를 축적**하면
불확실성을 극복할 수 있다.

Action-Perception 문제에서도 observer의 evidence를 축적하면서
실제 문제가 명확해지는 방향으로 action에 대한 planning을 최적화할 수 있다.

이 경험이 VisualBind에서 만나면:
- **Perception**: observer들의 evidence 축적 → belief 업데이트
- **Action**: 어떤 observer를 다음에 실행할지 능동적 선택
- **Planning**: belief가 충분하면 → 다음 단계 trigger, 불충분하면 → 추가 관측

---

# Part 3: 같은 근본 질문에 대한 다른 관점의 모델들

## 의료 AI 사례와의 구조적 동치

의료 AI에서 오랜 문제:
- GT로 수집한 라벨들이 실제 오랜 기간 훈련받은 의사들이 작성해도
  의사들의 진단 편차가 너무 커서 단일 모델을 학습시키기 어렵다.
- 한 명은 환자 맥락을 알고, 다른 한 명은 방사선 이미지 전문가라서
  같은 증례에 대해 다른 결론을 낸다.

우리 문제:
- "좋은 초상화인가?"에 대해 AU 모델은 근육 움직임 관점에서,
  Pose 모델은 머리 각도 관점에서, Expression 모델은 감정 표현 관점에서 답한다.

**이 둘은 같은 문제다.**

근본 질문은 하나("양성인가?", "좋은 초상화인가?")이고,
각 모델/의사는 자기 관점에서 불완전하게 답하는 것.
출력 형식의 차이(확률 vs AU intensity vs 각도)는
**인코딩 방식의 차이**일 뿐, 본질적 차이가 아니다.

## 통합된 문제 정의

> **"하나의 근본 질문에 대해, 서로 다른 관점을 가진 불완전한 N개 모델의 답으로부터,
> 어떤 단일 모델보다 나은 본질적 이해에 도달할 수 있는가"**

이 문제는 도메인을 가리지 않는다:
- 의료: 여러 의사/진단 모델 → 진단
- 초상화: 여러 analyzer → 품질 판단
- 자율주행: 여러 센서/모델 → 상황 인식
- 콘텐츠 모더레이션: 여러 분류기 → 위험 판단
- 제조: 여러 검사 모듈 → 불량 판정

## 핵심 가정

> **"불완전한 관점들의 조합을 통해 본질을 이해할 수 있다"**

이 가정이 성립하는 조건:
각 관점이 **본질의 서로 다른 측면을 부분적으로나마 반영**하고 있어야 한다.
완전히 무관한 noise가 아니라 **부분적 진실**이라는 것이 전제.

---

# Part 4: Learning from Crowds — 기존 프레임워크와의 연결

## Dawid-Skene (1979)에서 시작된 연구 흐름

"Learning from Crowds"는 여러 noisy annotator의 출력으로부터
underlying truth를 추정하는 문제로, 오래 연구되어 왔다:

```
전통적 학습:
  Input → Model → Output
  Loss = |Output - GT|           ← GT가 하나, 확실

Learning from Crowds:
  Input → Model → Output
  Annotator₁ → label₁ (noisy, biased)
  Annotator₂ → label₂ (noisy, biased)
  Annotator₃ → label₃ (noisy, biased)

  Model은 "모든 annotator의 출력을 가장 잘 설명하는
  underlying truth"를 학습
  + 각 annotator의 bias/noise 패턴도 동시에 학습
```

주요 연구:
- **Dawid & Skene (1979)**: EM으로 annotator reliability + true label 동시 추정
- **Raykar et al. (2010)**: "Learning from Crowds" — classifier + annotator model 동시 학습
- **Rodrigues & Pereira (2018)**: Deep learning from crowds — DNN + crowd layer

## Teacher를 넘는 메커니즘

단순 distillation에서는 student가 teacher의 출력을 따라하므로,
**student의 추론 퍼포먼스가 teacher를 앞지르는 순간 역으로 되돌리는 힘**이 작용한다.

Learning from Crowds는 이 문제를 해결한다:

```
단순 distillation:
  Student = Teacher 따라하기
  → Student ≤ Teacher (상한이 teacher)

Learning from Crowds:
  Student = "Teacher들의 출력 패턴 뒤에 있는 truth 학습"
  + "각 Teacher의 bias 모델도 학습"

  → Teacher₃이 이 상황에서 자주 틀린다는 것을 학습
  → Teacher₁과 Teacher₂가 동의하면 더 신뢰
  → 어떤 단일 Teacher보다 나은 판단 가능
```

**핵심**: student가 teacher의 **출력**을 따라하는 게 아니라,
teacher들의 **출력 패턴 뒤에 있는 truth**를 학습한다.
Teacher의 bias를 명시적으로 모델링하기 때문에,
teacher가 틀리는 지점에서 student가 보정할 수 있다.

```
Teacher_i의 출력 = underlying truth + bias_i
```

bias_i = teacher_i의 학습 경계 밖에서의 체계적 오류 패턴.

## 우리가 기존 연구에 추가하는 것

기존 Learning from Crowds와의 비교:

| 기존 | 우리 |
|------|------|
| Human annotator (같은 질문, 이산 label) | Frozen model (같은 근본 질문, 연속 다차원 출력) |
| 정적 데이터셋 | 매일 2-3000건 data stream |
| Offline 학습 | Online/continual 학습 가능 |
| Annotator는 고정 | Observer 추가/제거 유연 |

"Crowds"와 "Bounded Experts"를 구분하는 것에 본질적 차이는 없다.
"경계 밖에서 noisy"는 bias의 구체적 원인을 설명한 것일 뿐,
수학적으로 다루는 방식은 같다.

기여할 수 있는 확장 지점:
1. **Continuous heterogeneous output** (이산 label → 연속 다차원 신호)
2. **Daily data stream** (정적 데이터셋 → 지속적 공급)
3. **Cross-modal crowd** (같은 modality의 annotator → 다른 modality의 모델)

---

# Part 5: 구체적 아키텍처 방향

## 스스로의 판단 — 내부 모델의 필요성

단순 조합(attention, voting, weighted average, Bayesian update)은
**observer 출력의 재조합**에 불과하다.
아무리 잘 조합해도 observer들이 가진 정보의 범위를 넘지 못한다.

시스템에 **스스로의 판단**이 필요하다:

```
[조합만 하는 접근]
Observer₁ ─┐
Observer₂ ─┤→ 조합 함수 → 결과 (입력의 범위 안에 갇힘)
Observer₃ ─┘

[스스로의 판단이 있는 접근]
Observer₁ ─┐
Observer₂ ─┤→ 내부 모델(자기 세계관) → 판단 (경계를 넘을 수 있음)
Observer₃ ─┘        ↑
                 축적된 경험
                 (2-3000건/일 × N일)
```

내부 모델이 자기만의 **latent representation**을 가지면:
- Observer들은 감각 기관이고, 판단은 시스템 자신이 내린다
- Observer가 틀렸을 때 "내 경험상 이 상황에서는 이 observer가 부정확했다"고 감지
- 이것은 attention(누구 말을 들을까)이 아닌 **자기 판단에 근거한 선택**

## DNN 기반 구현 방향

Classical Bayesian은 likelihood 함수를 수동 설계해야 한다.
DNN은 데이터에서 직접 배울 수 있으므로 더 실용적.

### 구조

```
Frame (입력)
    ↓
[Student Encoder] — backbone (ResNet/ViT, pretrained, fine-tunable)
    ↓
z (latent state) — "이 프레임의 본질" (시스템 스스로의 이해)
    ↓
[Teacher Head₁] → expression 예측  vs Teacher₁ 실제 출력 → bias₁ 학습
[Teacher Head₂] → AU 예측          vs Teacher₂ 실제 출력 → bias₂ 학습
[Teacher Head₃] → pose 예측        vs Teacher₃ 실제 출력 → bias₃ 학습
...
[Teacher Head_N] → quality 예측    vs Teacher_N 실제 출력 → bias_N 학습
    ↓
[Task Head] → downstream task (최종 판단, z에서 직접)
```

### 학습 목표

```
Loss = Σᵢ wᵢ · |Headᵢ(z) - (Teacherᵢ_Output - biasᵢ)|
       + regularization on z
       + temporal consistency
```

**bias를 빼는 것이 핵심**: `Teacherᵢ_Output = truth + biasᵢ`로 모델링.
Student는 truth를 학습하고, bias는 별도로 학습한다.
Teacher가 틀리는 부분은 bias로 흡수되므로 student는 teacher에 끌려가지 않는다.

### MoE와의 차이

MoE(Mixture of Experts)는 특정 문제에 대해 적절한 expert를 선택하고
신뢰를 올리는 메커니즘이다.

우리 문제는 다르다:
**불완전한 expert들을 데리고 청출어람할 수 있는 범용 모델을 학습시키는 task.**

MoE에서 expert는 이미 유능하고 routing만 필요하지만,
우리의 observer들은 각자의 경계 내에서만 부분적으로 유능하다.
Student가 이들의 한계를 넘어서야 한다.

---

# Part 6: 자기 개선 루프 — 특이점의 밑거름

## 온라인 환경 적응

단일 모델의 drifting을 잘 활용하면 **온라인 환경 적응적 기능**을 만들 수 있다.

```
Day 1:  검출기 95% (벤치마크 대비 gap)
        + expression 70% + AU 75% + pose 85%
             ↓
        2-3000건/일 → 수만 프레임
             ↓
        다수 observer의 고신뢰 합의 → pseudo ground truth
        (3개 독립 모델 동시 동의 → 동시 오류율 ≈ 0.0125%)
             ↓
        Student 모델 점진적 적응
             ↓
Day 30: Student 97%+ (도메인 특화)
        + 더 좋은 검출 → 더 좋은 expression/AU 입력 → 전체 향상
             ↓
        선순환
```

**매일 2-3000건의 데이터를 annotation 작업 없이 자율적으로 활용하여
점진적으로 개선되는 모델을 생산하는 시스템.**

이것은 다른 연구 기관이 갖기 어려운 장점(도메인 특화 daily data stream)을
조직의 한계(annotation 리소스 부재)에도 불구하고 활용할 수 있게 한다.

## 기존 파이프라인 대체

현재 visualpath에서 복잡하게 매뉴얼 정의한 구조
(14개 모듈 → DAG → 수동 가중치 → 규칙 기반 스코어링)를
**하나의 모델이 학습으로 대체**하는 것이 최종 목표.

```
현재:
  14개 frozen 모듈 → 수동 파이프라인 → Fisher 가중치 → catalog 매칭
  = 복잡, 수동, 정적

VisualBind 목표:
  14개 frozen 모듈 (teacher로서) → Student 모델 학습 → 단일 모델 판단
  = 자동, 적응적, 성장하는
```

Student가 성장하면:
- Cross-modal 관계를 자연스럽게 학습 ("웃을 때 포즈가 어떻게 변하는지")
- Teacher 조합의 수동 규칙으로 포착 못하는 비선형 관계를 DNN이 학습
- 도메인 데이터가 쌓일수록 981파크 특유 패턴에 특화

---

# Part 7: catalog_scoring과의 관계

catalog_scoring은 이미 VisualBind의 핵심 통찰을 증명하고 있다:

| catalog_scoring이 하는 것 | VisualBind가 확장하는 방향 |
|--------------------------|-------------------------|
| 14개 모듈의 21D 시그널 조합 | N개 모듈의 임의 차원 시그널 |
| Fisher ratio로 가중치 자동 계산 | DNN이 가중치를 데이터에서 학습 |
| 카테고리별 centroid 매칭 | Latent space에서 연속적 표현 |
| 7장 참조 이미지 필요 | Teacher 합의로 pseudo-label 자동 생산 |
| 정적 (한 번 계산하면 고정) | 매일 데이터로 성장 |
| 선형 가중 유클리디안 거리 | 비선형 cross-modal 관계 학습 |

catalog_scoring의 한계 = VisualBind가 풀어야 할 것:
- `SIGNAL_RANGES` 하드코딩
- Fisher = 선형 분리력만 측정
- 카테고리 사전 정의 필요
- 가중치가 데이터셋 전체에 고정

catalog_scoring은 VisualBind의 **초기 prior** 또는 **baseline으로 공존**할 수 있다.

---

# Part 8: 학술적 포지셔닝

## 기존 why-visualbind.md의 포지셔닝 → 재검토

| 기존 프레이밍 | 문제 | 재프레이밍 |
|-------------|------|----------|
| "합의는 레이블이다" | 합의는 레이블이 아님. 관측의 신뢰도이거나, truth 추정의 힌트 | "합의는 underlying truth을 향한 drift signal" |
| Binding Problem 메타포 | 영감의 원천이지 정당화 근거가 아님 | Learning from Crowds가 직접적 이론 프레임워크 |
| 범용 프레임워크 (자율주행, 의료, 스포츠) | 아직 초상화에서도 검증 안 됨 | 먼저 초상화에서 catalog_scoring을 넘긴 후 일반화 |
| "Few-shot prototype matching" | catalog_scoring의 재구현에 가까움 | "Annotation-free model improvement via multi-observer evidence" |
| Snorkel + Co-training + Contrastive | 각 구성요소가 이미 잘 알려져 있음 | Cross-modal crowd + continuous output + daily stream이 차별점 |

## 논문 방향

**제목 후보**:
"Learning from Model Crowds: Annotation-Free Domain Adaptation via Multi-Observer Evidence Accumulation"

**핵심 contribution**:
1. 같은 근본 질문에 대한 heterogeneous frozen model 출력을 crowd annotation으로 재해석
2. Teacher bias 명시적 모델링으로 student가 teacher를 초월하는 메커니즘
3. Daily data stream 기반 continual self-improvement 실증
4. 981파크 실데이터에서 catalog_scoring 대비 성능 향상 곡선

**대상 학회**:
- FG (Face and Gesture) — 도메인 특화 실증
- ECCV/CVPR Workshop — 방법론 + 실증
- NeurIPS (ML 트랙) — 이론적 bound + 대규모 실증이 있을 경우

---

# Part 9: 전문가 리뷰와의 연결

10명의 전문가 리뷰(`visualbind-expert-review-unified.md`)에서 지적된 문제들 중,
이 재프레이밍에서 해소되는 것과 여전히 유효한 것:

## 재프레이밍으로 해소되는 문제

| 기존 지적 | 해소 방식 |
|-----------|----------|
| Snorkel과 구조적 동치 | Learning from Crowds로 명시적 포지셔닝. 연속 출력 + cross-modal이 차별점 |
| "합의 = 레이블" 프레이밍 문제 | 폐기. "합의 = underlying truth으로의 drift signal" |
| Linear projection의 conjunction 불가 | DNN으로 전환 (비선형 학습) |
| 뇌신경학적 메타포 과도 | 제거. Learning from Crowds가 이론 프레임워크 |
| 범용 프레임워크 주장의 미검증 | 먼저 초상화에서 검증. 범용성은 결과로 보여줌 |

## 여전히 유효한 문제 (구현 시 해결 필요)

| # | 문제 | 관련성 |
|---|------|--------|
| 1 | Both-low bias (`a*b`) | DNN 방식에서는 직접 해당 안 되나, teacher 출력의 정규화 문제로 변환 |
| 2 | `flat_vector` zero-padding | DNN 입력 설계에서 해결 |
| 8 | Softmax double-application | Teacher 출력 전처리에서 해결 |
| 7 | Temporal dynamics | Student 학습 시 temporal consistency loss로 해결 가능 |

---

# Part 10: 열린 질문

## 설계 결정이 필요한 것

| 질문 | 선택지 | 고려사항 |
|------|--------|---------|
| Student backbone | ResNet / ViT / lightweight MLP | 조직의 inference 리소스에 맞춰야 함 |
| 학습 빈도 | 야간 배치 / 주간 / 실시간 | 데이터 축적 속도 vs 리소스 |
| Cold start | catalog_scoring warm start / 순수 자가감독 / 초기 축적 후 시작 | 초기 성능 vs 순수성 |
| Teacher head 구조 | 공유 backbone + 독립 head / 완전 독립 | 파라미터 효율 vs 유연성 |
| Bias 모델링 방식 | 명시적 bias head / implicit (DNN이 자동 학습) | 해석가능성 vs 단순성 |
| 기존 PoC 코드 | 전면 재작성 / 부분 재활용 | PoC의 Collect 단계는 재활용 가능 |

## 검증이 필요한 가정

| 가정 | 검증 방법 |
|------|----------|
| 다수 observer 합의 = 높은 정확도 pseudo-label | 합의 샘플을 소량 수동 검증 |
| Student가 teacher를 넘을 수 있다 | Day 1 vs Day 30 성능 곡선 측정 |
| Daily data가 충분한 diversity를 제공한다 | 데이터 분포 분석 (계절, 시간대, 고객 다양성) |
| 도메인 특화가 범용 SOTA를 이긴다 | 실환경 A/B test |

## 다음 단계

1. **이 문서를 기반으로 why-visualbind.md 재작성** — 기존 초안 대체
2. **아키텍처 설계** — DNN 구조, 학습 목표, 데이터 파이프라인 구체화
3. **최소 실험** — 기존 14개 모듈 출력 데이터로 Learning from Crowds 검증
4. **기존 PoC 코드 처리** — 재활용 가능 부분 식별, 나머지 재작성
