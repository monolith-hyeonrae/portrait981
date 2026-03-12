# 왜 visualbind인가

> "레이블 없으면 학습 못 하는 거 아니야?"

이 질문은 자연스럽고, 전통적으로는 맞습니다. 이 문서는 왜 우리가 레이블 없이 학습하는 시스템을 만들어야 했는지, 그리고 그것이 어떻게 **범용 프레임워크**가 될 수 있는지를 설명합니다.

---

## 배경: Binding Problem

신경과학에서 **Binding Problem**(결합 문제)은 뇌가 분리된 감각 신호들—색, 형태, 움직임, 소리—을 어떻게 하나의 통합된 지각으로 결합하는가에 대한 근본적 질문입니다.

빨간 공이 날아옵니다. V1은 윤곽을, V4는 색을, MT는 움직임을 처리합니다. 이 분리된 신호들이 어떻게 "빨간 공이 날아온다"는 하나의 인식이 됩니까? 뇌는 이 신호들 사이의 **시간적 동기화**(temporal synchrony)와 **공간적 일관성**(spatial coherence)을 이용해 결합합니다.

우리의 시스템은 정확히 같은 문제를 풀고 있습니다.

---

## 우리가 가진 것

portrait981의 momentscan은 이미 14개의 분석 모듈을 동시에 실행합니다:

```
face.detect → face.expression, face.au, head.pose, face.parse, face.quality, ...
body.pose, hand.gesture, frame.quality, ...
```

한 프레임에서 이 모듈들이 동시에 말합니다:

| 모듈 | 출력 예시 |
|------|----------|
| face.expression | `{happy: 0.8, neutral: 0.15, ...}` |
| face.au | `{AU6: 0.7, AU12: 0.9, AU1: 0.1}` |
| head.pose | `{yaw: -15°, pitch: 5°, roll: 2°}` |
| face.quality | `{blur: 12.3, exposure: 142}` |
| body.pose | `{left_arm_angle: 45°, torso_lean: 3°}` |

이것들은 같은 순간, 같은 사람에 대한 **서로 다른 관점의 관찰**입니다. 뇌의 V1, V4, MT가 같은 물체를 다른 방식으로 보는 것과 동일합니다.

---

## 우리가 원하는 것

이 다중 관찰로부터 **"이 사람의 이 순간"을 하나의 벡터로 표현**하고 싶습니다. 이 벡터(임베딩)가 있으면:

- **유사한 순간 검색**: "이 미소와 비슷한 다른 순간들"
- **품질 판단**: 어떤 프레임이 이 사람의 가장 좋은 초상화인지
- **범주 구성**: 자연스러운 표정 군집 발견
- **시간적 요약**: 긴 영상에서 대표 순간 선별

문제는, 이 임베딩을 학습하려면 보통 **레이블이 필요합니다**.

---

## 레이블이 없는 이유

### 1. 정의할 수 없다

"좋은 초상화"의 정의는 무엇입니까? 웃는 얼굴? 자연스러운 표정? 정면? 측면이 더 매력적인 사람도 있습니다. AU6+AU12(뒤셴 미소)가 높으면 좋은 표정입니까? 맥락에 따라 다릅니다.

우리가 판단하고 싶은 것은 **단일 축으로 환원할 수 없는 다차원적 품질**입니다. 레이블을 만들려면 이 다차원을 하나의 숫자로 압축해야 하는데, 그 압축 자체가 정보 손실입니다.

### 2. 스케일하지 않는다

981파크는 매일 수천 명의 고객을 촬영합니다. 프레임 단위로 레이블을 다는 것은 물리적으로 불가능합니다. 설령 가능하더라도, 레이블러의 주관이 개입하면 일관성을 보장할 수 없습니다.

### 3. 이미 답을 가지고 있다

14개 모듈이 프레임마다 내놓는 출력은 이미 **부분적인 답**입니다. AU6+AU12가 높고, expression이 happy이고, face.quality가 좋으면—이 세 모듈이 **합의**하고 있습니다. 이 합의 자체가 supervision 신호입니다.

---

## 핵심 통찰: 합의는 레이블이다

여러 독립적 관찰자가 동시에 같은 방향을 가리킬 때, 그 합의는 어떤 개별 레이블보다 신뢰할 수 있습니다.

```
face.expression:  happy 0.8     ──┐
face.au:          AU6↑ AU12↑    ──┼── 합의: "진짜 미소"  → strong signal
head.pose:        frontal        ──┘

face.expression:  happy 0.6     ──┐
face.au:          AU6↓ AU12↑    ──┼── 불일치: "사회적 미소?"  → noisy, 버림
head.pose:        yaw 40°        ──┘
```

이것이 **Weak Supervision**(약한 감독)의 원리입니다. 개별 모듈의 출력은 noisy하지만, 여러 모듈의 **합의 패턴**은 robust합니다.

그리고 이 원리는 **얼굴 분석에만 국한되지 않습니다**.

---

## 일반화: 왜 이것이 프레임워크여야 하는가

### 같은 구조, 다른 도메인

| 도메인 | 분석 모듈들 (관찰자) | 결합 대상 | 학습할 임베딩 |
|--------|---------------------|----------|-------------|
| **초상화** | expression, AU, head pose, quality | 사람의 순간 | Portrait Embedding |
| **자율주행** | object detector, lane detector, depth estimator | 도로 장면 | Scene Embedding |
| **의료영상** | lesion detector, texture analyzer, shape classifier | 병변 영역 | Pathology Embedding |
| **스포츠** | pose estimator, ball tracker, formation analyzer | 선수 동작 | Action Embedding |
| **제조** | surface inspector, dimension checker, color analyzer | 제품 상태 | Defect Embedding |

모든 도메인에서 패턴이 동일합니다:

1. **여러 분석기가 같은 대상을 다른 관점으로 본다**
2. **개별 출력은 noisy하지만 합의는 robust하다**
3. **합의를 supervision 삼아 통합 표상을 학습한다**

바뀌는 것은 분석기와 백본뿐입니다. **합의 수집 → 신뢰도 계산 → 대조 학습**이라는 골격은 동일합니다.

### visualpath가 추론을 범용화했듯이

visualpath는 "여러 분석기를 DAG로 엮어 실행"하는 문제를 범용 프레임워크로 만들었습니다. 그 덕분에 momentscan뿐 아니라 어떤 도메인이든 분석 파이프라인을 선언적으로 구성할 수 있게 되었습니다.

visualbind는 그 다음 단계입니다. "여러 분석기의 출력을 모아 하나의 표상을 학습"하는 문제를 범용 프레임워크로 만듭니다.

```
visualpath: Module[] → DAG → Observation[]     (추론)
visualbind: Observation[] → Agreement → Embedding  (학습)
```

visualpath가 없었다면 visualbind도 없습니다. visualpath가 제공하는 `Module`과 `Observation`이라는 추상이 visualbind의 입력 인터페이스가 됩니다.

---

## 이름에 대하여

### 왜 "bind"인가

신경과학의 Binding Problem에서 가져왔습니다. 분리된 감각 신호를 하나의 통합된 지각으로 **결합**하는 문제—이것이 정확히 visualbind가 하는 일입니다.

- 여러 Analyzer의 출력(분리된 신호)을
- 하나의 Embedding(통합된 표상)으로
- 합의 기반으로 결합(bind)합니다

### 왜 "learn"이 아닌가

"learn"은 너무 넓습니다. 이 시스템의 본질은 학습 자체가 아니라 **결합**입니다. 학습은 결합의 수단이지 목적이 아닙니다. 목적은 분리된 관찰들을 하나의 표상으로 묶는 것입니다.

### 프로그래밍 문맥에서의 "bind"

`bind()`는 소켓 프로그래밍, 함수 바인딩 등에서 빈번하게 사용됩니다. 혼동을 피하기 위해:

- **패키지 이름은 항상 `visualbind` 풀네임**으로 사용합니다
- **내부 API에서 `bind`를 단독 동사로 사용하지 않습니다**
- 대신 도메인 특화 용어를 사용합니다: `fuse`, `agree`, `pair`, `encode`

```python
# ✗ 혼동 가능
from visualbind import bind, Binder

# ✓ 명확한 도메인 용어
from visualbind.fusion import HintFusion
from visualbind.agreement import AgreementMap
from visualbind.pairing import ContrastivePair
from visualbind.encoding import EngramTrainer
```

---

## 설계 원칙

visualpath에서 확립한 원칙을 계승하되, 학습 도메인에 맞게 확장합니다.

### 계승하는 원칙

| 원칙 | visualpath에서 | visualbind에서 |
|------|---------------|---------------|
| **선언적 > 명령적** | NodeSpec, ErrorPolicy | HintSpec, AgreementPolicy |
| **경고 > 에러** | 검증은 경고, 실행은 진행 | hint 불일치는 스킵, 학습은 진행 |
| **투명 > 명시** | 격리가 Module 인터페이스 뒤에 숨음 | 합의 계산이 학습 루프 뒤에 숨음 |
| **조합 > 상속** | Capabilities as metadata | AgreementStrategy as composable |

### 새로 추가하는 원칙

| 원칙 | 설명 |
|------|------|
| **합의 > 개별** | 단일 모듈의 출력을 직접 쓰지 않고, 복수 모듈의 합의만 신뢰 |
| **soft > hard** | 이진 판단 대신 연속적 신뢰도 분포 사용 |
| **버리기 > 억지** | noisy한 샘플을 억지로 쓰지 않고 과감히 제외 |
| **분포 > 점수** | 단일 스칼라 대신 분포 전체를 보존하고 매칭 |

---

## 무엇을 만드는가 (고수준 개요)

visualbind는 네 개의 핵심 단계로 구성됩니다:

```
┌─────────────────────────────────────────────────────────┐
│                    visualbind                            │
│                                                          │
│  ① Fusion      여러 Observation을 수집하고 정규화        │
│       │                                                  │
│  ② Agreement   hint 간 합의 패턴을 계산                  │
│       │                                                  │
│  ③ Pairing     합의 기반으로 contrastive pair 구성       │
│       │                                                  │
│  ④ Encoding    soft contrastive loss로 임베딩 학습       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### ① Fusion: 신호 수집

visualpath의 Observation을 hint로 재해석합니다. 각 Observation의 `signals` dict가 hint의 원천입니다.

```python
# visualpath Observation (이미 존재)
Observation(
    source="face.expression",
    signals={"happy": 0.8, "neutral": 0.15, "sad": 0.05},
    ...
)

# visualbind가 이것을 hint로 수집
HintVector(
    source="face.expression",
    values=np.array([0.8, 0.15, 0.05]),  # 정규화된 벡터
    confidence=0.92,
)
```

### ② Agreement: 합의 계산

같은 프레임에 대한 여러 hint가 일관된 방향을 가리키는지 측정합니다.

- **강한 합의**: expression=happy, AU6↑AU12↑, quality=good → 이 프레임은 "좋은 미소" (positive sample 후보)
- **강한 불일치**: expression=happy, AU6↓AU12↓ → 모순 (제외)
- **약한 신호**: 어느 쪽도 확실하지 않음 → 무시

### ③ Pairing: 대조 쌍 구성

합의가 강한 프레임끼리 positive pair, 합의 방향이 반대인 프레임끼리 negative pair를 구성합니다. 중간 영역은 버립니다.

### ④ Encoding: 임베딩 학습

구성된 pair로 contrastive learning을 수행합니다. 목표는 합의 패턴을 반영하는 저차원 임베딩 공간을 만드는 것입니다.

---

## visualpath와의 관계

```
visualbase (I/O)
  → visualpath (추론)
      → visualbind (학습)
```

visualbind는 visualpath의 **소비자**입니다:

- visualpath의 `Module` 인터페이스를 사용해 분석을 실행하고
- visualpath의 `Observation` 타입을 hint로 수집하고
- visualpath의 `FlowGraph`가 이미 해결한 의존성/실행 문제를 그대로 활용합니다

visualbind는 visualpath를 **수정하지 않습니다**. 기존 추론 파이프라인 위에 학습 계층을 **추가**할 뿐입니다.

---

## 학술적 맥락

이 접근은 기존 연구들과 다음과 같은 관계에 있습니다:

### Weak Supervision / Data Programming
Snorkel(2016)이 개척한 **labeling function의 조합**과 유사하지만, 우리의 labeling function은 사전 학습된 분석 모듈(Analyzer)입니다. Snorkel이 텍스트에 휴리스틱을 적용했다면, visualbind는 영상에 ML 모듈의 출력을 적용합니다.

### Multi-View Learning
같은 대상을 여러 관점(view)에서 보고 통합하는 **multi-view learning**의 프레임워크에 속합니다. 다만 전통적 multi-view가 같은 데이터의 다른 변환(augmentation)을 사용하는 반면, 우리는 **서로 다른 모델의 출력**을 view로 사용합니다.

### Contrastive Learning with Noisy Supervision
SimCLR/MoCo 계열의 self-supervised contrastive learning과 구조적으로 유사하지만, pair 구성을 augmentation이 아닌 **모듈 합의**로 수행합니다. noise-robust contrastive loss 연구(Li et al., 2022; Ghosh & Lan, 2021)의 성과를 활용합니다.

### Knowledge Distillation & Ensemble
여러 teacher 모델의 출력으로 student를 학습시키는 knowledge distillation과도 관련됩니다. 다만 teacher 모델들이 서로 다른 과제(expression vs AU vs pose)를 수행한다는 점에서, 전통적 ensemble distillation(같은 과제의 복수 모델)과 다릅니다.

### 차별점

기존 접근들과 visualbind의 핵심 차별점은 **프레임워크 수준의 범용성**입니다:

- Snorkel은 labeling function을 수동으로 작성합니다. visualbind는 이미 존재하는 Analyzer의 출력을 자동으로 수집합니다.
- Multi-view learning은 view 설계가 도메인 특화됩니다. visualbind는 visualpath의 Module 인터페이스를 통해 view를 플러그인처럼 교체합니다.
- 기존 contrastive learning은 pair 구성이 augmentation에 고정됩니다. visualbind는 agreement policy를 선언적으로 교체할 수 있습니다.

visualpath가 "분석기를 조합하는 방법"을 범용화했듯이, visualbind는 "분석기의 출력을 학습에 활용하는 방법"을 범용화합니다.

---

## 결정 근거 요약

| 결정 | 대안 | 선택 이유 |
|------|------|----------|
| 범용 패키지로 분리 | momentscan 내부 구현 | 도메인 무관한 패턴이므로 재사용 가능해야 함 |
| Binding Problem 메타포 | "learn", "fuse", "ensemble" | 핵심이 학습이 아니라 다중 신호의 결합이므로 |
| Observation을 hint로 재해석 | 별도 hint 타입 정의 | visualpath 인터페이스를 그대로 활용, 새 추상 최소화 |
| 합의 기반 pair 구성 | augmentation 기반, 레이블 수집 | 이미 있는 모듈 출력을 최대 활용, 레이블 비용 제거 |
| soft contrastive | hard triplet, classification | 합의 강도가 연속값이므로 soft label이 자연스러움 |
| Agreement 중간 영역 제외 | 전부 사용 | noisy sample 제거가 정확도에 결정적 (less is more) |
| 패키지 내 "bind" 단독 사용 금지 | 자유롭게 사용 | 소켓/함수 bind와의 혼동 방지 |

---

## 다음 단계

구체적인 아키텍처 설계, 모듈 구조, API 명세는 `how-visualbind.md`에서 다룹니다.
