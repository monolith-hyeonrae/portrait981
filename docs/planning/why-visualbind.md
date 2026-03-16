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
| **기존 파이프라인** | visualpath 기반 14개 모듈 조합 + catalog_scoring (Fisher-weighted 21D matching) |

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

---

## 핵심 통찰: Threshold가 만드는 Weak Classifier

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

이것은 운용에서 이미 일어나고 있는 현실이다.

### Noise의 이중 구조

```
모듈 출력 = underlying truth + model_bias + threshold_bias
```

- **model_bias**: 모델 자체의 학습 경계에서 오는 불완전함
- **threshold_bias**: 사람이 임의로 정한 임계값에서 오는 불완전함

수동 threshold는 "완전히 틀리지는 않다."
이것은 Learning from Crowds에서 "annotator가 random보다 낫다"는 전제와 정확히 대응한다.

### 현재 시스템의 한계

수동 threshold 기반 AND 결합이 가진 문제:

1. **Threshold가 최적인지 알 수 없다** — 경험과 직관으로 정함
2. **AND만 사용** — OR이나 weighted 조합이 나을 수 있음
3. **축 정렬 직사각형(axis-aligned rectangle)에 갇힘** — "웃을 때 고개가 약간 기울어야 자연스럽다"는 비선형 관계를 표현 불가
4. **정보 손실** — happy = 0.59와 0.61이 threshold 0.6에 의해 완전히 다른 결과가 됨
5. **새 버킷 추가 시 매뉴얼 작업** — threshold 정의, 참조 이미지 촬영, 가중치 조정

**VisualBind는 이 수동 과정을 데이터에서 자동으로 학습하여 대체한다.**

---

## Learning from Crowds — 이론적 프레이밍

### Dawid-Skene (1979)에서 시작된 연구 흐름

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
- **Snorkel (Ratner et al., 2017)**: Data Programming — 이산 labeling function의 조합

### 우리 시스템이 이 구조와 동치인 이유

```
Crowds 문제:            우리 시스템:
━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Annotator               Frozen 모듈 + threshold
Binary label            ✓/✗ (threshold 통과 여부)
True label              "이 프레임이 이 버킷에 정말 적합한가?"
Annotator bias          model_bias + threshold_bias
Task                    버킷 적합성 판단
```

Threshold를 적용한 모듈 = Snorkel의 Labeling Function = Dawid-Skene의 Annotator.

### 두 레이어의 공존

| 레이어 | 설명 | 이론적 프레임 |
|--------|------|-------------|
| Threshold 후 (binary vote) | 같은 질문의 weak classifier | Learning from Crowds (직접 적용) |
| Threshold 전 (연속 출력) | 풍부한 정보, 비선형 관계 | Multi-modal fusion |

VisualBind는 **두 레이어를 모두 활용**한다:
- Binary vote → pseudo-label (학습 목표, crowds 이론 적용)
- 연속 출력 → feature (학습 입력, threshold 정보 손실 복구)

---

## Student가 Teacher를 넘는 메커니즘

### 단순 distillation의 한계

```
Student = Teacher 따라하기
→ Student ≤ Teacher (상한이 teacher)
```

### Learning from Crowds의 구조

```
Teacher_i의 출력 = underlying truth + bias_i

Student가 학습하는 것:
  (1) 모든 teacher의 출력 패턴 뒤에 있는 truth
  (2) 각 teacher의 bias 패턴

→ Teacher₃이 이 상황에서 자주 틀린다는 것을 학습
→ Teacher₁과 Teacher₂가 동의하면 더 신뢰
→ 어떤 단일 Teacher보다 나은 판단 가능
```

핵심: Student가 teacher의 **출력**을 따라하는 게 아니라,
teacher들의 **출력 패턴 뒤에 있는 truth**를 학습한다.
Teacher가 틀리는 부분은 bias로 흡수되므로 student는 teacher에 끌려가지 않는다.

### MoE와의 차이

MoE(Mixture of Experts)는 적절한 expert를 선택하는 메커니즘이다.
우리 문제는 다르다: **불완전한 expert들을 데리고 청출어람할 수 있는 모델을 학습시키는 task.**
MoE에서 expert는 이미 유능하고 routing만 필요하지만,
우리의 observer들은 각자의 경계 내에서만 부분적으로 유능하다.

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

### Concept Drift = 방어 대상이 아니라 핵심 기능

일반적 ML 시스템에서 concept drift는 방어해야 할 문제다.
하지만 우리 시스템에서는 **의도적으로 활용**해야 한다.

```
시즌 1: warm_smile, cool_gaze 중심 수집
  → 디자인팀: "다음 시즌은 눈감은 장면이 필요해"
시즌 2: + eyes_closed (새 버킷, 50건 few-shot)
  → 디자인팀: "측면 portrait 비중을 높여줘"
시즌 3: lateral 버킷 가중치 상향
```

다소 주관적인 기준이 시즌마다 바뀌는 것이 우리 도메인의 본질이다.
catalog 메커니즘과 momentbank로 다양성을 저장하고,
**VisualBind의 multi-bucket + few-shot이 기준 변경에 빠르게 적응하는 메커니즘**이 된다.

```
momentbank  — 다양성 저장 (모든 가능성을 보존)
catalog     — 현재 시즌의 수집 기준 정의 ("무엇을 찾을지")
visualbind  — 기준 변경에 빠르게 적응하는 학습 ("어떻게 찾을지")
```

### 상용 가치

| 관점 | 가치 |
|------|------|
| 학술 | "known mechanisms in new context" — FG/Workshop급 |
| **상용** | **annotation-free drift adaptation — 모든 frozen 모델 기반 시스템에 적용 가능** |

VisualBind의 본질:
> **여러 frozen inference의 noise를 고려한 underlying truth를 학습을 통해
> 통합된 context를 표현하는 단일 inference를 만들어내고,
> 현장 데이터에 적응하는 과정을 유연하게 만들어냄으로써 model drift 문제를 해결한다.**

---

## catalog_scoring과의 관계

catalog_scoring은 사실상 **VisualBind의 수동 버전**이다:

| catalog_scoring | VisualBind |
|----------------|------------|
| 14개 모듈의 21D signal 조합 | N개 모듈의 임의 차원 signal |
| Fisher ratio로 가중치 계산 | DNN이 가중치를 데이터에서 학습 |
| 카테고리별 centroid matching (선형) | Latent space에서 비선형 boundary |
| 7장 참조 이미지 + 수동 threshold | Daily data에서 자동 pseudo-label |
| 새 카테고리 = threshold 정의 + 참조 촬영 | 새 버킷 = 50건 few-shot transfer |
| 정적 (한 번 계산하면 고정) | 매일 데이터로 성장 |

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

**"VisualBind: Annotation-Free Domain Adaptation via Cross-Task Crowd Distillation"**

### 선행연구 계보

VisualBind는 4개의 연구 흐름이 교차하는 지점에 있다.

#### 1. Learning from Crowds (이론적 기반)

여러 noisy annotator의 출력으로부터 underlying truth를 추정하는 연구 흐름.

- **Dawid & Skene (1979)**: EM으로 annotator reliability + true label 동시 추정
- **Raykar et al. (2010)**: classifier + annotator model 동시 학습
- **Rodrigues & Pereira (2018)**: Deep learning from crowds — DNN + crowd layer

> VisualBind에서: threshold 적용된 frozen 모듈 = noisy annotator.
> Dawid-Skene으로 모듈별 reliability 자동 추정.
> **차별점**: 이산 label → 연속 heterogeneous output 확장.

#### 2. Snorkel / Data Programming (pseudo-label 생성)

Noisy labeling function들의 조합으로 annotation 없이 학습하는 프레임워크.

- **Ratner et al. (2017)**: Snorkel — labeling function의 accuracy/correlation 자동 추정, GT 불필요
- **Ratner et al. (2019)**: Snorkel MeTaL — 계층적 multi-task에 대한 weak supervision 확장

> VisualBind에서: threshold 적용된 모듈 = labeling function.
> **차별점**: binary LF만 → threshold 전 연속 출력도 활용 (dual-mode).
> Snorkel MeTaL의 multi-task 구조와 multi-bucket 구조가 대응.

#### 3. Multi-Teacher Knowledge Distillation (아키텍처)

여러 teacher 모델의 지식을 단일 student로 압축하는 연구 흐름.

- **Hinton et al. (2015)**: Knowledge Distillation — soft target으로 학습
- **Zuchniak et al. (2023)**: Multi-teacher KD as ensemble compression — student가 개별 teacher를 초과
- **NeurIPS (2024)**: Ensemble-Then-Distill — ensemble pseudo-label → student distillation

> VisualBind에서: 14개 frozen teacher → student.
> **차별점**: homogeneous teacher(같은 task) → **cross-task teacher(다른 task)**.
> soft(연속 재구성) + hard(binary vote) dual-mode loss.

#### 4. Foundation Model Distillation (end-to-end 비전)

대형 모델의 지식을 자동 annotation으로 소형 모델에 전이하는 실무적 접근.

- **Xiao et al. (2024)**: Florence-2 (CVPR 2024) — FLD-5B(54억 자동 생성 annotation)으로 unified vision model 학습
- **Roboflow (2023)**: Autodistill — foundation model → pseudo-label → target model (YOLOv8 등)
- **Lu et al. (2025)**: Single-model multi-task face analysis — 검출+인식+속성을 단일 모델로

> VisualBind Phase 2에서: 14 Teachers → pseudo-label (annotation 공장) → Student CNN (end-to-end).
> **차별점**: 단일 foundation model이 아닌 **cross-task crowds 합의**로 pseudo-label 생성.
> 대규모 인프라 불필요 (GPU 1개 + 도메인 데이터).

### 선행연구 대비 차별점 요약

| 선행연구 | VisualBind 차별점 |
|---------|------------------|
| Learning from Crowds | 이산 label → **연속 heterogeneous output** |
| Snorkel | discrete LF → **continuous output + binary vote dual-mode** |
| Multi-Teacher KD | homogeneous → **cross-task** teachers |
| Autodistill | 단일 teacher → **multi-teacher crowds 합의** |
| Florence-2 | 대규모 인프라 → **GPU 1개 + 도메인 데이터** |
| Face Multi-Task | labeled data → **annotation-free pseudo-label** |

### 논문 Contribution

**Phase 1 (Signal-Level Distillation):**
1. **Frozen 모델 + threshold 운용 = Learning from Crowds**: 실무에서 이미 crowds 구조로 운용되고 있음을 보임. Dawid-Skene만으로도 수동 AND 조합 대비 개선 (Exp 0)
2. **Binary vote + 연속 출력 dual-mode 학습**: Snorkel의 이산 LF를 연속 확장. threshold 정보 손실을 복구하면서 crowds 이론 활용 (Exp 1-3)
3. **Multi-bucket shared representation + few-shot transfer**: 새 버킷을 수동 작업 없이 50건 내외로 추가 가능
4. **Daily data stream 기반 continual self-improvement + drift adaptation 실증**

**Phase 2 (End-to-End Distillation):**
5. **Cross-Task Multi-Teacher → 단일 End-to-End 모델**: 14개 specialist 파이프라인을 annotation-free pseudo-label로 단일 CNN에 distill. Inference 비용 대폭 감소.
6. **Teacher를 annotation 공장으로 전환**: 학습 시에만 Teacher 사용, inference에서 제거. Autodistill 원리를 cross-task crowds 합의로 확장.

### Novelty에 대한 정직한 포지셔닝

개별 메커니즘은 기존에 알려져 있다. Novelty는 **적용 맥락의 교차점**에 있다:
- 이산 → 연속 (crowds / Snorkel 확장)
- 동종 → 이종 (cross-task frozen models)
- 정적 → 온라인 (daily data stream)
- binary → dual-mode (hard vote + soft continuous)
- signal-level → end-to-end (Phase 1 → Phase 2 진화)

Phase 2까지 포함하면 **"annotation-free cross-task crowd distillation"**이라는
기존에 다뤄지지 않은 조합이 되며, 논문 임팩트가 크게 상승한다.

### 대상 학회

| 범위 | 학회 | 필요 완성도 |
|------|------|-----------|
| Phase 1 (signal-level) | FG / ECCV Workshop | Exp 0-5 + 내부 데이터 |
| Phase 1 + Phase 2 시작 | CVPR / ECCV Main | Exp 0-7 + ablation + 정량 비교 |
| Phase 1-2 완성 + 이론 | NeurIPS / ICML | 위 + N-observer bound + scaling law |

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
[9] Xiao, B. et al. (2024). Florence-2: Advancing a Unified Representation for a
    Variety of Vision Tasks. CVPR 2024.
[10] Roboflow (2023). Autodistill: Images to Inference with No Labeling.
     github.com/autodistill/autodistill.
[11] Lu, X. et al. (2025). A Single-Model Multi-Task Method for Face Recognition and
     Face Attribute Recognition. IET Image Processing.
```

---

## 의료 AI와의 구조적 동치

의료 AI에서 오랜 문제:
- 여러 의사/검사/모델이 같은 환자에 대해 다른 판단을 내림
- Clinical Decision Rule (CDR)은 정확히 우리 구조:
  각 검사 항목이 threshold를 통해 같은 임상 질문에 대한 weak vote로 변환

```
Ottawa Ankle Rule (골절 의심?):
  복사뼈 압통? → ✓/✗
  체중 부하 가능? → ✓/✗
  뼈 끝 압통? → ✓/✗
  → 하나라도 ✓ → X-ray 필요
```

각 항목은 다른 것을 측정하지만, threshold를 통해 같은 질문의 weak classifier가 된다.
VisualBind의 구조와 정확히 동일하다.

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
visualbind는 이 발판 위에서 자라서, 궁극적으로는 visualpath 전체를 단일 모델로 distill한다.

```
Phase 1: visualpath → 18D signal → Student → 판단 개선 (Teacher 조합 개선)
Phase 2: visualpath → pseudo-label (학습 시만) → Student CNN → 직접 판단 (end-to-end)
Phase 3: Student가 도메인 foundation model → visualpath 퇴역
```

**visualpath 없이는 visualbind가 시작할 수 없고,
visualbind가 성숙하면 visualpath가 필요 없어진다.**

---

## 다음 단계

구체적인 아키텍처, MVP 경로, 실험 설계, 안전장치는 `how-visualbind.md`에서 다룬다.
