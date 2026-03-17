# visualbind — 어떻게 풀 것인가

> `why-visualbind.md`에서 다룬 문제의식을 VisualBind가 어떤 설계로 해결하는지,
> 아키텍처, 기술 스택, MVP 경로, 실험 설계, 안전장치를 정리.

---

# Part 0: 아키텍처 개요

## 핵심 아이디어

```
학습 시: Raw Image → [14 Teachers] → 25D → crowds 합의 → pseudo-label (감독 신호)
         Raw Image → [Student backbone] → bucket 예측 → Loss(예측, pseudo-label)

추론 시: Raw Image → [Student backbone만] → bucket 예측
         14개 Teacher 실행 불필요!
```

**14-model inference를 1-model inference로 교체**하는 것이 핵심 가치.
25D Teacher 출력은 학습 시 supervision hint이지, Student의 입력이 아니다.

## 전체 흐름

```
                         ┌─────────────────────────────────────────────────┐
                         │              Raw Video Frame                    │
                         └───────────┬────────────────────┬───────────────┘
                                     │                    │
                          [학습 시에만]                [항상]
                                     │                    │
                    ┌────────────────▼────────────┐       │
                    │     14 Frozen Teachers       │       │
                    │  face.detect, face.au,       │       │
                    │  face.expr, head.pose, ...   │       │
                    └────────────┬─────────────────┘       │
                                 │                         │
                                 ▼                         ▼
                    ┌────────────────────┐    ┌──────────────────────┐
                    │   25D Signal       │    │   Vision Student     │
                    │   → threshold      │    │   (MobileNetV3       │
                    │   → binary votes   │    │    + linear heads)   │
                    │   → crowds 합의    │    │                      │
                    │   → pseudo-label   │    │   → bucket 예측      │
                    └────────┬───────────┘    └──────────┬───────────┘
                             │                           │
                             └────────┬──────────────────┘
                                      │
                                      ▼
                              Loss(예측, pseudo-label)
```

## Pseudo-Label 생성 (Crowds)

```
  25D Signal (학습 시에만 계산)
      │
      ▼ threshold 적용 (참조 이미지 기반 자동 도출 또는 수동)
  ┌──────────────────────────────────────────────┐
  │  face.detect:  conf > 0.8        → ✓        │
  │  head.pose:    yaw ∈ (0, 15)     → ✓        │
  │  face.expr:    happy > 0.6       → ✓        │  14개 binary vote
  │  face.au:      AU12 > 2.0        → ✗        │
  │  ...                                         │
  └──────────────────┬───────────────────────────┘
                     │
              ┌──────▼──────┐
              │  Majority   │  ← Level 1: 단순 다수결
              │  Vote       │     Level 2+: Dawid-Skene
              └──────┬──────┘
                     │
                     ▼
              pseudo-label (0 or 1)
```

## Phased Complexity

```
Level 0 (Day 0):  N_eff 분석 + Exp 0 (crowds 합의 품질 검증)
                  DNN 없음, go/no-go 판단

Level 1 (Week 1): MobileNetV3 + 1 linear head per bucket
                  + BCE against majority vote pseudo-labels
                  → 가장 단순한 것부터 시작

Level 2 (if L1 불충분): Dawid-Skene weighted pseudo-labels로 교체
                        Teacher reliability 가중 합의

Level 3 (if L2 불충분): Soft loss 추가 (teacher 출력 재구성)
                        + Kendall uncertainty weighting

Level 4 (if L3 불충분): Cross-observer transformer replacing DS

Level 5 (future):      Few-shot 새 버킷, drift adaptation
```

**원칙: 실패 모드가 구체적으로 확인된 후에만 복잡도를 올린다.**

---

# Part 1: 아키텍처

## 전체 구조와 비전

```
visualbase  — Read (미디어 I/O)
visualpath  — Process (분석 파이프라인: 14개 frozen 모듈 조합)
visualbind  — Bind & Transcend (통합 학습: visualpath를 넘어서는 모델)
```

### visualpath → visualbind 진화 경로

visualpath는 초창기에 어쩔 수 없이 필요한 **발판(scaffolding)**이다.
적합한 단일 모델이 없어서 frozen 모듈을 조합하여 수동 threshold로 운용하지만, 한계가 명확하다.

visualbind는 이 발판이 만들어낸 pseudo-label을 소비하여 **단일 Vision Student**를 학습한다.
학습이 끝나면 Student만으로 inference — 14개 Teacher가 필요 없어진다.

```
학습:  visualpath → 25D → pseudo-label 생성 (annotation 공장)
       Raw Image → Student backbone → 판단
       Loss(판단, pseudo-label)

추론:  Raw Image → Student만 → 판단
       visualpath 불필요!
```

**visualpath 없이는 visualbind가 시작할 수 없고,
visualbind가 성숙하면 visualpath가 필요 없어진다.**

이것이 Cross-Task Crowd Distillation의 완전한 형태다.

### 선행 근거

"Teacher를 annotation 공장으로 전환" 접근의 원리적 근거:
- **Florence-2** [Xiao 2024, CVPR]: 다양한 frozen 모델 출력으로 FLD-5B(54억 자동 annotation) 생성 → unified vision model 학습. 동일 원리가 작동함을 보여준다.
- **Autodistill** [Roboflow 2023]: foundation model → pseudo-label → target model. 가장 유사한 구조.
- **Ensemble-Then-Distill** [NeurIPS 2024]: ensemble pseudo-label → student distillation
- **Multi-Teacher KD** [Zuchniak 2023]: 여러 teacher → 단일 student, student가 개별 teacher 초과

VisualBind의 차별점: 단일 teacher가 아닌 **cross-task crowd 합의**로 pseudo-label 생성.
대규모 인프라 없이 GPU 1개 + 도메인 데이터(일 1000만+ 프레임)로 적용.

## Teacher 입력: 25D

frozen 모델의 출력만 사용한다. 규칙 기반 계산(blur, brightness 등)은 제외.
**이미 계산되는 출력은 전부 포함** — 불필요한 차원은 crowds 합의에서 자연스럽게 무시됨.

```
AU 12D:        AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26
               (face.au, LibreFace DISFA 0-5, 백엔드 전체 출력)
Emotion 8D:    happy, neutral, surprise, angry, contempt, disgust, fear, sad
               (face.expression, HSEmotion 8-class 전체 확률)
Pose 3D:       yaw (signed), pitch, roll (head.pose, 6DRepNet)
Confidence 1D: detection confidence (face.detect, InsightFace SCRFD)
Face Size 1D:  face bbox area / frame area (face.detect, 추가 비용 0)
```

변경 근거 (Round 4 전문가 리뷰):
- AU 10→12: AU17(Chin Raiser), AU20(Lip Stretcher) 추가. 이미 LibreFace가 출력, 추가 비용 0.
- Emotion 4→8: contempt은 cool_expression 버킷과 직접 관련 (smirk/confident look). 추가 비용 0.
- |yaw| → signed yaw: 좌/우 portrait 구분 정보 보존.
- Face Size: 프레임 대비 얼굴 크기 비율. portrait 품질 + face.detect 실패 모드 감지.

CLIP/DINO 등 범용 임베딩 모델은 제외 — 얼굴 표정/포즈 변화에 대한 변별력 부족 확인.

## Vision Student 아키텍처

### Level 1: 가장 단순한 버전 (Karpathy "200 lines" 철학)

```
Raw Image (224×224)
       ↓
[Pretrained MobileNetV3-Small]  ← ImageNet pretrained, backbone frozen or fine-tuned
       ↓
  feature (576D)
       ↓
[Linear Head₁] → "warm_smile"?       (BCE loss)
[Linear Head₂] → "cool_expression"?  (BCE loss)
[Linear Head₃] → "lateral"?          (BCE loss)
```

이것이 전부. 복잡한 것은 이것이 실패한 뒤에 추가한다.

### Level 2: Dawid-Skene Pseudo-Labels

Level 1과 동일한 모델 구조. 차이는 pseudo-label 생성:
- Level 1: majority vote (단순 다수결)
- Level 2: Dawid-Skene EM으로 모듈 reliability 가중 → 더 정확한 pseudo-label

### Level 3: Soft Loss 추가 (필요시)

```
Raw Image (224×224)
       ↓
[Pretrained MobileNetV3]
       ↓
  feature (576D)
       ↓
  ┌────┴──────────────────────────────────────┐
  │                                            │
[Teacher Head₁] → AU 재구성        [Bucket Head₁] → "warm_smile"?
[Teacher Head₂] → Emotion 재구성   [Bucket Head₂] → "cool_expression"?
[Teacher Head₃] → Pose 재구성      [Bucket Head₃] → "lateral"?
...                                 ...
```

추가 loss: Σ MSE(teacher_head(feat), target) — 연속 출력 재구성.
Kendall uncertainty weighting으로 soft/hard loss 자동 균형.

### Level 4: Cross-Observer Transformer (필요시)

Dawid-Skene를 학습 가능한 cross-attention으로 대체.
14개 Teacher vote를 token으로 취급 → attention으로 reliability 학습.

### 파라미터 규모

| 구성요소 | Level 1 | Level 3 |
|----------|---------|---------|
| MobileNetV3-Small backbone | ~2.5M | ~2.5M |
| Bucket Head × 3 (576→1) | ~1.7K | ~1.7K |
| Teacher Head × 14 (576→출력) | — | ~12K |
| Uncertainty params | — | ~17 |
| **합계** | **~2.5M** | **~2.5M** |

GPU 1개로 학습 가능. Backbone을 frozen으로 시작하면 학습할 파라미터는 head만 (~14K).

### Backbone 후보

| Backbone | Params | 비고 |
|----------|--------|------|
| **MobileNetV3-Small** | ~2.5M | **경량, Level 1 권장** |
| MobileNetV3-Large | ~5.5M | 성능 부족 시 확장 |
| EfficientNet-B0 | ~5M | 범용 |
| IResNet-18 (ArcFace) | ~11M | 얼굴 특화 pretrained |

### Add-When-Needed 옵션

아래는 Level 1-2가 불충분할 때만 고려하는 기법들:

| 기법 | 추가 시점 | 목적 |
|------|----------|------|
| Soft loss (teacher 재구성) | Level 3 | threshold 정보 손실 복구 |
| Kendall uncertainty weighting | Level 3 | multi-loss 자동 균형 |
| VICReg | Level 3 | representation collapse 방지 |
| Stochastic Head Masking | Level 3 | encoder cross-modal mixing 강제 |
| Cross-observer transformer | Level 4 | DS 대체, end-to-end reliability 학습 |

---

## Pseudo-Label 생성

### Binary Vote 방식

Threshold는 두 가지 방법으로 설정 가능:

**방법 1 — Catalog 참조 이미지 기반 자동 도출 (권장)**

```python
# 1. 참조 이미지에 frozen 모델 실행 → 25D 분포 확보
ref_outputs = [run_frozen_models(img) for img in catalog_refs["warm_smile"]]

# 2. 분포에서 threshold 자동 추정
thresholds = {
    "face.expr.happy": percentile(ref_outputs[:, happy_idx], 5),  # ≈ 0.55
    "face.au.AU12":    percentile(ref_outputs[:, au12_idx], 5),   # ≈ 1.8
    "head.pose.yaw":   percentile(ref_outputs[:, yaw_idx], 95),   # ≈ 18°
    ...
}
```

참조 이미지가 개념의 시드 역할 — frozen 모델이 독립 관찰 → 분포에서 경계 자동 도출.
Catalog에 이미 참조 이미지가 존재 (warm_smile 25장, cool_expression 27장, lateral 31장).

**방법 2 — 수동 threshold (fallback)**

기존 파이프라인의 threshold를 그대로 재활용.

**두 방법 모두 실험**하여 비교 (자동 threshold vs 수동 threshold).

### Tier 분류 (Level 2+) — Dawid-Skene Posterior 기반

Level 1에서는 단순 majority vote. Level 2+에서 DS posterior 사용:

| Tier | 기준 | 용도 |
|------|------|------|
| Tier 1 | DS posterior > 0.95 또는 < 0.05 | 학습 데이터 (고신뢰) |
| Tier 2 | 0.05 ≤ posterior ≤ 0.95 중 중간 | 보조 학습 or 제외 |
| Tier 3 | posterior ≈ 0.5 (불확실) | Blind spot 후보, 맹점 감지 |

---

## Multi-Bucket 표현

### 구조

```
backbone feature (576D)
    ↓
[Bucket Head₁] → "warm_smile" 적합?     (학습됨)
[Bucket Head₂] → "cool_expression" 적합?      (학습됨)
[Bucket Head₃] → "lateral" 적합?         (학습됨)
[Bucket Head₄] → "eyes_closed"?          (few-shot transfer)
```

### Few-Shot 새 버킷 추가

```
[기존] 3개 버킷 × 1000+ 샘플로 backbone + Heads 학습
[새 버킷] "eyes_closed" 정의
  → 기존 데이터에서 threshold 적용 → 50-100건 pseudo-label 자동 생성
  → Backbone 고정 + 새 Bucket Head만 학습
  → 1시간 이내 새 버킷 추가 가능
```

catalog_scoring에서 새 카테고리 추가 시 참조 이미지 촬영 + SIGNAL_RANGES 정의 + 가중치 조정이 필요했던 것의 대체.

### Hierarchical Dawid-Skene Transfer (Level 2+)

기존 버킷에서 추정한 모듈 reliability를 새 버킷의 prior로 사용:
- 기존 3개 버킷에서 각 모듈의 confusion matrix 추정 완료
- 새 버킷에 대해 이 prior를 초기값으로 → 10-20건이면 Dawid-Skene 수렴
- DNN transfer (backbone 고정 + Head만 학습)와 결합하면 두 레벨 시너지

---

# Part 2: 패키지 구조 & 기술 스택

## 패키지 구조

```
libs/visualbind/src/visualbind/
├── types.py           # SourceSpec, HintVector, HintFrame
├── collector.py       # Teacher 출력 수집 → parquet 저장
├── pseudo_label.py    # majority vote / Dawid-Skene → pseudo-label 생성
├── dataset.py         # PyTorch Dataset (raw image + pseudo-label)
├── model.py           # VisionStudent (MobileNetV3 + heads)
├── trainer.py         # Trainer (학습 루프)
├── evaluator.py       # Evaluator (Anchor Set 대비 평가, 메트릭)
├── cli.py             # visualbind analyze / train / eval / label CLI
└── __init__.py
```

### 각 모듈의 책임

| 모듈 | 입력 → 출력 | 의존성 |
|------|------------|--------|
| `collector` | Observation[] → parquet (25D + 이미지 경로) | numpy, pyarrow |
| `pseudo_label` | parquet → labeled parquet | crowdkit (Level 2+) |
| `dataset` | labeled parquet + 이미지 → DataLoader | PyTorch, PIL |
| `model` | Raw image → bucket predictions | PyTorch, torchvision |
| `trainer` | dataset → trained model (.pt) | PyTorch |
| `evaluator` | model + anchor set → metrics + HTML report | PyTorch, sklearn, plotly |
| `cli` | CLI args → 위 모듈 조합 | argparse |

## 의존성 관리

```toml
[project]
dependencies = ["numpy", "pyarrow"]

[project.optional-dependencies]
train = ["torch", "torchvision", "crowd-kit", "scikit-learn", "pandas", "plotly", "Pillow"]
dev = ["pytest>=7.0.0"]
```

- 기본 의존성: 수집(collector)만 사용하는 경우 가볍게 유지
- `train` extras: 학습/평가 시에만 필요한 무거운 패키지
- momentscan이 visualbind의 collector만 사용하면 torch를 끌고 오지 않음

## 프레임워크 선택 근거

| 결정 | 선택 | 이유 |
|------|------|------|
| 학습 프레임워크 | **PyTorch** | autograd, pretrained backbone, 코드 단순화 |
| Backbone | **torchvision** | MobileNetV3 pretrained weights 제공 |
| Crowds 알고리즘 | **crowdkit** | Dawid-Skene/MajorityVote, 검증된 구현 |
| 저장 포맷 | **parquet** | 컬럼 기반 분석 용이, pandas 친화 |
| 시각화 | **Plotly HTML** | momentscan-report 패턴, 단일 파일 |

---

# Part 3: CLI & 시각화

## 서브커맨드

```bash
# Day 0: 독립성 분석 + Dawid-Skene go/no-go
visualbind analyze --data ./teacher_outputs.parquet --output ./report.html

# Anchor Set 라벨링용 HTML 생성
visualbind label --data ./teacher_outputs.parquet \
    --buckets warm_smile,cool_expression,lateral \
    --sample 200 --output ./review.html

# 학습
visualbind train --data ./teacher_outputs.parquet \
    --images ./frames/ \
    --buckets warm_smile,cool_expression,lateral \
    --output ./models/student_v1.pt \
    --report ./train_report.html

# 평가
visualbind eval --model ./models/student_v1.pt \
    --anchor ./anchor_set/ \
    --report ./eval_report.html
```

실행 흐름: `analyze (go/no-go) → label (Anchor Set) → train → eval (검증)`

## HTML 리포트 내용

### analyze report
- 모듈별 출력 분포 (히스토그램, 25D 각각)
- 모듈 간 상관 행렬 (heatmap)
- N_eff 계산 과정과 결과
- 버킷별 majority vote vs Dawid-Skene 비교

### train report
- Loss curve
- Bucket Head별 정확도 추이
- 오분류 사례 (프레임 이미지 + Teacher 출력 + Student 판단)

### eval report
- Student vs catalog_scoring ROC curve
- 버킷별 confusion matrix
- Bootstrap CI 시각화
- 오분류 사례

### label (Anchor Set 리뷰)
- 프레임 이미지 + 버킷 정의
- 적합/부적합 판단 인터페이스 (Teacher vote blinding)
- 진행률 표시

모든 리포트는 **단일 HTML 파일**로 생성. momentscan-report와 동일한 패턴.

---

# Part 4: MVP 경로

## MVP 범위: 3개 버킷 동시 학습

bias 분리를 위해 **2개 이상 버킷 동시 학습이 필수**라는 전문가 합의에 따라:

```
MVP 버킷 (3개):
  - warm_smile  — 정면 따뜻한 미소 (가장 대표적, 참조 25장)
  - cool_expression   — 정면 시크/무표정 (반대 표정 → 표정 bias 분리)
  - lateral     — 측면 portrait (포즈 기반 → 포즈 bias 분리)
```

## Day 0 — Go/No-Go 프로토콜

MVP 투자 전에, **프로젝트 feasibility를 판단**한다.

### Observer 독립성 분석

```
(1) 14개 모듈의 binary vote에 대해 Fleiss' kappa 계산 (30분)
(2) 모듈 쌍별 Cohen's kappa → 오류 상관 행렬 (1시간)
(3) 500건 수동 라벨링 → 실제 정답 대비 각 vote의 정확도 (2-3일)
    ※ Teacher vote에 대한 blinding 필수 — 프레임 이미지와 버킷 정의만 보고 라벨링

N_eff 계산: 고유값 기반
    상관행렬 C의 고유값 λ₁...λ_N에 대해:
    N_eff = (Σλ)² / Σλ²
    → 이종 상관 구조를 정확히 반영
```

### 입력 의존성 체인 분석 — 계층적 N_eff

Face-crop 기반 모듈이 face.detect에 **결정적 의존** (상관이 아닌 deterministic failure).
detect가 얼굴을 놓치면 expression, AU, face_parse, face_quality, head_pose 전부 garbage.

```
face.detect 실패 → N_eff ≈ 1 (detect 자체만 정보 제공)
face.detect 성공 → 나머지 모듈 간 조건부 N_eff 계산
```

**계층적 분석 필수:**
- detect 성공/실패 stratum 분리하여 각각 N_eff 계산
- body.pose, hand.gesture 포함/제외 양쪽 실험
- Hierarchical clustering으로 하위 구조 분석

### Go/No-Go 기준

| N_eff | 판단 |
|-------|------|
| >= 3 | MVP 진행 |
| 2 <= N_eff < 3 | 합의 기준 상향 + 진행 |
| < 2 | 모델 다양화 또는 근본적 재고 |

### Exp 0 — DNN 없이 Crowds 검증

```
MajorityVote → Dawid-Skene → GLAD → Snorkel label model
vs Baseline: 수동 threshold AND 조합 (현행)
평가: 5-fold CV (fit과 eval 분리 — train-on-test 오염 방지)
```

crowdkit + snorkel을 활용하여 4종 비교.
**실패 시 전체 방향 재검토. 성공 시 Vision Student로 진행.**

---

## 실험 설계 (Exp 0-3)

### Baselines

```
Baseline 1: 수동 threshold AND 조합 (현행 파이프라인)
Baseline 2: catalog_scoring Fisher-weighted distance (선형)
Baseline 3: Logistic regression on 25D (simple ML baseline)
Baseline 4: XGBoost on 25D (strong ML baseline)
Baseline 5: Dawid-Skene weighted majority vote (Exp 0)
```

Baseline 3-4는 25D feature에 대한 전통 ML 성능 상한을 측정한다.
Vision Student가 이를 넘으면 raw image에서 25D 이상의 정보를 추출한다는 증거.

### Exp 1: Vision Student Level 1

```
입력:  Raw Image (224×224)
모델:  MobileNetV3-Small + linear head per bucket
Loss:  Σ BCE(head_k(feat), majority_vote_k)
비교:  vs Baseline 1-5
검증:  "raw image에서 직접 판단이 가능한가"
```

### Exp 2: Vision Student Level 2

```
입력:  Raw Image (224×224)
모델:  동일 (MobileNetV3 + linear heads)
변경:  majority vote → Dawid-Skene weighted pseudo-labels
비교:  vs Exp 1
검증:  "reliability 가중 pseudo-label의 가치"
```

### Exp 3: Vision Student Level 3 (필요시)

```
추가:  Teacher Head (soft loss) + Kendall uncertainty weighting
Loss:  Σ MSE(teacher_head_i(feat), target_i)/(2σ²_i) + log(σ_i)
     + Σ BCE(bucket_head_k(feat), vote_k)/(2σ²_k) + log(σ_k)
비교:  vs Exp 1, 2
검증:  "연속 출력 재구성의 가치" (threshold 정보 손실 복구)
```

### Cold Start 실험

Exp 1-2 각각에 대해:
- **Cold start**: Teacher 출력 + binary vote만으로 학습
- **Warm start**: catalog_scoring의 Fisher weights → Teacher reliability prior로 활용
- **Random prior**: uniform 분포에서 랜덤 prior (control)

### 단계별 추가 실험 (필요 시점에 실행)

| 시점 | 추가 실험 | 목적 |
|------|----------|------|
| **Exp 2 후** | Backbone ablation (MobileNetV3-Small/Large, EfficientNet-B0) | 모델 크기 적정성 |
| **Exp 2 후** | Frozen vs fine-tuned backbone | pretrained feature 충분한지 |
| **Exp 3 후** | per-teacher-head reconstruction R² | feat가 teacher 정보를 담는지 |
| **Exp 4** | few-shot: N당 10회 random draw | 분산 추정, per-bucket 분리 보고 |

## 성공/실패 판단 기준

### Primary Endpoint (사전등록)

> **Exp 2 (Vision Student + DS pseudo-label) vs catalog_scoring, macro-averaged AUC-ROC, alpha=0.01**

이것이 유일한 통계적 유의성 주장. 나머지 실험은 전부 exploratory (ablation/탐색).

### 통계 방법

**Block Bootstrap CI + Effect Size 병행:**
- Block bootstrap (session/day 단위 클러스터링, 1000회 리샘플링)
  — 같은 세션의 프레임은 독립이 아니므로 naive i.i.d. bootstrap 부적절
- 95% 신뢰구간 → CI가 겹치지 않으면 통계적 유의
- Effect size → AUC 2%p 이상 차이면 실무적으로 유의미
- 둘 다 만족해야 "개선"으로 판단

### 성공 기준

1. **N_eff >= 3** (go/no-go 전제 조건)
2. **Vision Student >= catalog_scoring 성능** (Bootstrap CI 분리)
3. **Pseudo-label 정확도 > 95%** (Anchor Set 기준)
4. **Teacher reliability 분산 존재** (bias 분리 가능성)

## MVP 타임라인

```
Day 0-1:   Observer 독립성 분석 + N_eff + Exp 0 (go/no-go)
Day 2-3:   Teacher 출력 저장 파이프라인 (collector → parquet + 이미지 경로)
Day 4-5:   Pseudo-label 생성 + Baselines (logistic regression, XGBoost on 25D)
Day 6-8:   Anchor Set 수동 검증 (500건, HTML 리뷰 페이지, Teacher vote blinding)
Day 8-10:  Vision Student 학습 (Exp 1 → 2, cold/warm 비교)
Day 11-12: Validation gate + model swap
Day 13-17: 1주일 파일럿 + 모니터링
```

---

# Part 5: 데이터 파이프라인

## Teacher 출력 수집

momentscan 레벨에서 Observation[]이 나오는 시점에 수집:

```
momentscan → Observation[] → collector → parquet 저장 (25D + 이미지 경로)
                           → momentbank (기존 프레임 저장, 변경 없음)
```

visualbind의 collector는 momentscan의 **소비자**로서 signal을 수집한다.
학습 시에는 이미지 경로로 raw frame을 로드하여 Student에 입력.

### Replay Buffer

학습 시 데이터 편향 방지:

```
50% = 최근 7일 데이터 (recency)
30% = 전체 버퍼에서 조건별 균등 추출 (coverage)
10% = Tier 2 (경계선) 샘플 (hard negatives — 모델 calibration 유지)
10% = 고신뢰 합의 샘플 (anchor — 안정성 유지)
```

조건 태그: brightness, pose bucket, 시간대 기반 자동 분류.
폐기: FIFO + 각 조건별 최소 보유량 유지.

### Anchor Set

시스템 건전성 검증을 위한 **최소 500건의 수동 검증 데이터** (validation set).
**Annotation-free training, annotation-efficient validation** — 학습에는 annotation 불필요,
검증에만 소규모 annotation 사용.

- 버킷별 stratified sampling (버킷당 ~170건)
- Tier 1 (고신뢰 합의) + Tier 3 (합의 실패)에서 균형 추출
- `visualbind label` CLI로 HTML 리뷰 페이지 생성 → 브라우저에서 라벨링
- **Teacher vote blinding 필수** — 프레임 이미지와 버킷 정의만 보고 라벨링 (anchoring bias 방지)
- Weekly 100건 추가 검증으로 갱신

---

# Part 6: Drift 적응

## 5가지 Drift와 대응

### Data Drift (입력 분포 변화)

- 원인: 계절, 시간대, 고객층, 조명 환경 변화
- 감지: Teacher 출력 분포의 일별 mean/std 추적
- 대응: Replay buffer의 coverage 비율 자동 조정, 재학습

### Concept Drift (기준 변화) — 의도적 활용

- 원인: 디자인팀 요구사항 변경, 시즌별 수집 기준 변경
- 대응: **새 버킷 추가 (few-shot)** 또는 기존 버킷 가중치 조정
- 이것은 방어가 아니라 **핵심 기능** — 빠른 기준 변경이 시스템의 가치

### Teacher Drift (모듈 교체)

- 원인: upstream 모듈 버전 업그레이드
- 감지: Teacher 모듈 버전을 학습 데이터에 태깅
- 대응: 버전 변경 감지 시 해당 모듈의 기존 데이터 무효화 + 재학습

### Preprocessing Drift (전처리 변경)

- 원인: face detection threshold/NMS 파라미터 변경, crop 로직 수정
- 위험성: 개별 Teacher 모델은 변경 없으나 **모든 face-crop 기반 모듈 출력이 동시 변화**
- 감지: face crop 통계(종횡비, crop 크기, confidence 분포) 독립 모니터링
- 대응: crop 통계 이상 감지 시 전체 Teacher 출력 재검증

### Self-Reinforcing Drift — 구조적 차단

가장 위험한 drift. Student의 편향이 자기 학습을 오염시키는 루프.

**핵심 원칙:**
> Student는 pseudo-label 생성에 참여하지 않는다.
> Teacher(frozen)만이 유일한 pseudo-label 소스다.

```
Teacher (frozen) → vote → pseudo-label → Student 학습
                                              ↓
                                         Bucket 판단 (운용)
                                              ✗ pseudo-label로 되돌아가지 않음
```

## 학습 빈도

단계적 전환:

```
초기:     수동 트리거 — visualbind train CLI로 필요 시 실행
검증 후:  주간 배치 — 주 1회 재학습
안정화 후: 야간 배치 — 매일 재학습
```

---

# Part 7: 안전장치

## 핵심 모니터링: 주간 Human Agreement Check

Production 전문가 제안: 복잡한 메트릭 7개보다 **1개 핵심 지표**에 집중.

> **주 1회, Anchor Set에서 100건 샘플링 → Student 판단 vs 사람 판단 일치율 측정**

이것이 떨어지면 모든 것이 문제. 이것이 유지되면 나머지는 부차적.

### Rollback 기준

```
[자동 롤백]
- 주간 human agreement가 이전 주 대비 5%p 이상 하락 → 이전 모델로 롤백

[경보 + 수동 조사]
- 주간 human agreement가 이전 주 대비 2%p 이상 하락 → 알림
- Tier 3 비율 전체의 30% 초과

[학습 중단]
- Student가 catalog_scoring 대비 Anchor Set에서 유의미하게 열위
```

### Blind Spot Registry

Tier 3 (합의 실패) 프레임을 별도 저장.
이것은 모든 teacher가 불확실한 영역 = 시스템의 맹점.
주기적으로 분석하여 새 모듈 추가나 threshold 조정의 근거로 활용.

---

# Part 8: 전환 경로 — Phased Complexity

## 단계별 로드맵

```
Level 0 (Day 0-1)
═══════════════════════════════════════════════════
  N_eff 분석 + Exp 0 (crowds 합의 품질)
  Go/No-Go 판단. 실패 시 전체 방향 재검토.

Level 1 (Week 1)
═══════════════════════════════════════════════════
  MobileNetV3-Small + 1 linear head per bucket
  BCE loss against majority vote pseudo-labels
  가장 단순한 것부터 검증

Level 2 (Week 2, if L1 불충분)
═══════════════════════════════════════════════════
  동일 모델, pseudo-label을 Dawid-Skene weighted로 교체
  Teacher reliability 가중 합의

Level 3 (if L2 불충분)
═══════════════════════════════════════════════════
  Soft loss 추가 (teacher 출력 재구성)
  Kendall uncertainty weighting
  VICReg (optional)

Level 4 (if L3 불충분)
═══════════════════════════════════════════════════
  Cross-observer transformer replacing DS
  End-to-end reliability 학습

Level 5 (future)
═══════════════════════════════════════════════════
  Few-shot 새 버킷
  Drift adaptation 자동화
  Student의 feature를 downstream task에 활용
```

각 level에서 이전 level 대비 개선폭 측정 → diminishing return 시 중단.
**Level 0이 실패하면 이후 전부 불가** — crowds 합의 품질이 모든 것의 전제.

**솔직한 고백:** Level 3-5는 Level 0-2가 불충분함을 증명할 때까지 추측이다.

## catalog_scoring과의 병행

```
Phase A — Shadow: 두 시스템 병렬 실행, Student 결과 로깅만
Phase B — Hybrid: α × Student + (1-α) × catalog 블렌딩
Phase C — Full: catalog 퇴역 → Student로 완전 대체
```

### 전환 가교

| catalog_scoring | VisualBind 대응 |
|----------------|----------------|
| `SIGNAL_FIELDS`, `SIGNAL_RANGES` | Vision backbone이 raw image에서 직접 학습 |
| Fisher ratio weights | Dawid-Skene reliability (Day 0 자동 계산) |
| Category centroids | Bucket Head linear classifier |
| `compute_importance_weights()` | Teacher reliability 초기값 (warm start 실험) |

## visualpath와의 관계 변화

```
학습 시:  visualpath = annotation 공장 (25D → pseudo-label 생성)
          visualbind = raw image → Student → 판단

추론 시:  visualpath = 불필요
          visualbind = raw image → Student만 → 판단

성숙 후:  visualpath = 퇴역 (새 도메인에서만 bootstrapping 용도)
          visualbind = 도메인 특화 model
```

---

# Part 9: 논문 실험 설계

## 실험 구조

```
Exp 0:  Dawid-Skene EM vs 수동 AND (DNN 없음, Day 0)
Exp 1:  Vision Student (majority vote) vs Baselines (logistic reg, XGBoost, catalog_scoring)
Exp 2:  Vision Student (DS pseudo-label) vs Exp 1
Exp 3:  Vision Student + soft loss vs Exp 2 (ablation, 필요시)
Exp 4:  Few-shot 새 버킷 transfer vs catalog_scoring 새 카테고리
Exp 5:  Cold start vs warm start (catalog_scoring prior)
Exp 6:  Inference 비용 비교 — 14 Teachers vs Student 단일 모델
Exp 7:  Teacher 수 vs 성능 — N개 Teacher subset으로 학습 시 성능 변화
```

### Exp 4 — 논문의 가장 Compelling한 Figure

```
기존 3개 버킷으로 학습된 Student
  → 새 버킷 1-2개 (hold-out)
  → N-shot (20, 50, 100, 200)에서의 성능 곡선
  → vs catalog_scoring의 centroid matching (7장 참조)
```

## 측정 지표

| 지표 | 설명 |
|------|------|
| AUC-ROC | 버킷 적합성 분류 성능 |
| Accuracy@threshold | 현행 threshold 기준 정확도 |
| catalog_scoring 대비 상대 개선 | Fisher distance vs Student 판단 |
| Few-shot learning curve | N-shot별 성능 곡선 |
| Inference latency | 14 Teachers vs Student 단일 모델 속도 비교 |

---

# Part 10: 열린 질문

## 설계 결정이 필요한 것

| 질문 | 선택지 | 현재 방향 |
|------|--------|----------|
| Backbone | MobileNetV3-Small / Large / EfficientNet-B0 | Small로 시작, 부족하면 확장 |
| Backbone frozen vs fine-tune | frozen + head only / full fine-tune | frozen으로 시작 |
| Pseudo-label 방식 | majority vote / DS / GLAD | majority vote(L1), DS(L2) |
| Image augmentation | basic / heavy / none | basic (flip, color jitter) |
| Bucket Head 구조 | Linear / 1-layer MLP | Linear로 시작 |

## 검증이 필요한 가정

| 가정 | 검증 방법 |
|------|----------|
| 다수 observer 합의 = 높은 정확도 pseudo-label | Anchor Set 500건 수동 검증 |
| Raw image에서 bucket 판단이 가능하다 | Exp 1에서 25D baseline 대비 비교 |
| Pretrained MobileNetV3가 충분한 feature를 제공한다 | Frozen vs fine-tuned 비교 |
| Observer 독립성이 충분하다 | Day 0 N_eff 분석 |
| Few-shot 버킷 추가가 가능하다 | Exp 4에서 hold-out 버킷 실험 |

## 알려진 제약 (Limitation)

1. **Identifiability**: GT 없이 truth와 bias를 완벽히 분리하는 것은 수학적으로 underdetermined. 실증 + Anchor Set으로 보완
2. **Observer 독립성 위반**: 14개 모델의 공유 backbone/데이터 + 입력 의존성 체인. N_eff는 14보다 상당히 낮을 것 (5-7개 추정)
3. **Vision Student 한계**: pretrained MobileNetV3가 얼굴 미세 표정을 충분히 구분하지 못할 수 있음. 이 경우 얼굴 특화 backbone (IResNet-18)으로 전환
4. **버킷 정의 자체의 타당성**: "warm_smile 정면" 버킷이 고객 만족과 대응하는가는 VisualBind 범위 밖
