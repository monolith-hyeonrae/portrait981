# visualbind — 어떻게 풀 것인가

> `why-visualbind.md`에서 다룬 문제의식을 VisualBind가 어떤 설계로 해결하는지,
> 아키텍처, 기술 스택, MVP 경로, 실험 설계, 안전장치를 정리.

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

visualbind는 이 발판 위에서 자라나는 시스템이다.
visualpath가 만들어낸 경험(데이터 + pseudo-label)을 먹고 자라서,
궁극적으로는 visualpath 전체를 단일 모델로 distill한다.

```
Phase 1 (MVP):   visualpath → 24D signal → Student MLP → 판단 개선
                 Teacher 출력의 수동 조합을 학습으로 대체
                 (inference: visualpath + Student 모두 필요)

Phase 2 (확장):  visualpath → pseudo-label 생성 (학습 시에만)
                 Raw Image → Student CNN/ViT → 직접 판단
                 (inference: Student만 필요, visualpath 불필요)

Phase 3 (성숙):  Student가 도메인 특화 foundation model
                 visualpath는 역할을 다하고 퇴역
                 Student의 feature를 다른 downstream task에 활용
```

**visualpath 없이는 visualbind가 시작할 수 없고,
visualbind가 성숙하면 visualpath가 필요 없어진다.**

이것이 Cross-Task Crowd Distillation의 완전한 형태다:
- Phase 1: Teacher 출력의 더 나은 조합 (signal-level distillation)
- Phase 2: Teacher 전체를 단일 모델로 흡수 (end-to-end distillation)

### Phase 1 구조 (MVP)

```
Raw Image → [14 Teachers (항상 실행)] → 24D signal → [Student MLP] → 버킷 판단
```

### Phase 2 구조 (Raw 입력)

```
학습 시: Raw Image → [14 Teachers] → pseudo-label (annotation 공장)
         Raw Image → [Student CNN] → 24D 예측 + 버킷 판단
         Loss = Teacher 출력 재구성 + 버킷 pseudo-label + VICReg

Inference 시: Raw Image → [Student CNN만] → 버킷 판단
              14개 Teacher 실행 불필요
```

Phase 2 backbone 후보:

| Backbone | Params | GPU 1개 학습 | 비고 |
|----------|--------|-------------|------|
| MobileNetV3 | ~5M | 가능 | 경량 |
| EfficientNet-B0 | ~5M | 가능 | 범용 |
| ResNet-18 | ~11M | 가능 | 범용 |
| **IResNet-18 (ArcFace)** | ~11M | 가능 | **얼굴 특화 pretrained, 추천** |
| ViT-Tiny | ~6M | 가능 | Attention 기반 |

### 선행 근거

Phase 2의 "Teacher를 annotation 공장으로 전환" 접근은 다수의 선행연구에서 검증됨:
- **Florence-2** [Xiao 2024, CVPR]: 기존 모델 출력으로 FLD-5B(54억 자동 annotation) 생성 → unified vision model 학습
- **Autodistill** [Roboflow 2023]: foundation model → pseudo-label → target model. 정확히 Phase 2 구조
- **Ensemble-Then-Distill** [NeurIPS 2024]: ensemble pseudo-label → student distillation
- **Multi-Teacher KD** [Zuchniak 2023]: 여러 teacher → 단일 student, student가 개별 teacher 초과

VisualBind의 차별점: 단일 teacher가 아닌 **cross-task crowd 합의**로 pseudo-label 생성.
대규모 인프라 없이 GPU 1개 + 도메인 데이터(일 1000만+ 프레임)로 적용.
전체 reference 목록은 `why-visualbind.md` 참조.

## Teacher 입력: 24D

frozen 모델의 출력만 사용한다. 규칙 기반 계산(blur, brightness 등)은 제외.
**이미 계산되는 출력은 전부 포함** — 모델이 불필요한 차원은 자동으로 무시하도록 학습.

```
AU 12D:       AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26
              (face.au, LibreFace DISFA 0-5, 백엔드 전체 출력)
Emotion 8D:   happy, neutral, surprise, angry, contempt, disgust, fear, sad
              (face.expression, HSEmotion 8-class 전체 확률)
Pose 3D:      yaw (signed), pitch, roll (head.pose, 6DRepNet)
Confidence 1D: detection confidence (face.detect, InsightFace SCRFD)
```

변경 근거 (Round 4 전문가 리뷰):
- AU 10→12: AU17(Chin Raiser), AU20(Lip Stretcher) 추가. 이미 LibreFace가 출력, 추가 비용 0.
- Emotion 4→8: contempt은 cool_gaze 버킷과 직접 관련 (smirk/confident look). 추가 비용 0.
- |yaw| → signed yaw: 좌/우 portrait 구분 정보 보존. 좌우 대칭이 필요하면 모델이 학습.

CLIP/DINO 등 범용 임베딩 모델은 제외 — 얼굴 표정/포즈 변화에 대한 변별력 부족 확인.

### 입력 정규화: LayerNorm

24D signal의 스케일이 제각각이다 (AU: 0-5, Emotion: 0-1, Pose: -60°~60°).
수동 정규화 대신 **Student MLP 첫 레이어에 LayerNorm을 적용**하여 모델이 자동 적응.

LayerNorm 선택 이유 (Round 4 전문가 리뷰):
- BatchNorm은 inference 시 single-frame 처리에서 running statistics 불안정
- LayerNorm은 feature 차원(24D)에 대해 정규화 → 배치 크기 무관, train/eval 동일 동작
- Teacher drift 시 (모듈 교체로 출력 범위 변화) 재학습만으로 새 분포에 적응
- 수동 범위 설정(`SIGNAL_RANGES` 하드코딩) 불필요

## Student 아키텍처

### 핵심 구조

```
14개 Teacher 연속 출력 (24D)
         ↓
    [LayerNorm]
         ↓
[Shared Encoder] — MLP (24D → z, 2-3 hidden layers)
         ↓
    z (16D latent state) — "이 프레임의 본질적 특성"
         ↓
    ┌────┴────────────────────────────────────┐
    │                                          │
[Teacher Head₁] → expression 예측    [Bucket Head₁] → "warm_smile" 적합?
[Teacher Head₂] → AU 예측            [Bucket Head₂] → "cool_gaze" 적합?
[Teacher Head₃] → pose 예측          [Bucket Head₃] → "lateral" 적합?
...                                   ...
[Teacher Head_N] → confidence 예측   [Bucket Head_k] → 새 버킷? (few-shot)
```

### 각 구성요소의 역할

| 구성요소 | 입력 → 출력 | 역할 |
|----------|------------|------|
| **LayerNorm** | 24D → 24D (정규화) | 스케일 통일, Teacher drift 적응, single-frame inference 안정 |
| **Shared Encoder** | 24D → z (16D) | model_bias를 보정한 버킷-불변 표현 학습 |
| **Teacher Head_i** | z → 모듈_i 연속 출력 | 각 teacher의 연속 출력 재구성 (soft loss) |
| **Bucket Head_k** | z → binary | 특정 버킷에 대한 적합성 판단 (hard loss) |

### Dual-Mode Loss — Uncertainty Weighting

**Kendall et al. (CVPR 2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"** 방식 채택.
task별 homoscedastic uncertainty(σ²)를 학습 가능한 파라미터로 두어 loss 가중치를 자동 조절.

```
Loss = Σᵢ [ MSE(teacher_head_i(z), target_i) / (2σ²_i) + log(σ_i) ]   [soft, 정보 보존]
     + Σₖ [ BCE(bucket_head_k(z), vote_k) / (2σ²_k) + log(σ_k) ]       [hard, 방향성]
     + λ × VICReg(z)                                                      [representation quality]
```

Uncertainty weighting의 장점:
- **이종 스케일 자동 해결**: AU(0-5), Emotion(0-1), Pose(±60°)의 gradient 지배 방지
  - 수동 α 없이, 스케일이 큰 task → σ² 자동 증가 → loss 자동 down-weight
- **Soft/Hard 비율 자동 학습**: 기존 α hyperparameter grid search 불필요
- 구현: per-task `log_var` 파라미터 추가 (~30개 스칼라), 코드 ~10줄

- **Soft loss** (Teacher Head): 연속 출력 재구성 → threshold 정보 손실 복구, cross-modal 관계 학습
- **Hard loss** (Bucket Head): binary vote → crowds 이론의 학습 방향성
- **VICReg**: variance(붕괴 방지) + covariance(차원 간 다양성 유지)

### Stochastic Head Masking

z의 특정 차원만 편향적으로 사용되는 것을 방지하기 위해,
학습 시 **매 step에서 1-2개 Teacher Head의 loss를 랜덤하게 제거** (전체 head masking).
z가 모든 teacher 정보를 유지하도록 강제 (어떤 head가 활성화될지 모르므로).

> 기존 설계의 neuron-level dropout 대신 head-level masking 채택 (Round 4 KD 전문가 제안).
> Dropout은 head 내부의 regularizer일 뿐, encoder의 cross-modal mixing을 강제하지 못함.

### 파라미터 규모

| 구성요소 | 파라미터 수 | 비고 |
|----------|-----------|------|
| LayerNorm (24D) | ~48 | |
| Shared Encoder (24→64→32→16) | ~3.5K | |
| Teacher Head × 14 (16→32→출력) | ~7K | |
| Bucket Head × 5 (16→32→1) | ~3K | |
| Uncertainty params (log_var) | ~19 | 14 teacher + 5 bucket |
| **합계** | **~14K** | CPU 학습 가능 (10-30분), GPU 1개 보유 (향후 확장 여지) |

---

## Pseudo-Label 생성

### Binary Vote 방식

기존 threshold를 그대로 재활용하여 pseudo-label 생성:

```python
# 버킷 "warm_smile"에 대한 모듈의 binary vote
votes = {
    "face.detect": confidence > 0.8,      # ✓/✗
    "head.pose":   yaw in (0, 15),         # ✓/✗
    "face.expr":   happy > 0.6,            # ✓/✗
    "face.quality": blur < 100,            # ✓/✗
    ...
}

pseudo_label = majority_vote(votes)        # 또는 Dawid-Skene weighted
confidence = vote_count / total_voters
```

추가 설계 불필요 — 현재 파이프라인의 threshold를 바로 사용 가능.

### Tier 분류 — Dawid-Skene Posterior 기반

raw vote 비율(80%/50%) 대신 **Dawid-Skene posterior probability**를 사용한다.
DS가 이미 모듈 reliability를 가중한 확률을 계산하므로, 이를 그대로 활용하는 것이
raw vote 대비 이론적으로 엄밀하고 정보 손실이 없다.

| Tier | 기준 | 용도 |
|------|------|------|
| Tier 1 | DS posterior > 0.95 또는 < 0.05 | 학습 데이터 (고신뢰) |
| Tier 2 | 0.05 ≤ posterior ≤ 0.95 중 중간 | 보조 학습 or 제외 |
| Tier 3 | posterior ≈ 0.5 (불확실) | Blind spot 후보, 맹점 감지 |

---

## Multi-Bucket 표현

### 구조

```
z (16D, 버킷 무관 표현, Shared Encoder가 학습)
    ↓
[Bucket Head₁] → "warm_smile" 적합?     (학습됨)
[Bucket Head₂] → "cool_gaze" 적합?      (학습됨)
[Bucket Head₃] → "lateral" 적합?         (학습됨)
[Bucket Head₄] → "eyes_closed"?          (few-shot transfer)
```

### 왜 가능한가

- Multi-task learning(Caruana 1997): 관련 task 동시 학습이 개별보다 robust
- 초상화 버킷들은 얼굴 품질/조명/선명도 등 공유 요소가 많음 → negative transfer 위험 낮음
- z가 model_bias를 보정한 "깨끗한" 표현이면, bucket head만 교체하여 transfer 가능

### Few-Shot 새 버킷 추가

```
[기존] 3개 버킷 × 1000+ 샘플로 Shared Encoder + Heads 학습
[새 버킷] "eyes_closed" 정의
  → 기존 데이터에서 threshold 적용 → 50-100건 pseudo-label 자동 생성
  → Encoder 고정 + 새 Bucket Head만 학습
  → 1시간 이내 새 버킷 추가 가능
```

catalog_scoring에서 새 카테고리 추가 시 참조 이미지 촬영 + SIGNAL_RANGES 정의 + 가중치 조정이 필요했던 것의 대체.

### Hierarchical Dawid-Skene Transfer

기존 버킷에서 추정한 모듈 reliability를 새 버킷의 prior로 사용:
- 기존 3개 버킷에서 각 모듈의 confusion matrix 추정 완료
- 새 버킷에 대해 이 prior를 초기값으로 → 10-20건이면 Dawid-Skene 수렴
- DNN transfer (Encoder 고정 + Head만 학습)와 결합하면 두 레벨 시너지

### z 차원 가이드

| 버킷 수 | 권장 z 차원 | 근거 |
|---------|-----------|------|
| 1-5 | 16D | 활성 threshold 수 × 2 ≈ 10-16 |
| 5-10 | 16-24D | 버킷 간 구분 정보량 증가 |
| 10+ | 24-32D | 상향 조정 |

---

# Part 2: 패키지 구조 & 기술 스택

## 새 패키지 구조

기존 PoC (4-stage: Collect→Agree→Pair→Encode)는 폐기하고, DNN Student 방향으로 재작성.

```
libs/visualbind/src/visualbind/
├── types.py           # SourceSpec, HintVector, HintFrame (기존 타입 참고)
├── collector.py       # Teacher 출력 수집 → parquet 저장
├── pseudo_label.py    # Dawid-Skene + majority vote → pseudo-label 생성
├── model.py           # StudentModel (nn.Module: Encoder + Teacher Heads + Bucket Heads)
├── trainer.py         # Trainer (학습 루프, dual loss, VICReg)
├── evaluator.py       # Evaluator (Anchor Set 대비 평가, 메트릭)
├── cli.py             # visualbind analyze / train / eval / label CLI
└── __init__.py
```

### 각 모듈의 책임

| 모듈 | 입력 → 출력 | 의존성 |
|------|------------|--------|
| `collector` | Observation[] → parquet | numpy, pyarrow |
| `pseudo_label` | parquet → labeled parquet | crowdkit |
| `model` | 24D tensor → z, teacher preds, bucket preds | PyTorch |
| `trainer` | labeled parquet → trained model (.pt) | PyTorch |
| `evaluator` | model + anchor set → metrics + HTML report | PyTorch, sklearn, plotly |
| `cli` | CLI args → 위 모듈 조합 | argparse |

## 의존성 관리

```toml
[project]
dependencies = ["numpy", "pyarrow"]

[project.optional-dependencies]
train = ["torch", "crowd-kit", "scikit-learn", "pandas", "plotly"]
dev = ["pytest>=7.0.0"]
```

- 기본 의존성: 수집(collector)만 사용하는 경우 가볍게 유지
- `train` extras: 학습/평가 시에만 필요한 무거운 패키지
- momentscan이 visualbind의 collector만 사용하면 torch를 끌고 오지 않음

## 프레임워크 선택 근거

| 결정 | 선택 | 이유 |
|------|------|------|
| MLP 프레임워크 | **PyTorch** | autograd, 17개 head backprop 수동 구현 불필요, 코드 단순화 |
| Crowds 알고리즘 | **crowdkit** | Dawid-Skene/GLAD/MajorityVote 비교 가능, 검증된 구현 |
| 저장 포맷 | **parquet** | 컬럼 기반 분석 용이, pandas 친화, 학습 데이터 적재 효율적 |
| 시각화 | **Plotly HTML** | momentscan-report 패턴, 단일 파일, 서버 불필요 |

---

# Part 3: CLI & 시각화

## 서브커맨드

```bash
# Day 0: 독립성 분석 + Dawid-Skene go/no-go
visualbind analyze --data ./teacher_outputs.parquet --output ./report.html

# Anchor Set 라벨링용 HTML 생성
visualbind label --data ./teacher_outputs.parquet \
    --buckets warm_smile,cool_gaze,lateral \
    --sample 200 --output ./review.html

# 학습
visualbind train --data ./teacher_outputs.parquet \
    --buckets warm_smile,cool_gaze,lateral \
    --output ./models/student_v1.pt \
    --report ./train_report.html

# 평가
visualbind eval --model ./models/student_v1.pt \
    --anchor ./anchor_set.parquet \
    --report ./eval_report.html
```

실행 흐름: `analyze (go/no-go) → label (Anchor Set) → train (Exp 0-3) → eval (검증)`

## HTML 리포트 내용

### analyze report
- 모듈별 출력 분포 (히스토그램, 24D 각각)
- 모듈 간 상관 행렬 (heatmap)
- N_eff 계산 과정과 결과
- Dawid-Skene reliability vs 수동 AND 비교
- 버킷별 Tier 1/2/3 분포

### train report
- Loss curve (soft, hard, VICReg 각각)
- z 공간 시각화 (t-SNE/UMAP, 버킷별 색상)
- Teacher Head별 재구성 오차 추이
- Bucket Head별 정확도 추이

### eval report
- Student vs catalog_scoring ROC curve
- 버킷별 confusion matrix
- Bootstrap CI 시각화
- 오분류 사례 (프레임 이미지 + Teacher 출력 + Student 판단)

### label (Anchor Set 리뷰)
- 프레임 이미지 + 24D Teacher 출력 시각화
- 적합/부적합 판단 인터페이스
- 진행률 표시

모든 리포트는 **단일 HTML 파일**로 생성. momentscan-report와 동일한 패턴.

---

# Part 4: MVP 경로

## MVP 범위: 3개 버킷 동시 학습

bias 분리를 위해 **2개 이상 버킷 동시 학습이 필수**라는 전문가 합의에 따라:

```
MVP 버킷 (3개):
  - warm_smile  — 정면 따뜻한 미소 (가장 대표적, 참조 25장)
  - cool_gaze   — 정면 시크/무표정 (반대 표정 → 표정 bias 분리)
  - lateral     — 측면 portrait (포즈 기반 → 포즈 bias 분리)
```

이 조합의 장점:
- warm_smile vs cool_gaze: **표정만 다름** → expression/AU 모듈의 bias 분리
- warm_smile vs lateral: **포즈가 다름** → pose 모듈의 bias 분리
- 24D signal 중 어떤 차원이 어떤 구분에 기여하는지 명확히 관찰 가능

## Day 0 — Go/No-Go 프로토콜

MVP 3주 투자 전에, **프로젝트 feasibility를 판단**한다.

### Observer 독립성 분석

```
(1) 14개 모듈의 binary vote에 대해 Fleiss' kappa 계산 (30분)
(2) 모듈 쌍별 Cohen's kappa → 오류 상관 행렬 (1시간)
(3) 500건 수동 라벨링 → 실제 정답 대비 각 vote의 정확도 (2-3일)
    ※ Teacher vote에 대한 blinding 필수 — 프레임 이미지와 버킷 정의만 보고 라벨링

N_eff 계산: 고유값 기반 (equicorrelated 가정 대신)
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
- body.pose, hand.gesture 포함/제외 양쪽 실험 (crowds robustness 검증)
- Hierarchical clustering으로 하위 구조 분석

### Go/No-Go 기준

| N_eff | 판단 |
|-------|------|
| ≥ 3 | MVP 진행 |
| 2 ≤ N_eff < 3 | 합의 기준 상향 + 진행 |
| < 2 | 모델 다양화 또는 근본적 재고 |

### Exp 0 — DNN 없이 Crowds 검증 (30분)

```
Dawid-Skene EM → weighted majority vote
vs Baseline: 수동 threshold AND 조합 (현행)
```

crowdkit을 활용하여 MajorityVote → Dawid-Skene → GLAD 순으로 비교.
Dawid-Skene만으로도 현행 대비 개선되면, crowds 프레이밍이 이 데이터에서 유효한 것.
**실패 시 전체 방향 재검토. 성공 시 DNN으로 진행.**

---

## 실험 설계 (Exp 0-3)

```
Baseline 1: 수동 threshold AND 조합 (현행 파이프라인)
Baseline 2: catalog_scoring Fisher-weighted distance (선형)
Baseline 3: Dawid-Skene weighted majority vote (Exp 0)
```

### Exp 1: Hard-only (binary vote → MLP)

```
입력:  24D 연속 출력
Loss:  BCE(bucket_head(z), binary_vote)
비교:  vs Baseline 1, 2, 3
검증:  "비선형 threshold 조합의 가치"
```

### Exp 2: Soft-only (연속 출력 → MLP + task head)

```
입력:  24D 연속 출력
Loss:  Σ MSE(teacher_head_i(z), continuous_output_i) + BCE(task_head(z), target)
비교:  vs Exp 1
검증:  "연속 출력 활용의 가치" (threshold 정보 손실 복구)
```

### Exp 3: Hybrid (soft + hard 결합 + uncertainty weighting)

```
Loss:  Kendall uncertainty weighting (soft + hard 자동 균형) + λ × VICReg
비교:  vs Exp 1, Exp 2
검증:  ablation study, 학습된 σ² 분석 (어떤 teacher/bucket이 어떤 가중치를 받는지)
```

### Cold Start 실험

Exp 1-3 각각에 대해:
- **Cold start**: Teacher 출력 + binary vote만으로 학습 (catalog_scoring 무관)
- **Warm start**: catalog_scoring의 Fisher weights → Teacher reliability prior로 활용

어느 쪽이 나은지 실증. Cold start로 이기면 가장 강력한 contribution.

## 성공/실패 판단 기준

**Bootstrap CI + Effect Size 병행:**
- Bootstrap (1000회 리샘플링) → 95% 신뢰구간 → CI가 겹치지 않으면 통계적 유의
- Effect size → AUC 2%p 이상 차이면 실무적으로 유의미
- 둘 다 만족해야 "개선"으로 판단

### 성공 기준

1. **N_eff ≥ 3** (go/no-go 전제 조건)
2. **Student MLP ≥ catalog_scoring 성능** (Bootstrap CI 분리)
3. **Week 2 Student > Week 1 Student** (데이터 축적 효과)
4. **Pseudo-label 정확도 > 95%** (Anchor Set 기준)
5. **Teacher reliability 분산 존재** (bias 분리 가능성)

## MVP 타임라인 (~3주)

```
Day 0-1:   Observer 독립성 분석 + N_eff + Exp 0 (go/no-go)
Day 2-3:   Teacher 출력 저장 파이프라인 (collector → parquet)
Day 4-5:   Pseudo-label 생성 (Tier 1/2/3 분류)
Day 6-8:   Anchor Set 수동 검증 (500건, HTML 리뷰 페이지, Teacher vote blinding)
Day 8-10:  MLP Student 학습 (Exp 1 → 2 → 3, cold/warm 비교)
Day 11-12: Validation gate + model swap
Day 13-17: 1주일 파일럿 + 모니터링
```

---

# Part 5: 데이터 파이프라인

## Teacher 출력 수집

momentscan 레벨에서 Observation[]이 나오는 시점에 수집:

```
momentscan → Observation[] → collector → parquet 저장
                           → momentbank (기존 프레임 저장, 변경 없음)
```

visualbind의 collector는 momentscan의 **소비자**로서 signal을 수집한다.
momentbank는 "분석 결과 → 생성"의 브릿지이고, visualbind는 "분석 신호 → 학습"의 소비자.

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

> 기존 20% anchor → 10% hard negative + 10% anchor로 변경 (Round 4 MLOps 리뷰).
> anchor만 유지하면 easy case 편향 — hard negative가 calibration에 필수.

### Anchor Set

시스템 건전성 검증을 위한 **최소 500건의 수동 검증 데이터** (validation set).
**Annotation-free training, annotation-efficient validation** — 학습에는 annotation 불필요,
검증에만 소규모 annotation 사용. Snorkel 논문도 500-2,000건 annotated test set 사용.

- 버킷별 stratified sampling (버킷당 ~170건)
- Tier 1 (고신뢰 합의) + Tier 3 (합의 실패)에서 균형 추출
- `visualbind label` CLI로 HTML 리뷰 페이지 생성 → 브라우저에서 라벨링
- **Teacher vote blinding 필수** — 프레임 이미지와 버킷 정의만 보고 라벨링 (anchoring bias 방지)
- Weekly 100건 추가 검증으로 갱신

---

# Part 6: Drift 적응

## 4가지 Drift와 대응

### Data Drift (입력 분포 변화)

- 원인: 계절, 시간대, 고객층, 조명 환경 변화
- 감지: Teacher 출력 분포의 일별 mean/std 추적
- 대응: Replay buffer의 coverage 비율 자동 조정, 재학습

### Concept Drift (기준 변화) — 의도적 활용

- 원인: 디자인팀 요구사항 변경, 시즌별 수집 기준 변경
- 대응: **새 버킷 추가 (few-shot)** 또는 기존 버킷 가중치 조정
- 이것은 방어가 아니라 **핵심 기능** — 빠른 기준 변경이 시스템의 가치

```
기존: 기준 변경 → threshold 재정의 + 참조 촬영 + 가중치 조정 (수일-수주)
VisualBind: 기준 변경 → 새 Bucket Head 학습 (50건, 1시간 이내)
```

### Teacher Drift (모듈 교체)

- 원인: upstream 모듈 버전 업그레이드 (e.g., 6DRepNet → 다른 모델)
- 감지: Teacher 모듈 버전을 학습 데이터에 태깅
- 대응: 버전 변경 감지 시 해당 모듈의 기존 데이터 무효화 + 재학습
- LayerNorm이 새 출력 분포에 자동 적응

### Preprocessing Drift (전처리 변경)

- 원인: face detection threshold/NMS 파라미터 변경, 이미지 resize 방식 변경, crop 로직 수정
- 위험성: 개별 Teacher 모델은 변경 없으나 **모든 face-crop 기반 모듈 출력이 동시 변화**
  - Teacher Drift 감지 안 됨 (모델 버전 변경 없음)
  - Data Drift로 오인 가능 (입력 이미지 자체는 변경 없음)
- 감지: face crop 통계(종횡비, crop 크기, detection confidence 분포) 독립 모니터링
- 대응: crop 통계 이상 감지 시 전체 Teacher 출력 재검증

### Self-Reinforcing Drift — 구조적 차단

가장 위험한 drift. Student의 편향이 자기 학습을 오염시키는 루프.

**Phase 1 원칙 (MVP):**
> Student는 pseudo-label 생성에 참여하지 않는다.
> Teacher(frozen)만이 유일한 pseudo-label 소스다.

```
Teacher (frozen) → vote → pseudo-label → Student 학습
                                              ↓
                                         Bucket 판단 (운용)
                                              ✗ pseudo-label로 되돌아가지 않음
```

**Phase 2 원칙 (Student 성능 검증 후):**
> Student는 N+1번째 observer로 pseudo-label 생성에 참여하되,
> 14개 frozen Teacher가 앵커로 존재해야 한다.

```
세대 1: 14 Teachers → Student v1 학습
세대 2: 14 Teachers + Student v1 (15번째 observer) → pseudo-label → Student v2 학습
세대 3: 14 Teachers + Student v2 (15번째 observer) → pseudo-label → Student v3 학습
```

Phase 2 안전장치 (Round 4 리뷰 반영):

1. **Student reliability cap**: Dawid-Skene 추정과 무관하게 Student의 reliability를
   median Teacher 이하로 제한. 편향 Student가 과도한 영향력을 갖는 것을 구조적으로 차단.

2. **세대 간 disagreement stability**: Student v_N과 v_{N+1}이 Teachers와 불일치하는
   프레임 집합의 Jaccard similarity 모니터링. Jaccard < 0.7이면 체계적 편향 이동으로 판단, 중단.

3. **Anchor Set 정확도 단조 증가**: 세대 N+1의 Anchor Set 정확도가 세대 N보다 낮으면 즉시 중단,
   세대 N 모델로 롤백.

> 참고: Teacher-Student 간 Cohen's kappa 비교는 부적절 — Teacher는 좁은 신호(AU, pose)를,
> Student는 통합 판단(bucket suitability)을 출력하므로 추상화 레벨이 다름.

Phase 2 진입 조건:
- Student ≥ catalog_scoring (Anchor Set 기준, Bootstrap CI 분리)
- 최소 2주 파일럿 안정성 확인

## 학습 빈도

단계적 전환:

```
Phase C (MVP):    수동 트리거 — visualbind train CLI로 필요 시 실행
Phase A (검증 후): 주간 배치 — 주 1회 재학습
Phase B (안정화 후): 야간 배치 — 매일 재학습
```

학습이 잘 되는지 확인한 후에 자동화. 13K params CPU 학습이므로 비용 부담 없음.

---

# Part 7: 안전장치

## Self-Degrading 방지

자기 개선 루프는 자기 악화 루프가 될 수 있다.

### 모니터링 메트릭 (7개)

1. 합의율 (전체)
2. Teacher-Student 괴리
3. Pseudo-label 생산량
4. Confidence calibration
5. Subgroup 합의율 (조명/각도/시간대)
6. Teacher Reliability 추이
7. 합의 엔트로피

### Rollback 기준 — Z-Score 기반

절대값 임계치 대신 **7일 rolling window 기반 z-score** 사용.
10M+/일 데이터에서 자연 분산이 매우 작으므로, 절대값 기준은 너무 늦거나 너무 민감함.

```
[자동 롤백]
- 합의율 z-score < -5 (2시간 연속) → 즉시 이전 모델로 롤백
- Pseudo-label 생산율 (건/1000프레임 기준) z-score < -4 → 롤백
  ※ 절대량이 아닌 입력 대비 생산율 — 입력량 변동에 둔감

[경보 + 수동 조사]
- 합의율 z-score < -3 (2시간 연속) → 알림
- 3일 연속 성능 하락 추세 (방향성)
- Tier 3 비율 전체의 30% 초과
- 어떤 subgroup에서 합의율 z-score < -3

[학습 중단]
- Student가 catalog_scoring 대비 Anchor Set에서 유의미하게 열위
```

### Blind Spot Registry

Tier 3 (합의 실패) 프레임을 별도 저장.
이것은 모든 teacher가 불확실한 영역 = 시스템의 맹점.
주기적으로 분석하여 새 모듈 추가나 threshold 조정의 근거로 활용.

---

# Part 8: 전환 경로 — visualpath에서 visualbind로

## 전체 진화 로드맵

```
Phase 1 — Signal-Level Distillation (MVP, ~3주)
  Step 1:  3개 버킷, binary vote → MLP (Exp 0-3)
  Step 2:  Multi-bucket 확장, soft+hard dual loss
  Step 3:  Few-shot 새 버킷 (catalog_scoring 대비 비교)
  Step 4:  Bias head 분리 (identifiability 해결 후)
  → catalog_scoring 대체 완료
  → visualpath는 여전히 inference에 필요

Phase 2 — End-to-End Distillation (Phase 1 성공 후)
  Step 5:  Raw image + CNN backbone 도입
  Step 6:  Teacher 출력 재구성 + 버킷 판단 동시 학습
  Step 7:  Teacher 없이 Student만으로 inference
  → visualpath는 학습 시 annotation 공장으로만 사용
  → inference 파이프라인 단순화 (14개 모듈 → 단일 모델)

Phase 3 — Domain Foundation (Phase 2 성숙 후)
  Step 8:  Student의 중간 feature를 downstream task에 활용
  Step 9:  새 도메인 전이 (초상화 → 다른 얼굴 분석 task)
  → visualpath 퇴역
  → Student가 도메인 특화 foundation model
```

각 step에서 이전 step 대비 개선폭 측정 → diminishing return 시 중단.
**Phase 1이 실패하면 Phase 2도 불가** — pseudo-label 품질이 모든 것의 전제.

## catalog_scoring과의 병행 (Phase 1 내)

```
Phase A — Shadow: 두 시스템 병렬 실행, Student 결과 로깅만
Phase B — Hybrid: α × Student + (1-α) × catalog 블렌딩
Phase C — Full: catalog 퇴역 → Student로 완전 대체
```

catalog_scoring의 장점(학습 없이 즉시 동작)을 유지하면서 점진적 전환.

### 전환 가교

| catalog_scoring | VisualBind 대응 |
|----------------|----------------|
| `SIGNAL_FIELDS`, `SIGNAL_RANGES` | BatchNorm이 자동 적응 |
| Fisher ratio weights | Dawid-Skene reliability (Day 0 자동 계산) |
| Category centroids | z 공간의 Bucket Head |
| `compute_importance_weights()` | Teacher reliability 초기값 (warm start 실험) |

## visualpath와의 관계 변화

```
Phase 1: visualpath = inference 필수 + 학습 데이터 생성
         visualbind = catalog_scoring 대체 (메타 판단 레이어)

Phase 2: visualpath = 학습 데이터 생성만 (annotation 공장)
         visualbind = inference 전체 담당 (end-to-end)

Phase 3: visualpath = 퇴역 (새 도메인에서만 bootstrapping 용도)
         visualbind = 도메인 foundation model
```

---

# Part 9: 논문 실험 설계

## 실험 구조

### Phase 1 실험 (Signal-Level)

```
Exp 0:  Dawid-Skene EM vs 수동 AND (DNN 없음, Day 0)
Exp 1:  MLP + hard loss vs Exp 0 + catalog_scoring
Exp 2:  MLP + soft loss vs Exp 1
Exp 3:  MLP + hybrid (soft+hard) vs Exp 1, 2 (ablation)
Exp 4:  Multi-bucket + few-shot transfer vs catalog_scoring 새 카테고리
Exp 5:  Cold start vs warm start (catalog_scoring prior)
```

### Phase 2 실험 (End-to-End, Phase 1 성공 후)

```
Exp 6:  Raw image + CNN backbone vs Phase 1 best (24D MLP)
Exp 7:  Inference 비용 비교 — 14 Teachers + MLP vs CNN 단일 모델
Exp 8:  Teacher 수 vs 성능 — N개 Teacher subset으로 학습 시 성능 변화
```

### Exp 4 — 논문의 가장 Compelling한 Figure

```
기존 3개 버킷으로 학습된 Student
  → 새 버킷 1-2개 (hold-out)
  → N-shot (20, 50, 100, 200)에서의 성능 곡선
  → vs catalog_scoring의 centroid matching (7장 참조)
```

"N-shot에서의 새 버킷 성능 곡선"이 reviewer에게 가장 인상적인 결과가 될 수 있다.

## 측정 지표

| 지표 | 설명 |
|------|------|
| AUC-ROC | 버킷 적합성 분류 성능 |
| Accuracy@threshold | 현행 threshold 기준 정확도 |
| catalog_scoring 대비 상대 개선 | Fisher distance vs z-space distance |
| Few-shot learning curve | N-shot별 성능 곡선 |
| Temporal improvement | Week 1 vs Week 2 vs Week 4 성능 변화 |

---

# Part 10: 열린 질문

## 설계 결정이 필요한 것

| 질문 | 선택지 | 현재 방향 |
|------|--------|----------|
| z 차원 | 16 / 24 / 32 | 16D로 시작, 버킷 수에 따라 확장 |
| Soft/Hard α | 0.3 / 0.5 / 0.7 | Exp 3에서 grid search |
| Bucket Head 구조 | Linear / 1-layer MLP / 2-layer MLP | 1-layer MLP |
| N+1 observer 참여 시점 | Phase 2 진입 조건 | Student ≥ catalog_scoring + 2주 안정성 |

## 검증이 필요한 가정

| 가정 | 검증 방법 |
|------|----------|
| 다수 observer 합의 = 높은 정확도 pseudo-label | Anchor Set 200건 수동 검증 |
| Student가 teacher를 넘을 수 있다 | Exp 1-3에서 catalog_scoring 대비 비교 |
| Daily data가 충분한 diversity를 제공한다 | 데이터 분포 분석 (계절, 시간대, 고객 다양성) |
| Observer 독립성이 충분하다 | Day 0 N_eff 분석 |
| Few-shot 버킷 추가가 가능하다 | Exp 4에서 hold-out 버킷 실험 |

## 알려진 제약 (Limitation)

1. **Identifiability**: GT 없이 truth와 bias를 완벽히 분리하는 것은 수학적으로 underdetermined. 실증 + structural regularization으로 보완
2. **Observer 독립성 위반**: 14개 모델의 공유 backbone/데이터 + 입력 의존성 체인. N_eff는 14보다 상당히 낮을 것 (5-7개 추정)
3. **Catastrophic forgetting**: 새 데이터 학습 시 이전 학습 망각 위험. Replay buffer + 주기적 전체 재학습으로 대응
4. **버킷 정의 자체의 타당성**: "warm_smile 정면" 버킷이 고객 만족과 대응하는가는 VisualBind 범위 밖
