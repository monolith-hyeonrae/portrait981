# VisualBind 전문가 리뷰 통합 보고서

> 4라운드, 14명의 전문가가 독립적으로 visualbind를 분석하고 교차 검증한 결과를 **주제별**로 통합.
> Round 1 (2026-03-12): ML, Sensor Fusion, Systems — 3명 (PoC 코드 리뷰)
> Round 2 (2026-03-12): ML/AI, Signal Processing, Neuroscience, Architecture — 4명 (PoC 코드 리뷰)
> Round 3 (2026-03-16): ML, AI/DL, ISP — 3명 (PoC 코드 리뷰 + 로드맵)
> Round 4 (2026-03-16): Crowds, KD/Self-Training, Production ML, Medical AI — 4명 (Reframe 리뷰)

---

# Part 1: 현재 문제점

## 1.1 Agreement 함수의 근본적 결함 — `a * b`

> 합의: **10/10 전원 일치** (Round 1-3 전체)

현재 `_positive_agreement(a, b) = a * b`는 3가지 수준에서 실패한다.

### 수학적 문제: both-low bias

```
b=1.0 │ 0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
b=0.5 │ 0.0  0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
b=0.1 │ 0.0  0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10
       a=0.0 ─────────────────────────────────────────── a=1.0
```

Neutral 상태 (em_happy=0.1, AU12=0.1) → agreement=**0.01**. Positive threshold(0.5 이상)를 절대 넘지 못해 **체계적으로 negative pool에 빠진다**.

- **R1 ML**: "주석과 구현의 불일치. both low → ~1.0이라고 적혀 있지만 실제는 0.01"
- **R2 Signal**: "등고선이 쌍곡선. agreement=0.25를 넘으려면 둘 다 0.5 이상이어야 한다"
- **R2 Neuro**: "Inverse effectiveness를 정면으로 위반. 약한 신호의 일치야말로 통합 가치가 가장 높은 순간"
- **R3 ML**: "Gradient flow에서도 문제. ∂(ab)/∂a = b이므로, a가 작으면(neutral) gradient도 vanish. Neutral 상태의 representation이 random에 가깝게 남는다"

### 대안 함수 비교

| 시나리오 | (a, b) | a*b | 1-\|a-b\| | 2ab/(a+b) |
|---------|--------|-----|-----------|-----------|
| Both high (happy+AU12) | (0.9, 0.8) | 0.72 | 0.90 | 0.85 |
| Both low (neutral) | (0.1, 0.1) | **0.01** | **1.00** | 0.10 |
| Mixed (conflict) | (0.9, 0.1) | 0.09 | 0.20 | 0.18 |
| Both medium | (0.5, 0.5) | 0.25 | **1.00** | 0.50 |

**Gradient flow 비교** (R3 ML):
- `a * b`: ∂/∂a = b. Neutral에서 gradient vanish
- `1 - |a - b|`: ∂/∂a = -sign(a-b). 상수 gradient이나 activation 정보 없음
- `2ab/(a+b)`: ∂/∂a = 2b²/(a+b)². Activation에 비례하나, both-low에서 여전히 약함

**결론 (전원 합의)**: 단일 함수로는 불충분. **2-channel 접근** 필요.

### 2-Channel 해법

```
Channel 1: similarity  = 1 - |a - b|   → "두 값이 얼마나 비슷한가"
Channel 2: activation  = (a + b) / 2    → "그 일치가 의미 있는 수준인가"
```

| 영역 | sim | act | 의미 | PairMiner 처리 |
|------|-----|-----|------|---------------|
| (1.0, 0.9) | 높음 | 높음 | co-activation | positive |
| (1.0, 0.1) | 높음 | 낮음 | co-absence (neutral) | **quiet positive** |
| (0.2, 0.5) | 낮음 | 중간 | conflict | negative |
| (0.8, 0.05) | 높음 | 극저 | noise | 제외 |

Neutral 상태가 **자체 positive pool을 형성**할 수 있어 both-low 문제의 근본적 해결.

---

## 1.2 Positive 선택 기준의 치명적 오류

> 합의: **10/10 전원 일치** (Round 1-3 전체)

현재 `pairing.py`에서 positive를 "agreement score가 anchor와 **가장 가까운** high-agreement 프레임"으로 선택한다.

```python
pos_idx = min(other_high, key=lambda j: abs(scores[j] - scores[anchor_idx]))
```

- **R1 ML**: "Agreement score 0.72인 happy_frontal과 0.71인 surprise_open이 pair가 될 수 있다. Agreement score는 '동의 정도'이지 '상태 유사성'이 아니다. 근본적 설계 오류."
- **R3 AI/DL**: "Co-training 이론의 fundamental assumption — positive pair는 의미적으로 유사해야 한다 — 이 위반된다"

### 해법: Signal vector cosine similarity

```python
anchor_vec = np.array(vectors[anchor_idx])
pos_idx = max(
    other_high,
    key=lambda j: float(np.dot(
        anchor_vec / (np.linalg.norm(anchor_vec) + 1e-8),
        np.array(vectors[j]) / (np.linalg.norm(np.array(vectors[j])) + 1e-8)
    )),
)
```

Agreement score는 "이 프레임이 신뢰할 만한가" 판단, cosine similarity는 "두 프레임이 같은 상태인가" 판단. **관심사의 분리.**

---

## 1.3 Softmax 이중 적용 — MI 40-80% 파괴

> 합의: **R2-3 6/6 전원 일치**

HSEmotion은 `logits=False`로 호출하여 **이미 softmax가 적용된 확률 벡터**를 반환한다. HintCollector에서 `normalize="softmax"`를 다시 적용하면 이중 softmax.

원본: `p = [0.80, 0.10, 0.05, 0.05]` → 이중 softmax: `[0.395, 0.196, 0.186, 0.186]`

- Shannon entropy: 0.92 → 1.36 bits (**48% 증가 = 판별 정보 절반 손실**)
- em_happy=0.80 "확실한 미소" vs 0.60 "애매한 미소": 이중 softmax 후 차이 0.20→0.05
- Agreement: 원래 `0.80 × 0.80 = 0.64` → `0.395 × 0.395 = 0.156` (**4배 축소**)
- **R3 AI/DL**: "KL divergence 기준, 원본 분포 첨도(kurtosis)에 비례하여 MI 최대 80%+ 손실 가능"

**해법**: `normalize="none"`. 코드 1줄 변경.

---

## 1.4 Pose Range 정규화 — 유효 해상도 1/6

> 합의: **R2-3 ISP+Signal**

MinMax(-90, 90)에서 테마파크 고객의 실제 yaw 범위(±30도)는 정규화 후 0.33-0.67로 압축.

| 신호 | 설정 range | 실제 99% 분포 | 유효 해상도 |
|------|-----------|-------------|-----------|
| head_yaw | (-90, 90) | ±30도 | **33%** |
| head_pitch | (-90, 90) | ±15도 | **17%** |
| head_roll | (-90, 90) | ±10도 | **11%** |

Pose 신호 판별력 축소 → expression-AU agreement가 전체 score 독점. **Pose 정보의 embedding 기여가 사실상 0.**

**해법**: `sigmoid(center=0, scale=0.05)`. ±30도에서 0.18-0.82 해상도.

---

## 1.5 flat_vector 차원 불일치 — Source 누락 시 Crash

> 합의: **10/10 전원 일치**

`flat_vector()`는 present한 source만 포함하므로, source 누락 시 벡터 차원이 줄어들어 `_W` 행렬과 불일치 → **crash**.

**해법**: `all_sources` 인자를 받아 missing source를 zero-padding.

---

## 1.6 Observer Independence 위반 — Expression↔AU의 Tautology

> 합의: **R1-3 ML+Signal+AI+ISP**

Expression classifier(HSEmotion)와 AU detector(LibreFace)는:
- **같은 입력** (동일 face crop)을 받아
- **같은 물리 현상** (안면 근육 수축)을 관측하되
- **다른 추상 수준**에서 출력 (AU = 개별 근육, expression = 의미론적 해석)

이것은 **부분적 tautology**. Agreement가 높다는 것이 "독립적 검증"이 아니라 "동어반복에 가까운 확인".

- **R2 Signal**: "진정으로 독립적인 검증 — body.pose와 expression, CLIP aesthetic과 AU — 에 더 높은 weight를 줘야 한다"
- **R3 ML**: "Independence-adjusted weighting: $w_{ij}^{\text{adj}} = w_{ij} \cdot (1 - \rho_{ij})$. MI로 ρ 추정 시 expression↔AU는 ~0.7, weight 0.3배 감소"

---

## 1.7 Linear Projection의 표현력 한계

> 합의: **R1+R3 ML+AI**

`TripletEncoder`는 단일 linear layer (`W @ x`) + L2 normalization만 사용.

- **R1 ML**: "AU6가 높고 AU12가 높을 때만 진짜 미소(Duchenne)라는 conjunction은 linear projection으로 원천적 학습 불가능"
- **R3 AI/DL**: "$f(x) = Wx/\|Wx\| \in S^{d-1}$. 비선형 분리 불가. $x_{\text{AU6}} \cdot x_{\text{AU12}}$ 곱 항이 필요하며 linear으로 표현 불가"
- **PoC에서 수용 가능** (R1 ML): "11차원 저차원에서 cluster가 linearly separable한 경우 동작. 94.2% kNN은 이 조건 만족의 증거"

---

## 1.8 Triplet Loss의 구조적 한계

> 합의: **R1-3 ML+AI**

| 문제 | 설명 |
|------|------|
| O(1) negative | Batch 내 1개 negative만 활용. InfoNCE는 O(N)개 |
| Gradient starvation | Easy triplet(d_neg > d_pos + margin)에서 gradient=0. 학습 후반 대부분 easy |
| Margin 임의성 | margin=0.3은 arbitrary. Collapse risk는 낮으나 충분한 separation도 강제 못함 |
| Uniformity 부재 | Alignment에만 기여, uniformity에 간접적. Dimensional collapse 위험 |

**R3 AI/DL**: "300 epoch 학습은 gradient starvation의 증거. Semi-hard mining 적용 시 50-100 epoch이면 충분"

---

## 1.9 IB 관점의 Stage 2 병목

> 출처: **R3 AI/DL**

Stage 2 (Agreement)에서 11차원 신호 공간을 **스칼라 agreement score 하나로 붕괴**. Threshold 기반 이진 분할로 사용되므로 실효 정보량은 ~1-2비트.

$$H(S) \leq \log_2(|\text{bins}|)$$

**Agreement score가 Y(true state)에 대한 sufficient statistic이 아님** — 현 시스템의 근본적 병목.

---

## 1.10 FACS 커버리지 부족

> 합의: **R1-2 Sensor+Neuro, R3 ISP**

| 감정 | FACS AU 조합 | 현재 커버 | 커버리지 |
|------|-------------|----------|---------|
| Happy | AU6 + AU12 | AU6, AU12 | **부분** (conjunction 미표현) |
| Surprise | AU1+2+5+25+26 | AU25, AU26 | **부분** (3/5 누락) |
| Anger | AU4+5+7+23+24 | 없음 | **0%** |
| Disgust | AU9+15+25 | 없음 | **0%** |
| Fear | AU1+2+4+5+20+26 | 없음 | **0%** |
| Sadness | AU1+4+15 | 없음 | **0%** |

Demo의 4개 CrossCheck는 6가지 기본 감정 중 happy, surprise만 커버.

---

## 1.11 Gradient 근사의 누적 오차

> 합의: **R1+R3 ML+AI**

`_step` 메서드에서 L2 normalization gradient를 건너뛴다.

- **R3 AI/DL**: "Jacobian $J = (I - \hat{e}\hat{e}^T)/\|e\|$을 skip하면, (1) gradient 방향 왜곡 — $\hat{e}$에 수직인 성분만 유효한데 $\hat{e}$ 방향도 포함, (2) 크기 왜곡 — norm이 큰 embedding일수록 과대추정. PoC에서 작은 LR(0.005)로 '우연히 작동'하지만 '올바르게 수렴'이 아님"

---

## 1.12 각 옵저버의 Noise Profile과 Failure Mode

> 출처: **R3 ISP**

### HSEmotion (face.expression)
- Face crop bbox jitter(2-5px) → 동일 표정에서 확률값 0.05-0.15 진동
- 측면(yaw>25도): neutral 확률 인위적 상승 (학습 데이터 정면 편향)
- 모션 블러: 모든 감정 확률 uniform 수렴 (각 ~0.125)

### LibreFace (face.au)
- AU12(lip corner) SNR 양호(±0.3), AU6(cheek raiser) 조명 극도 민감(±0.8)
- DISFA 학습 데이터 백인 중심 → 한국인 포함 다양한 인종에서 AU6, AU9 체계적 과소추정
- FACS intensity는 지각적 로그 스케일 → MinMax(0,5) 정규화는 비선형성 무시

### 6DRepNet (head.pose)
- 공식 MAE ~3.5도, production 4-6도
- 프레임 간 temporal jitter ~1-2도(정면), ~3-5도(측면)

### 환경 요인의 비대칭적 영향

| 조건 | HSEmotion | LibreFace AU | 6DRepNet | Agreement 영향 |
|------|-----------|-------------|----------|----------------|
| 모션 블러 | uniform 수렴 | intensity→0 | MAE 급증 | **모든 옵저버 동시 실패** |
| 선글라스 | happy -15~25% | AU6 완전 누락 | 정상 | Duchenne 판별 불가 |
| 마스크 | happy/surprise 구분 불가 | AU12,25,26 불가 | 정상 | CrossCheck 대부분 무효 |

### 테마파크 특수성
- 롤러코스터 하차: 땀 specular reflection → detection confidence↓
- 야간 LED (3000K-6500K): auto WB hunting → 피부색 프레임간 변동
- 어린이: 얼굴 비율 상이 → HSEmotion accuracy 10-15%↓

---

# Part 2: 학술적 포지셔닝과 논문 전략

## 2.1 기존 연구 대비 Novelty 평가

> 합의: **R1-3 ML+AI 전원** — "현재 novelty만으로는 NeurIPS/ICML main track 부족"

| 기법 | VisualBind와의 관계 | 차별점 |
|------|-------------------|--------|
| CLIP/ImageBind | raw modality alignment | Frozen observer + rule-based agreement. 구조적으로 다름 |
| **Snorkel** | labeling function ensemble → probabilistic label | **CrossCheck = labeling function. 핵심 구조가 동치** |
| Co-training (Blum & Mitchell 1998) | multi-view consistency | N-view 일반화이나 기존 확장 연구 존재 |
| SimCLR/MoCo/BYOL | augmentation-driven pair | Agreement-driven pair가 유일한 진정한 차별점 |
| Prototypical Networks | centroid distance | catalog_scoring이 정확히 이것 |

**핵심 문제** (R3 ML): "Frozen observer agreement as weak supervision은 Snorkel의 labeling function과 구조적으로 동치(isomorphic). CrossCheck = labeling function, AgreementEngine = label model."

### 차별화 가능한 진정한 novelty

1. **Frozen ML 모듈의 출력을 입력으로 사용** (Snorkel은 텍스트/규칙 기반 labeling function)
2. **Continuous signal에서의 agreement** (Snorkel은 discrete label)
3. **Contrastive learning의 soft supervision으로 활용** (Snorkel은 probabilistic label → supervised learning)

## 2.2 Top-tier 게재를 위한 3가지 경로

> 출처: **R3 ML**

**경로 A: 이론적 contribution**

Co-training PAC bound를 N-observer로 확장:

$$\text{err}(h) \leq \frac{1}{\binom{N}{2}} \sum_{i<j} \text{agreement\_err}_{ij}(h) + O\left(\sqrt{\frac{d}{Nm}}\right)$$

Observer 수 $N$에 따라 $\sqrt{1/N}$으로 bound 감소 → novelty.

**경로 B: Empirical contribution (벤치마크 + scaling law)**

"Multi-Observer Agreement Benchmark (MOAB)":
- AffectNet / BP4D / DISFA에서 6가지 baseline 대비 비교
- Observer 수 {2, 4, 7, 14}에 따른 scaling law 실증

**경로 C: Application paper**

portrait981 실제 운영 데이터에서 "label 없이 학습한 embedding이 hand-crafted scoring을 능가" 입증.

## 2.3 논문 제목 후보

1. **"Learning from Frozen Observers: Agreement-Supervised Contrastive Representations without Labels"**
2. **"Observer Agreement is All You Need: Self-Supervised Metric Learning from Multi-Model Consensus"**
3. **"VisualBind: Binding Multi-Observer Signals via Weak Agreement Supervision"**

## 2.4 게재 전략

| 목표 | 필요 완성도 | 예상 시기 |
|------|-----------|----------|
| FG Workshop / ECCV Workshop | Tier 1 + synthetic + 내부 데이터 | 4-6주 |
| FG Main | Tier 1-2 + BP4D/DISFA + ablation | 2-3개월 |
| ECCV/CVPR Main | Tier 1-3 + AffectNet + 이론 + human eval | 4-6개월 |
| NeurIPS/ICML | Tier 1-3 + N-observer bound + scaling law | 6-9개월 |

## 2.5 실험 설계

### Ablation Study (R3 AI/DL)

| Component | 제거/변경 | 기대 효과 |
|-----------|----------|----------|
| Agreement stage | Random pair selection | Embedding 품질 급감 → agreement 핵심 기여 입증 |
| CrossCheck rules | 순수 통계적 합의만 | Domain knowledge 주입 효과 정량화 |
| Agreement function | `a*b` vs `1-|a-b|` vs 2-channel | Both-low 처리 효과 |
| Linear vs MLP | 2-layer MLP | 비선형 projection 이득 |
| Number of observers | 1, 2, 3 subsets | Observer 수 vs embedding 품질 관계 |

### 평가 메트릭 (kNN accuracy 외 필수)

| 메트릭 | 의미 |
|--------|------|
| Silhouette Score | Cluster compactness vs separation |
| NMI / ARI | Cluster vs ground truth |
| Retrieval@K | Top-K retrieval precision |
| Alignment & Uniformity | Wang & Isola 2020 |
| **Human evaluation** | 가장 중요 — 최종 portrait 주관적 품질 |

### Baseline

Catalog Scoring, Raw PCA, SimCLR-style augmentation, Ensemble average, Random embedding

### Real-world Dataset

| 데이터셋 | 용도 | 규모 |
|----------|------|------|
| 981파크 내부 데이터 | Primary evaluation | ~100 비디오, ~50만 프레임 |
| AffectNet | Emotion benchmark | 400K, V-A annotation |
| DISFA | AU intensity | 27명, 4845 frames |
| BP4D | Spontaneous expression | 41명, AU + emotion |

---

# Part 3: 개선 방향 — 학습 메커니즘

## 3.1 InfoNCE with Agreement Weight

> 합의: **R1-3 ML+AI** — "Triplet loss는 2024년 이후 top-tier 미수용"

$$\mathcal{L} = -\sum_{i} w_i \cdot \log \frac{\exp(\text{sim}(z_i, z_{p_i}) / \tau)}{\sum_{j} \exp(\text{sim}(z_i, z_j) / \tau)}$$

$w_i = a_i^{\gamma}$, $\gamma$는 focusing parameter. Agreement 높은 샘플에 더 큰 weight.

대안: Sample-wise temperature $\tau_i = \tau_{\text{base}} \cdot (1 + \alpha(1-a_i))$. Agreement 높을수록 sharp gradient.

## 3.2 Collapse 방지: VICReg Regularization

> 합의: **R3 ML+AI**

$$\mathcal{L}_{\text{reg}} = \lambda_v \cdot v(Z) + \lambda_c \cdot c(Z)$$

- **Variance**: $v(Z) = \frac{1}{d} \sum_j \max(0, \gamma - \text{Std}(z^j))$
- **Covariance**: $c(Z) = \frac{1}{d} \sum_{i \neq j} [C(Z)]_{ij}^2$

InfoNCE보다 collapse에 robust, batch size에 덜 민감.

## 3.3 Curriculum Learning Schedule

> 합의: **R1+R3 ML**

```python
def threshold_schedule(epoch, max_epochs):
    progress = epoch / max_epochs
    if progress < 0.3:      return 0.85, 0.15   # Phase 1: 확실한 샘플만
    elif progress < 0.7:                          # Phase 2: 점진적 확장
        t = (progress - 0.3) / 0.4
        return 0.85 - 0.25 * t, 0.15 + 0.15 * t
    else:                    return 0.55, 0.35   # Phase 3: ambiguous 포함
```

## 3.4 Hard Negative Mining — 2차원 결합

> 출처: **R3 ML**

$$n^* = \arg\max_{n \in \mathcal{N}} [\beta \cdot \text{sim}(z_a, z_n) + (1-\beta) \cdot a_n]$$

β는 epoch에 따라 0 (agreement space 우선) → 1 (embedding space 우선)로 annealing.

## 3.5 Linear → MLP Projection Head

> 합의: **R1+R3 ML+AI**

```
input(D) → Linear(D, 2D) → BN → ReLU → Linear(2D, D) → BN → ReLU → Linear(D, embed_dim)
```

SimCLR 발견: Projection head output이 아니라 **중간 representation**을 downstream에 사용.

전환 시기: Real-world 데이터에서 linear의 kNN accuracy가 plateau에 도달했을 때.

---

# Part 4: 개선 방향 — 신호 처리와 도메인 지식

## 4.1 최적 정규화 전략

> 합의: **R2-3 Signal+ISP**

| 옵저버 | 현재 | 문제 | 제안 | 근거 |
|--------|------|------|------|------|
| face.expression | softmax | 이중 적용, MI 50%↓ | **`none`** | 이미 확률 |
| face.au | minmax(0,5) | 비선형 무시 | **`sigmoid(2.0, 0.8)`** | FACS 지각 비선형성 |
| head.pose yaw | minmax(-90,90) | 유효 range 33% | **`sigmoid(0, 0.05)`** | ±30도에서 0.18-0.82 |
| head.pose pitch | minmax(-90,90) | 유효 range 17% | **`sigmoid(0, 0.1)`** | ±15도 focus |
| face.quality blur | minmax(0,500) | 데이터 종속 | **adaptive percentile** | face.baseline 활용 |

## 4.2 Pose-dependent Confidence Gating

> 합의: **R1-3 전체**

```
confidence_multiplier(yaw) = {
    expression: 1.0 if |yaw|<15, 선형 감쇠 if 15≤|yaw|<30, 0.55 이하 if ≥30
    AU:         1.0 if |yaw|<20, 선형 감쇠 if 20≤|yaw|<35, 0.40 이하 if ≥35
    pose:       1.0 if |yaw|<60
}
```

CrossCheck의 static relation으로 불가 → AgreementEngine에 `GatingRule` 프로토콜 추가.

## 4.3 Conflict/Quiet/Missing 3-way 분류

> 합의: **R1-2 Sensor+Signal+Neuro, R3**

현재: agreement 낮음 = "conflict" or "quiet" 구분 불가.

- **Conflict**: em_happy=0.9 + AU12=0.1 → 모순 (hard negative로 가치 높음)
- **Quiet**: em_happy=0.1 + AU12=0.1 → 둘 다 비활성 (neutral positive)
- **Missing**: em_happy=? + AU12=0.3 → 한쪽 결측

2-channel activation level로 quiet 구분. Dempster-Shafer의 conflict measure K 영감.

## 4.4 Duchenne Smile Derived Signal

> 합의: **R2 Neuro + R3 ISP**

현재 pairwise CrossCheck로 conjunction(AU6 AND AU12) 표현 불가.

```python
def _compute_duchenne(au_hint):
    return min(au_hint.get("AU6"), au_hint.get("AU12")) / 5.0
```

HintFrame에 추가 후 `CrossCheck("face.expression", "em_happy", "face.au_derived", "duchenne")`.

## 4.5 FACS 6-Emotion 자동 CrossCheck

> 합의: **R1-2 Sensor+Neuro, R3 ISP**

FACS 매핑 테이블에서 자동 생성: 4개 → ~16개 CrossCheck.

가용 AU(LibreFace DISFA subset: 12개)에서 AU7, AU20, AU23, AU24 누락 고려 시 실질 ~16개.

## 4.6 Antagonistic AU Pairs (이상치 탐지)

> 출처: **R3 ISP**

| AU 쌍 | 동시 불가 이유 |
|-------|--------------|
| AU1 ↔ AU4 | corrugator vs frontalis (이마 길항) |
| AU12 ↔ AU15 | zygomaticus vs depressor (입꼬리 길항) |
| AU25 ↔ AU24 | 벌림 vs 다묾 (입술 길항) |

동시 intensity > 2.0 → AU detector 오류. Agreement에 penalty 또는 pair mining 제외.

## 4.7 Temporal Dynamics 보정

> 합의: **R1-2 Signal+Neuro+Arch**

- AU onset lag: spontaneous smile에서 AU6는 AU12보다 67-170ms (2-5프레임) 지연
- 프레임 독립 비교에서 onset 구간의 false conflict 발생
- Apex window pooling (5-15프레임) 또는 EMA smoothing (α=0.3, ~3프레임 lookback)

## 4.8 Observer Independence-adjusted Weighting

> 출처: **R3 ML**

$$w_{ij}^{\text{adj}} = w_{ij} \cdot (1 - \rho_{ij})$$

Expression↔AU MI ~0.7 → weight 0.3배 감소. Expression↔body.pose MI ~0.1 → weight 0.9배 유지.

---

# Part 5: 상용화 임팩트

## 5.1 Frozen Model Orchestration의 Structural Advantage

> 합의: **R1-3 전체**

| 측면 | End-to-end Fine-tuning | VisualBind (Frozen + Agreement) |
|------|----------------------|-------------------------------|
| GPU 비용 | 전체 모델 gradient | numpy agreement + linear layer만. **1000배+ 절감** |
| 모델 교체 | 재학습 필수 | Observer 교체 → CrossCheck만 수정 |
| 배포 | 단일 거대 모델 | 각 observer 독립, embedding만 업데이트 |
| 디버깅 | Black box | Agreement score = 해석 가능 중간 단계 |
| 규제 | 모델 전체 감사 | 각 observer + rule 개별 감사 |

## 5.2 시스템 통합

> 출처: **R1 Systems + R2 Architecture**

- **Inference latency**: `encode()` = matmul 1회 + L2 norm. 마이크로초 단위. 전체 파이프라인 대비 무시
- **메모리**: `_W` 행렬 168 float64 + HintFrame 캐시. KB 단위
- **학습**: ~10,000 프레임, 200 epoch, numpy 기반. 수십 초. 일일 야간 배치
- **extract.py 대체**: 스코어링 관련 ~130줄 대체 가능, gate/classify/identity ~230줄 유지

## 5.3 Catalog → VisualBind 전환 전략

> 합의: **R1-2 Systems+Architecture, R3**

**Phase A: Shadow Mode** — 기존 catalog 그대로 + visualbind 병렬 실행, 결과 로깅만

**Phase B: Hybrid Mode** — `final = (1-α) × catalog + α × bind`. α를 0→1로 점진적

**Phase C: Full Replacement** — catalog scoring 제거, bind embedding 기반

롤백: α=0.0으로 즉시 catalog-only 복귀.

## 5.4 도메인 확장 가능성

- **의료영상** (가장 빠른 확장): 레이블 비용 극고, frozen CAD 모델 다수, explainability 규제
- **자율주행**: LiDAR + Camera + Radar observer agreement
- **산업검사**: visual + infrared + ultrasonic 센서 합의
- **스포츠 분석**: Pose + ball tracker + formation analyzer 합의

## 5.5 Production 필수 사항

- **`_W` save/load**: numpy `.npz` 파일, 수 KB
- **Adaptive normalization**: face.baseline Welford stats 활용 (R3 ISP)
- **Calibration drift 감지**: agreement 분포 일별 mean/std 추적, z>3 자동 플래그 (R3 ISP)
- **Per-CrossCheck drift**: 특정 CrossCheck만 하락 → 해당 옵저버 문제 (R3 ISP)

---

# Part 6: 통합 로드맵 v3

## Tier 1: 즉시 수정 (논문 + 상용 모두 blocking)

| # | 작업 | 난이도 | 논문 영향 | 상용 영향 | 합의 |
|---|------|--------|----------|----------|------|
| 1 | **Positive 선택 → signal cosine similarity** | 낮음 | 매우 높음 | 매우 높음 | 10/10 |
| 2 | **Agreement 2-channel (similarity + activation)** | 낮음 | 매우 높음 | 매우 높음 | 10/10 |
| 3 | **flat_vector zero-padding** | 낮음 | 낮음 | 매우 높음 | 10/10 |
| 4 | **Softmax 이중 적용 제거** | 매우 낮음 | 중간 | 높음 | 6/6 |
| 5 | **Pose range → sigmoid** | 낮음 | 낮음 | 높음 | R3 ISP |

## Tier 2: 핵심 개선 (논문 contribution 확보)

| # | 작업 | 난이도 | 논문 영향 | 상용 영향 | 합의 |
|---|------|--------|----------|----------|------|
| 6 | **Triplet → InfoNCE + agreement weight** | 중간 | 매우 높음 | 중간 | ML+AI |
| 7 | **VICReg regularization** | 중간 | 높음 | 중간 | ML+AI |
| 8 | **Conflict/quiet/missing 3-way** | 중간 | 높음 | 높음 | 7/10 |
| 9 | **Pose-dependent confidence gating** | 중간 | 중간 | 높음 | 전체 |
| 10 | **Linear → MLP projection head** | 중간 | 높음 | 중간 | ML+AI |

## Tier 3: 확장 (full contribution paper)

| # | 작업 | 난이도 | 논문 영향 | 상용 영향 |
|---|------|--------|----------|----------|
| 11 | Curriculum learning | 중간 | 중간 | 중간 |
| 12 | FACS 6-emotion 자동 CrossCheck | 낮음 | 중간 | 중간 |
| 13 | Duchenne smile derived signal | 낮음 | 중간 | 높음 |
| 14 | Temporal smoothing | 중간 | 중간 | 중간 |
| 15 | Observer independence weighting | 중간 | 높음 | 중간 |
| 16 | Hard negative mining | 중간 | 중간 | 중간 |
| 17 | N-observer scaling law | 높음 | 매우 높음 | 낮음 |
| 18 | Adaptive normalization | 중간 | 낮음 | 높음 |

## 실행 순서

```
Phase I  (1-2주): Tier 1 전부 (#1-5)
  → 이것 없이는 이후 모든 개선의 효과 측정이 오염됨

Phase II (1-2주): #6 (InfoNCE) + #7 (VICReg) + #10 (MLP)
  → 학습 인프라 안정화

Phase III (2-3주): #8, #9, #11-16
  → Domain knowledge 개선

Phase IV (2-4주): #17, #18 + 실험/논문 작성
  → Scaling law, real-world dataset, ablation study
```

---

# Part 7: Round 4 — Reframe 리뷰 (2026-03-16)

> `docs/planning/visualbind-reframe.md` 기반.
> 4명: Crowds (Learning from Crowds 전문가), Distill (KD/Self-Training 전문가),
> MLEng (Production ML 전문가), Medical (Medical AI 전문가).
> 상호 토론 + 사용자 참여 논의 포함.

## 7.1 핵심 프레이밍 전환

### 기존 → 신규

| 기존 (Round 1-3) | 신규 (Round 4) |
|------------------|---------------|
| "합의는 레이블이다" + Binding Problem 메타포 | Learning from Crowds + bias 모델링 |
| Contrastive pair 구성 + soft contrastive loss | DNN Student + dual-mode loss (soft+hard) |
| 범용 프레임워크 주장 | 먼저 초상화에서 catalog_scoring을 넘긴 후 일반화 |
| "Few-shot prototype matching" | "Annotation-free domain adaptation via multi-observer evidence" |
| 4-stage pipeline (Fusion→Agreement→Pairing→Encoding) | Student encoder + teacher heads + bucket heads |

### 4명 전원 동의 사항

1. **Learning from Crowds 프레이밍이 기존보다 학술적으로 견고**
2. **DNN Student + bias 모델링 방향이 올바름** (linear projection 한계 해소)
3. **Daily data stream이 진짜 차별점**
4. **catalog_scoring → VisualBind 진화 경로가 자연스러움**
5. **MVP = Teacher 출력(21D) → MLP Student** (GPU 불필요, CPU 10-30분)

## 7.2 "같은 질문 vs 다른 질문" 논쟁과 해소

### 초기 논쟁 (Round 4 전반)

| 입장 | 주장자 | 근거 |
|------|--------|------|
| 같은 질문 (crowds) | 사용자 | AU, Pose, Expression 모두 "좋은 초상화인가?"에 대해 자기 관점에서 답 |
| 다른 질문 (multi-modal) | Crowds, Medical, MLEng | AU는 근육, Pose는 각도 — 다른 질문 |

### 사용자의 핵심 반론: Threshold가 모듈을 Weak Classifier로 변환

> "모듈이 원래 다른 질문에 답하도록 설계되었더라도, 운용 맥락에서 threshold를 통해
> 같은 질문의 weak classifier로 변환되고 있다."

```
face.detect confidence > 0.8  → ✓/✗  (weak classifier)
pose yaw ∈ [30°, 60°]        → ✓/✗  (weak classifier)
expression happy > 0.6        → ✓/✗  (weak classifier)
ALL 만족 → "이 버킷에 적합"
```

Noise의 이중 구조: `모듈 출력 = underlying truth + model_bias + threshold_bias`

### 4명 전원 입장 수정 → 합의

| 전문가 | 이전 입장 | 수정된 입장 |
|--------|----------|-----------|
| Crowds | "다른 질문이므로 crowds 직접 적용 불가" | "버킷 맥락에서 crowds 직접 적용 가능. 연속 확장이 novelty" |
| Medical | "multi-modal fusion이지 inter-annotator 아님" | "raw에서는 multi-modal, 운용에서는 crowds. **둘 다 유효**" |
| Distill | — | "soft label vs hard label 구분과 정확히 대응. 둘 다 사용이 최적" |
| MLEng | 동의 | "MVP가 더 단순해짐. binary classification + crowd annotation" |

**최종 합의: 두 레이어가 공존한다.**

```
Layer 1 (threshold 전, 연속 출력): 다른 질문 → multi-modal fusion
    ↓ threshold 변환
Layer 2 (threshold 후, binary vote): 같은 질문 → Learning from Crowds
```

## 7.3 Dual-Mode 아키텍처 합의

4명이 독립적으로 같은 결론에 도달:

```
Student 입력: 14개 모듈의 연속 출력 (~21D)  ← continuous feature
학습 목표:
  α × Σ MSE(teacher_head_i(z), continuous_output_i)  [soft, 정보 보존]
  + (1-α) × Σ BCE(bucket_head_k(z), binary_vote_k)   [hard, 방향성]
```

Hinton et al. (2015)의 soft+hard distillation과 대응.

## 7.4 Multi-Bucket 표현 + Few-Shot Transfer

### 4명 합의

| 항목 | 합의 |
|------|------|
| 단일 z가 다양한 버킷 표현 가능 | 4/4 — Multi-task learning 표준 구조 |
| z = 16D 시작, 버킷 10개+ 시 24-32D | 4/4 |
| Few-shot 새 버킷 추가 가능 (50-100건) | 4/4 |
| catalog_scoring과 병행 → 점진적 대체 | 4/4 |
| MVP는 단일 버킷, multi-bucket은 Step 2 | 4/4 |

### Few-shot 조건 (Crowds + MLEng 합의)

- **기존 차원의 새 조합**: 높은 성공 확률 (z에 이미 관련 정보 인코딩)
- **완전히 새로운 특성**: 불가 (z에 해당 정보 없음, teacher가 제공하지 않는 정보)
- **최소 50건, 안정적 100건+**

## 7.5 MVP 설계 합의

### Day 0 Go/No-Go

```
N_eff = 14 / (1 + 13×ρ)

N_eff ≥ 3 → 진행
2 ≤ N_eff < 3 → 합의 기준 상향
N_eff < 2 → 근본적 재고
```

입력 의존성 체인(face.detect → 하위 7개 모듈) 반영 필수.

### 실험 설계 (4명 합의)

```
Exp 0: Dawid-Skene EM vs 수동 AND (Day 0, DNN 없음)
Exp 1: MLP + BCE(binary vote) — hard only
Exp 2: MLP + MSE(continuous output) — soft only
Exp 3: MLP + hybrid (soft+hard) — ablation
Exp 4: Multi-bucket + few-shot transfer
```

### 성공 기준 (5개)

1. N_eff ≥ 3
2. Student MLP ≥ catalog_scoring 성능
3. Week 2 Student > Week 1 Student
4. Pseudo-label 정확도 > 95% (Anchor Set)
5. Teacher reliability 분산 존재

## 7.6 공통 우려 3가지

### ① Observer 독립성 위반

- 14개 모델의 공유 backbone/데이터 + 입력 의존성 체인
- "3개 동시 오류율 0.0125%" 계산은 독립성 가정 하에서만 유효
- 실질적 독립 observer 수는 5-7개 수준 추정
- **대응**: Day 0 분석, Fleiss' kappa, hierarchical clustering

### ② Identifiability 문제

- `Teacher_i = truth + bias_i`에서 GT 없이 분리 불가
- Degenerate solution 위험 (truth를 상수로 학습)
- **대응**: MVP에서 bias head 분리 ❌, 암묵적 학습, limitation으로 명시

### ③ Self-Degrading Loop

- Self-improving이 self-degrading이 될 수 있음
- Correlated error → 잘못된 pseudo-label 고착화
- **대응**: Validation gate, rollback 기준, Tier 3 blind spot registry

## 7.7 논문 포지셔닝 합의

### Contribution (4명 합의)

1. Frozen 모델 + threshold 운용 = Learning from Crowds (구조적 동치 입증)
2. Binary vote + 연속 출력 dual-mode 학습 (Snorkel soft-label 확장)
3. Multi-bucket shared representation + few-shot transfer
4. Daily data stream 기반 continual self-improvement

### Related Work (Crowds + Distill 합의)

1. Learning from Crowds (Dawid-Skene → CrowdLayer)
2. Data Programming / Snorkel — 가장 가까운 선행연구
3. Knowledge Distillation (Hinton et al.) — soft+hard 결합 이론
4. Self-Training / Continual Learning — daily stream 운용

### Novelty 포지셔닝 (4명 합의)

"잘 알려진 메커니즘을 새로운 맥락에 적용" — 정직한 포지셔닝.
메커니즘 자체가 아니라 적용 맥락의 교차점이 novelty:
이산→연속, 동종→이종, 정적→온라인, binary→dual-mode.

## 7.8 Round 4 로드맵 수정

Round 1-3의 18-item 로드맵(Tier 1-3)은 **기존 PoC 코드 개선** 대상.
Round 4의 방향 전환(DNN Student)에 따라, 로드맵의 위치가 변경:

| Round 1-3 항목 | Round 4에서의 위치 |
|---------------|-------------------|
| #1 2-channel agreement | DNN이 암묵적으로 학습 (별도 구현 불필요) |
| #2 zero-padding | Teacher 출력 저장 스키마에서 해결 |
| #3 Positive 선택 | Pseudo-label majority vote로 대체 |
| #4 Softmax 이중 적용 | Teacher 출력 전처리에서 해결 |
| #5 Pose range sigmoid | Teacher 출력 정규화에서 해결 |
| #6-7 InfoNCE + VICReg | Dual-mode loss + VICReg covariance 반영 |
| #8 Conflict/quiet/missing | Tier 1/2/3 분류로 대체 |
| #9 Pose gating | Teacher reliability에서 학습 |
| #10 MLP head | Student 아키텍처 자체가 MLP |
| #11-18 | 확장 경로(Step 2-6)에 편입 |

**새 로드맵은 `how-visualbind.md`의 MVP 경로 + 확장 경로를 따른다.**
