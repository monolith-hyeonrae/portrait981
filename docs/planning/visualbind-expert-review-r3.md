# VisualBind 전문가 리뷰 Round 3: 심층 보충 분석

> Round 3 (2026-03-13): ML, AI/DL, ISP — 3명
> 목적: 상용화 임팩트 + 탑티어 학회(NeurIPS, ICML, CVPR, ECCV) 게재를 위한 구체적 전략
> 이전 리뷰: [visualbind-expert-review.md](visualbind-expert-review.md) (Round 1-2, 7명)

---

# 교차 검증 합의 매트릭스

## 3/3 만장일치

| # | 이슈 | ML | AI/DL | ISP |
|---|------|-----|-------|-----|
| 1 | Positive 선택 기준이 치명적 오류 — agreement proximity ≠ semantic similarity | ✅ | ✅ | ✅ |
| 2 | Agreement 함수 2-channel 전환 필수 — both-low neutral 복원 | ✅ | ✅ | ✅ |
| 3 | Softmax 이중 적용이 MI 40-55% 파괴 — `normalize="none"` 필수 | ✅ | ✅ | ✅ |
| 4 | 현재 novelty만으로는 NeurIPS/ICML main track 부족 | ✅ | ✅ | ✅ |
| 5 | InfoNCE 전환이 논문화 필수 조건 (triplet loss는 2024년 이후 top-tier 미수용) | ✅ | ✅ | - |
| 6 | VICReg/uniformity regularization으로 collapse 방지 필요 | ✅ | ✅ | - |
| 7 | expression↔AU는 tautological confirmation — 진정한 독립 검증 아님 | ✅ | ✅ | ✅ |

## 2/3 강한 권장

| # | 이슈 | ML | AI/DL | ISP |
|---|------|-----|-------|-----|
| 8 | Pose range MinMax(-90,90) → sigmoid로 유효 해상도 3배 향상 | - | - | ✅✅ |
| 9 | Curriculum learning: agreement threshold의 epoch별 schedule | ✅ | ✅ | - |
| 10 | Linear → MLP projection head 전환 (conjunction 표현 불가) | ✅ | ✅ | - |
| 11 | Duchenne smile derived signal로 conjunction 간접 표현 | - | - | ✅✅ |
| 12 | FACS 6-emotion 자동 CrossCheck (4개→16개 커버리지 확장) | - | - | ✅✅ |

---

# ML 전문가 심층 분석

## 1. 학술적 포지셔닝: 솔직한 평가

### 1.1 Novelty 평가 — 기존 연구 대비

| 기법 | VisualBind와의 관계 | 차별점 |
|------|-------------------|--------|
| CLIP/ImageBind | raw modality alignment | visualbind는 frozen observer + rule-based agreement. 구조적으로 다름 |
| **Snorkel (Data Programming)** | labeling function ensemble → probabilistic label | **CrossCheck = labeling function. 핵심 구조가 동일** |
| Co-training (Blum & Mitchell 1998) | multi-view consistency | N-view 일반화이나, multi-view co-training 확장 연구가 이미 존재 |
| SimCLR/MoCo/BYOL | augmentation-driven pair | agreement-driven pair가 유일한 진정한 차별점 |
| VICReg/Barlow Twins | collapse 방지 | EntropyRegularizer는 이 계보의 단순화 |
| Prototypical Networks | centroid distance | catalog_scoring이 정확히 이것. visualbind는 metric space 전환 |

**핵심 문제**: "Frozen observer agreement as weak supervision"은 **Snorkel의 labeling function과 구조적으로 동치(isomorphic)**. CrossCheck = labeling function, AgreementEngine = label model. 차이점은 (1) ML 모듈 출력이 입력, (2) agreement를 contrastive learning의 soft label로 사용.

이 차이만으로는 NeurIPS/ICML main track은 어렵다. FG(IEEE Face and Gesture) 또는 ECCV/CVPR workshop이 현실적.

### 1.2 Top-tier 게재를 위한 3가지 경로

**경로 A: 이론적 contribution 강화**

Co-training PAC bound를 N-observer로 확장하는 정리 증명:

$$\text{err}(h) \leq \frac{1}{\binom{N}{2}} \sum_{i<j} \text{agreement\_err}_{ij}(h) + O\left(\sqrt{\frac{d}{Nm}}\right)$$

Observer 수 $N$에 따라 $\sqrt{1/N}$으로 bound 감소를 보이면 novelty 확보.

**경로 B: Empirical contribution (벤치마크 + scaling law)**

"Multi-Observer Agreement Benchmark (MOAB)" 제안:
- AffectNet (표정 + AU annotation)
- BP4D (spontaneous AU + expression)
- DISFA (AU intensity)

6가지 baseline 대비 비교:
1. Raw signal concatenation + PCA
2. Late fusion (각 observer embedding 평균)
3. Snorkel-style probabilistic labeling + supervised learning
4. SimCLR (augmentation pair)
5. VICReg (variance-invariance-covariance)
6. VisualBind (agreement-driven pair)

Observer 수 {2, 4, 7, 14}에 따른 scaling law 실증이 핵심 empirical contribution.

**경로 C: Application paper (CVPR/ECCV main)**

portrait981 실제 운영 데이터로 "label 없이 학습한 portrait embedding이 hand-crafted scoring을 능가" 입증. 정량적 개선 + human evaluation 필수.

### 1.3 논문 제목 후보

1. **"Learning from Frozen Observers: Agreement-Supervised Contrastive Representations without Labels"** — 가장 정확
2. **"Observer Agreement is All You Need: Self-Supervised Metric Learning from Multi-Model Consensus"** — 도발적
3. **"VisualBind: Binding Multi-Observer Signals into Unified Representations via Weak Agreement Supervision"** — 시스템 논문용

### 1.4 Abstract 구조

```
[Problem] 다수의 pre-trained 분석 모델이 동일 입력을 관찰할 때,
이들의 출력을 통합하여 단일 representation을 학습하는 문제.
[Gap] 기존 방법은 레이블(supervised), augmentation(self-supervised),
또는 raw modality alignment(CLIP)에 의존.
Frozen model의 출력 간 agreement를 직접 supervision으로 활용하는
프레임워크는 존재하지 않음.
[Method] Observer Agreement를 contrastive learning의 soft supervision으로
사용하는 4-stage pipeline 제안.
도메인 지식은 선언적 CrossCheck로 주입.
[Results] Synthetic + real-world (portrait quality) 실험에서
supervised baseline의 X%에 도달.
Observer 수 증가에 따른 logarithmic improvement 확인.
[Significance] Label-free, training-free observer orchestration framework.
```

## 2. Loss 설계와 학습 메커니즘

### 2.1 InfoNCE with Agreement-weighted Temperature

$$\mathcal{L} = -\sum_{i} \log \frac{\exp(\text{sim}(z_i, z_{p_i}) / \tau_i)}{\sum_{j \neq i} \exp(\text{sim}(z_i, z_j) / \tau_i)}$$

Sample-wise temperature:

$$\tau_i = \tau_{\text{base}} \cdot (1 + \alpha \cdot (1 - a_i))$$

Agreement 높을수록 temperature 낮아져 sharp gradient, 낮을수록 soft gradient.

**대안 (더 실용적)**: Agreement를 loss weight로:

$$\mathcal{L} = -\sum_{i} w_i \cdot \log \frac{\exp(\text{sim}(z_i, z_{p_i}) / \tau)}{\sum_{j} \exp(\text{sim}(z_i, z_j) / \tau)}$$

$w_i = a_i^{\gamma}$, $\gamma$는 focusing parameter (focal loss 차용). 구현 간단하고 해석 명확.

### 2.2 Curriculum Learning Schedule

```python
def agreement_threshold_schedule(epoch, max_epochs):
    progress = epoch / max_epochs
    if progress < 0.3:      # Phase 1: 확실한 샘플만
        return 0.85, 0.15
    elif progress < 0.7:    # Phase 2: 점진적 확장
        t = (progress - 0.3) / 0.4
        return 0.85 - 0.25 * t, 0.15 + 0.15 * t
    else:                   # Phase 3: ambiguous zone 포함
        return 0.55, 0.35
```

### 2.3 Hard Negative Mining — 2차원 결합

Agreement space + embedding space mining 결합:

$$n^* = \arg\max_{n \in \mathcal{N}} \left[\beta \cdot \text{sim}(z_a, z_n) + (1-\beta) \cdot a_n\right]$$

$\beta$는 epoch에 따라 0 (agreement space 우선) → 1 (embedding space 우선)로 annealing.

### 2.4 Collapse 방지: VICReg Regularization

$$\mathcal{L}_{\text{reg}} = \lambda_v \cdot v(Z) + \lambda_c \cdot c(Z)$$

- **Variance term**: $v(Z) = \frac{1}{d} \sum_{j=1}^{d} \max(0, \gamma - \text{Std}(z^j))$
- **Covariance term**: $c(Z) = \frac{1}{d} \sum_{i \neq j} [C(Z)]_{ij}^2$

InfoNCE보다 collapse에 robust하고 batch size에 덜 민감.

## 3. Agreement 함수의 ML 이론적 분석

### 3.1 Gradient Flow 비교

**$a \cdot b$ gradient**: $\partial(ab)/\partial a = b$. **$a$가 작으면(neutral) gradient도 vanish**. Neutral sample이 학습에 전혀 기여하지 않아 representation이 random에 가깝게 남음.

**$1 - |a - b|$ gradient**: $-\text{sign}(a-b)$. 상수 gradient. 어디서든 동일 크기. 하지만 activation 정보 없으므로 both-low=0.01과 both-high=0.99가 동일 agreement.

**$2ab/(a+b)$ gradient**: $2b^2/(a+b)^2$. Activation level에 선형 비례. **$a \cdot b$보다 나은 compromise이나, both-low에서 여전히 gradient 약함.**

### 3.2 2-Channel의 정보 이론적 의미

Agreement를 2D 공간 $(\text{sim}, \text{act}) \in [0,1]^2$에 매핑:

| 영역 | sim | act | 의미 | PairMiner 처리 |
|------|-----|-----|------|---------------|
| (1.0, 0.9) | 높음 | 높음 | co-activation | positive |
| (1.0, 0.1) | 높음 | 낮음 | co-absence (neutral) | quiet positive |
| (0.2, 0.5) | 낮음 | 중간 | conflict | negative |
| (0.8, 0.05) | 높음 | 극저 | noise | 제외 |

**Neutral 상태가 자체 positive pool을 형성**할 수 있어 both-low 문제의 근본적 해결.

### 3.3 Observer Independence 위반 처방

Expression↔AU는 동일 물리 현상의 서로 다른 추상 수준 → conditional independence 위반.

**Independence-adjusted weighting**:

$$w_{ij}^{\text{adj}} = w_{ij} \cdot (1 - \rho_{ij})$$

$\rho_{ij}$는 두 observer의 MI를 [0,1]로 정규화한 값. Expression↔AU의 MI가 ~0.7이면 weight 0.3배 감소. Expression↔body.pose의 MI가 ~0.1이면 weight 0.9배 유지. k-NN MI estimator (Kraskov et al., 2004)로 추정.

## 4. 상용화 임팩트

### 4.1 Frozen Model Orchestration의 Structural Advantage

| 측면 | End-to-end Fine-tuning | VisualBind (Frozen + Agreement) |
|------|----------------------|-------------------------------|
| GPU 비용 | 전체 모델 gradient | Agreement는 numpy, 학습은 linear layer만. **1000배+ 절감** |
| 모델 교체 | 재학습 필수 | Observer 교체 → CrossCheck만 수정 |
| 배포 | 단일 거대 모델 | 각 observer 독립 배포, embedding만 업데이트 |
| 디버깅 | Black box | Agreement score가 해석 가능한 중간 단계 |
| 규제 | 모델 전체 감사 | 각 observer + rule 개별 감사 |

### 4.2 도메인 확장 가능성

- **의료영상** (가장 빠른 확장): 레이블 비용 극고, frozen CAD 모델 다수, explainability 규제 요건
- **자율주행**: LiDAR + Camera + Radar observer agreement
- **산업검사**: visual + infrared + ultrasonic 센서 합의
- **스포츠 분석**: Pose estimator + ball tracker + formation analyzer 합의

---

# AI/DL 전문가 심층 분석

## 1. Information Bottleneck 관점의 4-Stage Pipeline

### 1.1 Stage 2 (Agreement)가 핵심 병목

다차원 신호 공간을 **스칼라 agreement score 하나로 붕괴**. 11차원 입력에서 1-2비트 수준의 정보만 추출.

$$H(S) \leq \log_2(|\text{bins}|)$$

연속값이지만 threshold 기반 이진 분할로 사용되므로 실효 정보량 ~1-2비트. **Agreement score가 Y(true state)에 대한 sufficient statistic이 아님** — 이것이 현 시스템의 근본적 병목.

### 1.2 Softmax 이중 적용의 정보 이론적 분석

원본 `p = (0.80, 0.10, 0.05, 0.05)` → 이중 softmax 후 `(0.33, 0.23, 0.22, 0.22)`

- Shannon entropy: 0.92 bits → 1.36 bits (최대 2.0 bits)
- **MI 감소율: 단순 50%가 아닌, KL divergence 기준으로 원본 분포 첨도(kurtosis)에 비례하여 최대 80%+ 손실 가능**

### 1.3 Linear Projection이 학습하는 것의 정확한 기술

$$f(x) = \frac{Wx}{\|Wx\|} \in S^{d-1}$$

데이터 manifold를 hypersphere 위 geodesic distance로 재해석. **한계**: 비선형 분리 불가. "AU6 AND AU12 동시 높을 때만 Duchenne smile"이라는 conjunction은 $x_{\text{AU6}} \cdot x_{\text{AU12}}$ 곱 항이 필요하며 linear projection으로 원천적 표현 불가.

### 1.4 Alignment and Uniformity (Wang & Isola, 2020) 관점

현재 triplet margin loss는 alignment에만 직접 기여, uniformity에는 간접적. **Uniformity regularization 없이 dimensional collapse 위험**. EntropyRegularizer가 아직 구현되지 않아 이 문제 악화.

## 2. 최적화와 학습 안정성

### 2.1 Gradient 근사의 수학적 분석

L2 normalization Jacobian $J = (I - \hat{e}\hat{e}^T) / \|e\|$을 건너뛰면:

1. **방향 왜곡**: 실제 gradient는 $\hat{e}$에 수직인 성분만 유효한데, $\hat{e}$ 방향 성분도 포함
2. **크기 왜곡**: $1/\|e\|$ 인자 빠짐 → norm이 큰 embedding일수록 gradient 과대추정

PoC에서는 작은 LR(0.005)로 "우연히 작동"하지만, "올바르게 수렴"이 아님.

### 2.2 Triplet Loss Gradient Landscape

3개 영역:
1. **Easy** ($d_{\text{neg}} > d_{\text{pos}} + m$): Gradient = 0. 학습 기여 없음
2. **Active** ($d_{\text{pos}} < d_{\text{neg}} < d_{\text{pos}} + m$): Semi-hard. 가장 유용
3. **Hard** ($d_{\text{neg}} < d_{\text{pos}}$): 크지만 noise 취약

학습 진행 시 대부분 easy region → **gradient starvation**. 300 epoch 학습은 이 비효율의 증거. Semi-hard mining 적용 시 50-100 epoch이면 충분.

### 2.3 Convergence 실패 반례

1. **Margin collapse**: 초기 embedding이 이미 모든 triplet에서 조건 만족 → gradient 0
2. **Oscillation**: Normalization gradient 근사로 loss landscape과 다른 방향 업데이트
3. **Label noise divergence**: 잘못된 positive 선택으로 모순적 gradient 누적

## 3. 논문화를 위한 실험 설계

### 3.1 Ablation Study

| Component | 제거/변경 | 기대 효과 |
|-----------|----------|----------|
| Agreement stage | Random pair selection | Embedding 품질 급감 → agreement 핵심 기여 입증 |
| CrossCheck rules | 순수 통계적 합의만 | Domain knowledge 주입 효과 정량화 |
| Normalization | 모든 source에 `none` | Signal scale 불일치 영향 |
| Ambiguous zone drop | 전체 데이터 사용 | "Less is more" 가설 검증 |
| Linear vs MLP | 2-layer MLP | 비선형 projection 이득 |
| Agreement function | `a*b` vs `1-|a-b|` vs 2-channel | Both-low 처리 효과 |
| Number of observers | 1, 2, 3 subsets | Observer 수 vs embedding 품질 관계 |

### 3.2 정량적 평가 메트릭

kNN accuracy 외 필수:

| 메트릭 | 의미 |
|--------|------|
| Silhouette Score | Cluster compactness vs separation |
| NMI | Cluster assignment vs ground truth |
| ARI | Chance-corrected cluster agreement |
| Retrieval@K | Top-K retrieval precision |
| Alignment & Uniformity | Wang & Isola 2020 |
| t-SNE/UMAP | 정성적 cluster separation |
| **Human evaluation** | 가장 중요 — 최종 portrait 주관적 품질 |

### 3.3 Baseline 선정

| Baseline | 공정 비교 조건 |
|----------|--------------|
| Catalog Scoring (현재) | 동일 signal vector, 같은 데이터 |
| Raw signal PCA | 같은 input/output dim |
| SimCLR-style augmentation | 같은 backbone, 데이터 |
| Ensemble average | 같은 observers |
| Random embedding | 같은 dim (lower bound) |

### 3.4 Real-world Dataset

| 데이터셋 | 용도 | 규모 |
|----------|------|------|
| 981파크 내부 데이터 | Primary evaluation | ~100 비디오, ~50만 프레임 |
| AffectNet | Emotion benchmark | 400K, V-A annotation |
| DISFA | AU intensity benchmark | 27명, 4845 frames |
| BP4D | Spontaneous expression | 41명, AU + emotion |

### 3.5 Minimum Viable Paper vs Full Contribution

**Minimum Viable Paper** (FG Workshop, ECCV Workshop):
- Phase I (#1-3, #8) 완료 + synthetic + 981파크 내부 데이터
- Catalog scoring 대비 A/B + ablation

**Full Contribution Paper** (FG main, ECCV main):
- Phase I-III 전체 + AffectNet/DISFA/BP4D
- Theoretical analysis + V-A space emergence + human evaluation(100+건)
- **1개 이상 외부 도메인 적용**

---

# ISP 전문가 심층 분석

## 1. 각 옵저버의 출력 특성과 Noise Profile

### HSEmotion (face.expression)

- **출력**: softmax 적용된 8-class 확률 벡터 (합=1.0). `logits=False`로 호출
- **전형적 분포**: neutral 지배 프레임 `{neutral: 0.65, happy: 0.15, sad: 0.08, ...}`, 명확한 미소 `{happy: 0.82, neutral: 0.12, ...}`
- **Noise**: face crop bbox jitter (2-5px) → 동일 표정에서 확률값 0.05-0.15 진동
- **Failure 1**: 측면(yaw>25도)에서 neutral 확률 인위적 상승. 학습 데이터(VGGFace2, AffectNet) 정면 편향
- **Failure 2**: 모션 블러 시 모든 감정 확률이 uniform 수렴 (각 ~0.125). 이때 agreement 무의미

### LibreFace (face.au)

- **출력**: DISFA 기반 12 AU intensity. 0-5 연속, 실제 95%가 0-3.5 분포
- **비선형 특성**: FACS intensity는 지각적 로그 스케일. level 1→2와 3→4의 물리적 차이 상이. **MinMax(0,5) 정규화는 이 비선형성 무시**
- **Noise**: AU12(lip corner) SNR 양호(±0.3), AU6(cheek raiser) 조명 방향 극도 민감(±0.8)
- **Bias**: DISFA 학습 데이터 백인 중심 → 한국인 포함 다양한 인종에서 AU6, AU9 체계적 과소추정

### 6DRepNet (head.pose)

- **출력**: yaw [-90,90], pitch [-60,60], roll [-60,60] (도)
- **정확도**: 공식 MAE ~3.5도, production 4-6도
- **MinMax(-90,90) 치명적 문제**: 테마파크 고객 yaw 대부분 ±30도 이내. 정규화 후 0.33-0.67 범위 압축. **정면(0도)과 측면(30도)의 정규화 차이 겨우 0.17. 유효 dynamic range 1/6로 축소**

### 환경 요인의 비대칭적 영향

| 조건 | HSEmotion | LibreFace AU | 6DRepNet | Agreement 영향 |
|------|-----------|-------------|----------|----------------|
| 실외→실내 전환 | WB 변화로 1-2프레임 교란 | 상대적 robust | 영향 미미 | 전환 구간 false conflict |
| 강한 역광 | neutral 편향 증가 | AU 검출률 20-40%↓ | MAE +3-5도 | Coverage 감소 |
| 모션 블러 | uniform 수렴 | intensity→0 수렴 | MAE 급증 | **모든 옵저버 동시 실패** → false positive 위험 |
| 선글라스 | AU6 관측 불가, happy -15~25% | AU6 완전 누락 | 정상 | Duchenne 판별 불가 |
| 마스크 | 하반부 가림, happy/surprise 구분 불가 | AU12,25,26 관측 불가 | 정상 | expression-AU CrossCheck 대부분 무효 |

### 테마파크 특수성

- 롤러코스터 하차 직후: 땀 specular reflection → face detection confidence↓, AU intensity 교란
- 야간 LED 조명 (3000K-6500K): auto WB hunting → 피부색 톤 프레임간 변동
- 어린이: 얼굴 비율 성인과 상이 → HSEmotion accuracy 10-15%↓

## 2. 정규화 심층 분석

### 2.1 Softmax 이중 적용 정량적 분석

원본: `p = [0.80, 0.10, 0.05, 0.05]` → 이중 softmax: `[0.395, 0.196, 0.186, 0.186]`

- Shannon entropy: 0.92 → 1.36 bits (48% 증가 = 판별 정보 절반 손실)
- em_happy=0.80 "확실한 미소" vs 0.60 "애매한 미소": 이중 softmax 후 0.395 vs 0.345 (차이 0.20→0.05)
- Agreement: 원래 `0.80 * 0.80 = 0.64` → `0.395 * 0.395 = 0.156` (4배 축소)

### 2.2 MinMax Range의 Cascade 효과

| 신호 | 설정 range | 실제 99% 분포 | 유효 해상도 |
|------|-----------|-------------|-----------|
| AU (0-5) | (0, 5) | 0-3.5 | 70% |
| head_yaw | (-90, 90) | ±30도 | **33%** |
| head_pitch | (-90, 90) | ±15도 | **17%** |
| head_roll | (-90, 90) | ±10도 | **11%** |

Pose 신호 판별력 축소 → expression-AU agreement가 전체 score 독점. Demo CrossCheck weight: expression-AU 관련 4.5 vs pose 관련 0. **Pose 정보의 embedding 기여가 사실상 0.**

### 2.3 최적 정규화 전략

| 옵저버 | 현재 | 문제 | 제안 | 근거 |
|--------|------|------|------|------|
| face.expression | softmax | 이중 적용, MI 50%↓ | **`none`** | 이미 확률 |
| face.au | minmax(0,5) | 비선형 무시 | **`sigmoid(center=2.0, scale=0.8)`** | FACS 지각 비선형성 |
| head.pose yaw | minmax(-90,90) | 유효 range 33% | **`sigmoid(center=0, scale=0.05)`** | ±30도에서 0.18-0.82 |
| head.pose pitch | minmax(-90,90) | 유효 range 17% | **`sigmoid(center=0, scale=0.1)`** | ±15도 focus |
| face.quality blur | minmax(0,500) | 데이터 종속 | **adaptive percentile** | face.baseline 활용 |

### 2.4 Agreement Response Surface 등고선

**현재 `a * b`**:
```
b=1.0 │ 0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
b=0.5 │ 0.0  0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
b=0.1 │ 0.0  0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10
       a=0.0 ─────────────────────────────────────────── a=1.0
```

등고선이 쌍곡선. agreement=0.25 넘으려면 **둘 다 0.5 이상** 필요. Neutral (0.1, 0.1) = 0.01.

**제안 2-channel**:
- Channel 1 (similarity = 1 - |a-b|): (0.1, 0.1) = 1.0 → **neutral 일치 인정**
- Channel 2 (activation = (a+b)/2): (0.1, 0.1) = 0.1 → **activation level 분리**

## 3. FACS 도메인 지식

### 3.1 현재 CrossCheck의 FACS 커버리지

| 감정 | FACS AU 조합 | 현재 커버 | 커버리지 |
|------|-------------|----------|---------|
| Happy | AU6 + AU12 | AU6, AU12 | **부분** (conjunction 미표현) |
| Surprise | AU1+2+5+25+26 | AU25, AU26 | **부분** (3/5 누락) |
| Anger | AU4+5+7+23+24 | 없음 | **0%** |
| Disgust | AU9+15+25 | 없음 | **0%** |
| Fear | AU1+2+4+5+20+26 | 없음 | **0%** |
| Sadness | AU1+4+15 | 없음 | **0%** |

### 3.2 Duchenne Smile 구현 방안

현재 pairwise CrossCheck로 conjunction 표현 불가. **Derived signal 방안**:

```python
def _compute_duchenne(au_hint: HintVector) -> float:
    au6 = au_hint.get("AU6")
    au12 = au_hint.get("AU12")
    return min(au6, au12) / 5.0  # 두 AU 최소값 = Duchenne 강도
```

HintFrame에 추가 후 `CrossCheck("face.expression", "em_happy", "face.au_derived", "duchenne", relation="positive")`로 conjunction 간접 표현.

### 3.3 Antagonistic AU Pairs (이상치 탐지)

| AU 쌍 | 설명 | 동시 불가 이유 |
|-------|------|--------------|
| AU1 ↔ AU4 | 이마 근육 길항 | corrugator vs frontalis |
| AU12 ↔ AU15 | 입꼬리 길항 | zygomaticus vs depressor |
| AU25 ↔ AU24 | 입술 길항 | 벌림 vs 다묾 |

동시 intensity > 2.0이면 AU detector 오류. AgreementEngine에 antagonistic pair 검사 추가, 위반 시 전체 agreement에 penalty 또는 pair mining 제외.

### 3.4 "좋은 초상화"의 정량적 정의

1. **표정**: Duchenne smile (AU6≥2.0 AND AU12≥2.5) 또는 자연스러운 미소 (AU12≥1.5, em_happy≥0.5)
2. **포즈**: |yaw|<25도, |pitch|<15도, |roll|<10도
3. **품질**: face_blur<50, exposure∈[80,200], face_area_ratio>0.05
4. **비수선**: AU4<1.5, AU9<1.0 (찡그림/불쾌 배제)
5. **개방**: 눈 열림 (1-em_neutral>0.3), 과도한 jaw drop 부재 (AU26<3.5)

### 3.5 Observer Redundancy 재평가

Expression↔AU는 **부분적 tautology**. 같은 입력(face crop), 같은 물리 현상(안면 근육), 다른 추상 수준 (AU=개별 근육, expression=의미론적 해석).

**진정한 독립 검증**:
- body.pose "양팔 올림" + em_happy = 다른 신체 부위 관측 → 독립
- CLIP aesthetic + AU6+AU12 = 시각적 품질 + 근육 활동 → 독립

**가중치 재배분**: expression↔AU weight 낮추고, cross-modal agreement(expression↔body.pose, AU↔CLIP, pose↔quality)에 높은 weight 부여.

## 4. 상용화 ISP 관점

### 4.1 Adaptive Normalization

고정 range 대신 face.baseline Welford online stats 활용:

```python
class AdaptiveMinMax:
    def __init__(self, warmup_frames=100, percentile_lo=5, percentile_hi=95):
        self.history = RingBuffer(warmup_frames)

    def normalize(self, value, field):
        self.history.push(value)
        if len(self.history) < self.warmup_frames:
            return static_minmax(value, field)
        lo = np.percentile(self.history, self.percentile_lo)
        hi = np.percentile(self.history, self.percentile_hi)
        return clip((value - lo) / (hi - lo + 1e-8), 0, 1)
```

### 4.2 Calibration Drift 감지

- **Agreement 분포 모니터링**: 일별 mean/std 추적. Mean 하락 >0.1은 카메라 교체/조명 변경/모델 degradation 증거
- **Per-CrossCheck drift**: 특정 CrossCheck만 하락 → 해당 옵저버 문제
- **Self-check**: 비디오 시작 10초 vs 끝 10초 agreement 분포 KS-test. 유의미 차이 → temporal drift 경고

---

# 통합 로드맵 v3 (Round 1-3 교차 검증)

## Tier 1: 즉시 수정 (논문 + 상용 모두 blocking)

| # | 작업 | 난이도 | 논문 영향 | 상용 영향 | 합의 |
|---|------|--------|----------|----------|------|
| 1 | **Positive 선택 → signal cosine similarity** | 낮음 | 매우 높음 | 매우 높음 | R1-3 10/10 |
| 2 | **Agreement 2-channel (similarity + activation)** | 낮음 | 매우 높음 (핵심 contribution) | 매우 높음 | R1-3 10/10 |
| 3 | **flat_vector zero-padding** | 낮음 | 낮음 (bug fix) | 매우 높음 (crash 방지) | R1-3 10/10 |
| 4 | **Softmax 이중 적용 제거** (normalize="none") | 매우 낮음 | 중간 | 높음 (MI 40-55% 회복) | R2-3 6/6 |
| 5 | **Pose range 정규화** MinMax(-90,90) → sigmoid | 낮음 | 낮음 | 높음 (유효 해상도 3배) | R3 ISP |

## Tier 2: 핵심 개선 (논문 contribution 확보)

| # | 작업 | 난이도 | 논문 영향 | 상용 영향 | 합의 |
|---|------|--------|----------|----------|------|
| 6 | **Triplet → InfoNCE** + agreement weight | 중간 | 매우 높음 (필수) | 중간 | R1-3 ML+AI |
| 7 | **VICReg regularization** (collapse 방지) | 중간 | 높음 (reviewer 필수 지적) | 중간 | R3 ML+AI |
| 8 | **Conflict/quiet/missing 3-way 분류** | 중간 | 높음 (DST 연결) | 높음 | R1-2 7/10 |
| 9 | **Pose-dependent confidence gating** | 중간 | 중간 | 높음 | R1-3 모두 |
| 10 | **Linear → MLP projection head** | 중간 | 높음 (conjunction 표현) | 중간 | R3 ML+AI |

## Tier 3: 확장 (full contribution paper)

| # | 작업 | 난이도 | 논문 영향 | 상용 영향 | 합의 |
|---|------|--------|----------|----------|------|
| 11 | **Curriculum learning** (threshold schedule) | 중간 | 중간 | 중간 | R3 ML+AI |
| 12 | **FACS 6-emotion 자동 CrossCheck** (4→16개) | 낮음 | 중간 | 중간 | R2-3 |
| 13 | **Duchenne smile derived signal** | 낮음 | 중간 | 높음 | R2-3 Neuro+ISP |
| 14 | **Temporal smoothing** (EMA/apex window) | 중간 | 중간 | 중간 | R1-2 |
| 15 | **Observer independence-adjusted weighting** | 중간 | 높음 (이론) | 중간 | R3 ML+AI |
| 16 | **Hard negative mining** (agreement+embedding space) | 중간 | 중간 | 중간 | R3 ML |
| 17 | **N-observer scaling law** 실험 | 높음 | 매우 높음 (경로 B) | 낮음 | R3 ML |
| 18 | **Adaptive normalization** (Welford 활용) | 중간 | 낮음 | 높음 | R3 ISP |

## 실행 순서 최적화

```
Phase I (1-2주): Tier 1 전부 (#1-5)
  → 이것 없이는 이후 모든 개선의 효과 측정이 오염됨

Phase II (1-2주): #6 (InfoNCE) + #7 (VICReg) + #10 (MLP)
  → 학습 인프라 안정화. Phase III 실험의 통계적 검정력 확보

Phase III (2-3주): #8, #9, #11-16
  → Domain knowledge 개선. Phase I-II 위에서 효과 발휘

Phase IV (2-4주): #17, #18 + 실험/논문 작성
  → Scaling law, real-world dataset, ablation study
```

## 논문 게재 전략

| 목표 | 필요 완성도 | 예상 시기 |
|------|-----------|----------|
| FG Workshop / ECCV Workshop | Phase I + synthetic + 내부 데이터 | 4-6주 |
| FG Main | Phase I-II + BP4D/DISFA + ablation | 2-3개월 |
| ECCV/CVPR Main | Phase I-III + AffectNet + 이론 분석 + human eval | 4-6개월 |
| NeurIPS/ICML | Phase I-IV + N-observer bound 증명 + scaling law | 6-9개월 |
