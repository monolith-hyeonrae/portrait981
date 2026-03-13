# VisualBind 전문가 리뷰 보고서

> 3라운드, 10명의 전문가가 독립적으로 visualbind PoC를 분석하고 교차 검증한 결과.
> Round 1 (2026-03-12): ML, Sensor Fusion, Systems — 3명
> Round 2 (2026-03-12): ML/AI, Signal Processing, Neuroscience, Architecture — 4명
> Round 3 (2026-03-13): ML, AI/DL, ISP — 3명 → [심층 보충 분석](visualbind-expert-review-r3.md)

---

# Round 1: ML/Contrastive Learning 전문가

## 프레임워크의 이론적 위치

VisualBind의 접근법은 multi-view self-supervised learning의 변형으로 볼 수 있다. CLIP이 (image, text) 쌍의 co-occurrence를, ImageBind가 image를 anchor로 6개 modality를 binding하는 것처럼, VisualBind는 동일 프레임에서 독립적 observer들의 **cross-modal agreement**를 supervision signal로 사용한다.

이론적 근거는 건전하다: 독립적 모델들이 동일 현상에 대해 일관된 신호를 보내면, 그 신호는 개별 모델의 noise보다 신뢰도가 높다는 것은 ensemble theory와 multi-view information bottleneck 관점에서 타당하다.

### 기존 연구와의 비교

Multi-view learning (Xu et al., 2013)과 가장 직접적으로 연결된다. 그러나 결정적 차이가 있다:

- **CLIP/ImageBind**: 서로 다른 modality의 raw representation을 직접 contrastive learning으로 align. Supervision이 co-occurrence 자체.
- **VisualBind**: Raw signal을 직접 align하지 않고, **도메인 지식 기반 rule(CrossCheck)로 agreement를 먼저 계산**한 뒤 그 score를 supervision으로 사용. 본질적으로 **rule-based pseudo-labeling + contrastive learning**의 2-stage 구조.

### 2-stage 분리의 장단점

**장점**: Agreement score가 해석 가능하다. "AU12와 em_happy가 동시에 높으면 진짜 미소"라는 도메인 지식을 명시적으로 인코딩할 수 있다. CLIP처럼 수백만 pair가 필요하지 않으며, 소규모 데이터에서도 meaningful한 signal을 얻을 수 있다.

**단점**: Agreement score의 품질이 수작업 CrossCheck rule의 completeness에 완전히 의존한다. 이것은 self-supervised learning의 핵심 가치인 "사람이 모르는 패턴도 학습한다"는 이점을 크게 훼손한다. 본질적으로 **domain expert가 이미 알고 있는 상관관계만** supervision으로 사용하므로, 학습된 embedding이 expert knowledge를 넘어서기 어렵다.

## 학습 메커니즘의 약점

### Linear Projection의 한계

`TripletEncoder`는 단일 linear layer (`W @ x`)에 L2 normalization만 적용한다.

**불충분한 이유**: 입력 signal들이 이미 normalize된 scalar 값들의 concatenation이므로, 이들 간의 **비선형 상호작용**을 포착할 수 없다. 예를 들어, "AU6가 높고 AU12가 높을 때만 진짜 미소"라는 conjunction은 linear projection으로는 원천적으로 학습 불가능하다. SimCLR 논문(Chen et al., 2020)이 밝힌 핵심 발견 중 하나가 바로 nonlinear projection head가 representation quality를 크게 개선한다는 것이다.

**PoC에서는 수용 가능한 이유**: 입력이 11차원의 저차원 normalized signal이라는 점에서, feature 간 interaction의 복잡도가 제한적이다. 4차원 embedding으로의 linear projection은 사실상 PCA + rotation에 가까운 효과를 내며, 합성 데이터처럼 cluster가 linearly separable한 경우에는 동작한다. 문제는 real-world data에서 cluster boundary가 비선형일 때 시작된다.

### Triplet Loss의 한계

현재 구현은 classic triplet margin loss: `L = max(0, d(a,p) - d(a,n) + margin)`.

**Pair-level vs batch-level**: Triplet loss는 한 번에 하나의 (anchor, pos, neg) 관계만 본다. InfoNCE는 batch 내 모든 sample 간의 관계를 동시에 고려하므로 gradient가 훨씬 informative하다. N개 sample이 있을 때 triplet loss는 O(1)개의 negative 관계를 보지만, InfoNCE는 O(N)개를 본다. 현재 `max_pairs_per_anchor=3`으로 제한되어 있어 negative landscape의 극히 일부만 탐색한다.

**Gradient saturation**: `d(a,n) > d(a,p) + margin`인 easy triplet은 gradient가 0이다. 학습 초기에 대부분의 triplet이 이 조건을 만족하면 학습이 멈춘다.

**Margin 설정과 collapse**: `margin=0.3`은 arbitrary한 선택. L2 normalization 후의 embedding space에서 margin=0.3은 유효 범위 [0, 2] 대비 상당히 작아서 collapse risk는 낮지만, 충분한 separation을 강제하지도 못한다.

### Agreement-based Pair Mining의 Failure Modes

이것이 VisualBind의 가장 심각한 구조적 약점이다.

**False Positive (high agreement, wrong state)**: `_positive_agreement`는 `a * b`로 계산된다. 문서의 주석에는 "Both high → ~1.0, both low → ~1.0"이라고 적혀 있지만, 실제 구현은 `a * b`이므로 **both low일 때 agreement가 0에 가깝다**. 이것은 주석과 구현의 불일치이며, "neutral 상태에서 expression도 low, AU도 low"인 경우를 correct agreement로 인식하지 못한다. Neutral face는 모든 CrossCheck에서 agreement가 낮게 나올 수밖에 없고, 따라서 negative pool로 빠진다. 이것은 **시스템적 bias**다.

**Positive 선택 전략의 문제**: `PairMiner.mine()`에서 positive는 "agreement score가 anchor와 가장 가까운 high-agreement 프레임"으로 선택된다. 이것은 **semantic similarity가 아니라 agreement score의 수치적 근접성**으로 positive를 고르는 것이다. Agreement score 0.72인 happy_frontal과 agreement score 0.71인 surprise_open이 pair가 될 수 있다. Agreement score는 "observer들이 얼마나 동의하는가"이지 "두 프레임이 같은 상태인가"가 아니다. 이것은 근본적인 설계 오류다.

### Gradient 구현의 근사

`_step` 메서드에 `# Approximate: skip normalization gradient (works for PoC)` 주석이 있다. L2 normalization의 Jacobian을 무시하면 gradient 방향이 왜곡된다. Normalization gradient는 `(I - e*e^T)/||e||` 형태이며, 이를 건너뛰면 embedding이 norm이 큰 방향으로 치우치는 경향이 생긴다. PoC 한정으로 수용 가능하나, production은 autograd 필수.

## 개선 제안

### Hard Negative Mining

현재 negative sampling은 uniform random이다. 세 가지 전략을 순차적으로 도입할 수 있다:

1. **Semi-hard mining**: `d(a,p) < d(a,n) < d(a,p) + margin`인 negative만 선택. `PairMiner`에 encoder reference를 전달하는 구조로 확장 가능.
2. **Agreement-aware hard negatives**: Agreement score가 positive_threshold 바로 아래인 프레임(현재 ambiguous로 버려지는)을 hard negative로 활용. "거의 맞지만 아닌" 상태를 대표하므로 학습에 가장 유용.
3. **Batch-내 hardest**: InfoNCE 전환 시 자연스럽게 해결.

### InfoNCE 전환

Triplet loss를 `L = -log( exp(sim(a,p)/τ) / Σ_k exp(sim(a,n_k)/τ) )`로 대체. Temperature τ는 agreement score 분포의 표준편차 역수에서 초기값 유도 가능. CLIP처럼 learnable temperature를 사용하면 자동 조정.

### Curriculum Learning

Agreement score 자체를 difficulty measure로 사용:

- **Phase 1**: Agreement > 0.8 (매우 명확한 상태)만으로 학습
- **Phase 2**: Threshold를 점진적으로 낮추어 0.5까지 확장
- **Phase 3**: Ambiguous zone (0.2~0.5)의 프레임도 soft label로 포함

`PairMiner`의 `positive_threshold`와 `negative_threshold`를 epoch에 따라 schedule하는 것만으로 구현 가능.

### Agreement Score 자체의 학습 (Meta-learning)

CrossCheck weight를 학습 가능하게:

- Encoder의 loss를 outer objective로, CrossCheck weight를 inner variable로 하는 bi-level optimization
- 더 간단하게는, CrossCheck별 agreement와 최종 embedding quality 간의 상관관계를 측정하여, 기여도가 낮은 check의 weight를 자동 감소시키는 pruning 방식
- **주의**: meta-learning의 classic 문제인 순환 구조(agreement → pair mining → encoder → agreement weight) — 수렴 보장되지 않음

### Online / Streaming 학습

- **Sliding window agreement**: 최근 K 프레임(K=300, 10초 @ 30fps)의 agreement history로 pair mining
- **EMA weight update**: MoCo의 momentum encoder와 유사한 구조
- **Memory bank**: 과거 high-agreement 프레임의 embedding을 저장. momentbank 인프라와 자연스럽게 연결

## 스케일링 과제

- **Observer 가용성**: coverage가 낮을 때 score가 소수 check에 의해 불안정. `coverage` 필드가 있지만 score에 미반영
- **Embedding drift**: 비디오 세션이 길어지면 초기 학습된 embedding space가 후반부에 안 맞을 수 있음. Continual learning 문제
- **CrossCheck 확장 병목**: 새 observer 추가 시 전문가가 새 rule을 작성해야 함. 자동 CrossCheck 생성이 필수

**핵심 제언**: 가장 시급한 두 가지는 (1) `_positive_agreement`의 "both low" case 처리와 (2) positive 선택 기준을 signal vector similarity로 변경하는 것.

---

# Round 1: Sensor Fusion 전문가

## 핵심 평가

현재 VisualBind는 fusion이 아니라 **agreement scoring** 시스템이다. 이론적 기반이 ad-hoc.

### Dempster-Shafer Theory

다중 옵저버 결합의 정립된 이론 프레임워크. 핵심 개념:

- **Belief function**: 각 옵저버를 mass assignment로 표현
- **Conflict measure K**: 두 옵저버가 얼마나 모순되는지 정량화
- **Plausibility vs Belief**: 불확실성(uncertainty)과 무지(ignorance)를 구분 — "모르겠다"를 표현 가능

현재 VisualBind에 없는 것: conflict의 명시적 정량화, 불확실성과 무지의 구분.

### CrossCheck 표현력의 한계

positive/negative relation만으로는 표현 불가능한 관계들:

- **비선형 관계**: "yaw < 15도이면 AU 신뢰, > 30도이면 무시" (조건부)
- **삼자 관계**: "AU6 AND AU12 AND em_happy" (conjunction)
- **비대칭 관계**: "em_happy가 높으면 AU12가 높아야 하지만, AU12가 높다고 반드시 em_happy일 필요는 없다"

### Softmax 재적용 문제

원시값이 이미 확률 분포(HSEmotion 출력)이면, softmax를 다시 적용하면 분포가 왜곡된다. 정보 이론적으로 MI(Mutual Information) 감소.

### FACS 매핑 기반 자동 CrossCheck

현재 happy/surprise만 커버. FACS 매핑 테이블에서 자동 파생 가능:

| 감정 | AU 조합 |
|------|---------|
| Happy | AU6 + AU12 |
| Surprise | AU1 + AU2 + AU5 + AU25 + AU26 |
| Anger | AU4 + AU5 + AU7 + AU23 + AU24 |
| Disgust | AU9 + AU15 + AU25 |
| Fear | AU1 + AU2 + AU4 + AU5 + AU20 + AU26 |
| Sadness | AU1 + AU4 + AU15 |

### Attention-based Fusion

장기적으로, CrossCheck 없이 옵저버 간 관계를 데이터에서 학습하는 attention mechanism. 수동 정의의 한계를 넘어설 수 있는 경로.

---

# Round 1: Real-time Systems 전문가

## 성능 평가

- **Inference latency**: visualbind의 `encode()` = 행렬 곱셈 1회 + L2 정규화. 마이크로초 단위. 전체 파이프라인(face detect ~10ms, expression ~5ms) 대비 무시 가능
- **메모리**: `_W` 행렬 168 float64 + HintFrame 캐시. KB 단위. GPU 메모리와 무관
- **배치 학습**: 일일 축적 데이터(~10,000 프레임)에 대해 200 epoch, numpy 기반. 수십 초

## extract.py 대체 가능성

extract.py(358줄)의 스코어링 관련 함수가 대체 대상. 단, gate/classify 등 비-수치적 로직은 visualbind의 대상이 아님.

## Production 필수 사항

- **`_W` 행렬 save/load**: 현재 인터페이스 부재. `.npz` 형식 수 KB
- **새 analyzer 추가 시**: input_dim 변경 → 반드시 재학습 필요 (구조적 약점)
- **단계적 도입**: Shadow(로깅만) → Hybrid(alpha 혼합) → Full replacement
- **재학습 주기**: 일일 야간 배치 권장

---

# Round 2: ML/AI 전문가

## 학술적 포지셔닝

### Co-Training Theory (Blum & Mitchell, 1998)

VisualBind는 co-training과 이론적으로 연결된다. 각 analyzer는 독립된 view이며, agreement는 multi-view consistency를 측정한다. 차이점: co-training은 2-view를 가정하지만, VisualBind는 N-observer를 일반화한다.

### Prototypical Networks (Snell et al., 2017)

기존 momentscan의 catalog_scoring은 본질적으로 **hand-crafted prototypical network**이다 — category centroid에 대한 weighted distance 계산. VisualBind는 이 prototype matching을 **agreement-supervised metric space**에서 data-driven으로 전환한다.

### 논문 가능성

"**Few-shot prototype matching in agreement-supervised metric spaces**"
— 대상 학회: FG (IEEE Face and Gesture), ECCV Workshop

핵심 contribution:
1. Multi-observer agreement를 contrastive learning의 supervision으로 활용하는 프레임워크
2. 선언적 CrossCheck에 의한 도메인 지식 주입 메커니즘
3. Catalog(hand-crafted prototype) → data-driven embedding으로의 점진적 전환 방법론

### Hybrid Catalog Approach

Fisher importance → weight prior, category centroids → prototype initialization. Catalog에서 VisualBind로의 이론적 가교.

---

# Round 2: 영상 신호 처리 전문가

## 1. 정규화 전략의 정보 이론적 분석

### Softmax (face.expression에 사용)

정보 이론에서 softmax는 **최대 엔트로피 분류기의 출력 변환**이다. 입력 벡터를 확률 심플렉스에 사영하는 과정에서 다음이 발생한다:

- **정보 보존**: 원본 logit의 순서(ordering)와 상대 크기(ratio)는 보존됨
- **정보 손실**: 절대 크기(magnitude)가 사라짐. `[2.0, 1.0, 0.5]`와 `[4.0, 2.0, 1.0]`은 같은 확률을 생성
- **핵심 문제**: HSEmotion은 이미 softmax를 적용하여 확률을 출력. HintCollector에서 softmax를 재적용하면 **이중 softmax** → 분포가 더 균일(flat)해진다. MI(Mutual Information) 50% 이상 감소 추정

### MinMax (face.au, head.pose에 사용)

정보 이론적으로 가장 "정직한" 변환:

- 원본 분포의 형태를 보존, 단지 support를 [0,1]로 이동
- **문제**: range 파라미터가 사전 고정(AU: 0-5, pose: -90~90)이므로 실제 데이터 분포와 불일치 시 유효 범위가 극히 좁아진다
- 예: head_yaw의 실제 범위가 -15~15도인 경우 정규화된 값은 0.42~0.58 사이에 몰린다. 이 **dynamic range 압축**은 agreement 계산에서 해당 신호의 판별력을 크게 약화

### 구조적 편향

**편향 1: Softmax의 상호 배타성 가정** — `em_happy`와 `em_neutral`이 softmax를 통과하면, 한쪽이 높으면 다른 쪽이 반드시 낮다. Neutral 상태에서 em_happy이 softmax에 의해 인위적으로 0에 가까워지므로 agreement가 체계적으로 낮아진다.

**편향 2: MinMax의 range 불일치에 의한 비대칭** — AU는 대부분 0~3 범위에서 활동하므로 정규화 후 0~0.6 범위에 분포하는 반면, yaw는 대부분 -20~20도이므로 0.39~0.61 범위에 몰린다. agreement = a * b에서 AU 쪽이 값 범위가 넓으므로 pose보다 agreement를 더 크게 좌우한다.

### 최적 정규화 제안

1. **Expression**: 이미 확률이면 softmax 제거. `normalize="none"` 또는 `normalize="minmax"` with range=(0,1)
2. **AU**: 현재 minmax(0,5)는 합리적이나, **데이터 적응형 range** 권장. 첫 N프레임에서 percentile(5%, 95%)로 range 설정. face.baseline의 Welford online stats 인프라 재활용 가능
3. **Pose**: 관심 범위가 좁으므로(±30도) sigmoid가 적합. `center=0, scale=0.1`로 ±30도 범위에서 0.05~0.95 해상도

## 2. Agreement 함수의 신호 처리적 분석

### `a * b` (곱, 현재 구현)

**Co-activation detector**로서의 특성:

- 수학적으로 AND gate의 연속(soft) 버전
- Response surface: z = a*b는 [0,1]^2 → [0,1] 위의 쌍곡선 포물면
- 주파수 도메인에서는 두 신호 스펙트럼의 합성곱(convolution)
- **편향**: both-low에서 실패. (0.1, 0.1) → 0.01

### `1 - |a - b|` (L1 유사도)

- 순수한 거리 기반 일치도. 두 값이 가까울수록 1에 가까움
- (0.1, 0.1) → 1.0, (0.9, 0.9) → 1.0, (0.5, 0.5) → 1.0
- **문제**: trivial agreement. 두 신호가 모두 0.01이면 agreement=0.99인데, 이것이 정말 "강한 일치"인가? **Absence of evidence를 evidence of absence로 혼동**
- SNR 관점: 낮은 값 영역에서는 측정 노이즈 비율이 높으므로 일치의 신뢰도가 낮다

### `2ab/(a+b)` (조화 평균)

- F1-score가 precision과 recall의 조화 평균인 것처럼, 두 "기여도"의 균형 평균
- (0.1, 0.1) → 0.1, (0.9, 0.9) → 0.9, (0.9, 0.1) → 0.18
- both-low에서 곱(0.01)보다 나으면서 L1 유사도(1.0)처럼 과도하지 않음

### Response Curve 비교 (핵심)

| 시나리오 | (a, b) | a*b | 1-\|a-b\| | 2ab/(a+b) |
|---------|--------|-----|-----------|-----------|
| Both high (happy+AU12) | (0.9, 0.8) | 0.72 | 0.90 | 0.85 |
| Both low (neutral) | (0.1, 0.1) | **0.01** | **1.00** | 0.10 |
| Mixed (conflict) | (0.9, 0.1) | 0.09 | 0.20 | 0.18 |
| Both medium | (0.5, 0.5) | 0.25 | **1.00** | 0.50 |
| Slight mismatch | (0.7, 0.4) | 0.28 | 0.70 | 0.51 |

### 결론: 단일 함수로는 불충분

이 도메인의 agreement는 두 가지 질적으로 다른 상황을 포괄해야 한다:

1. **Co-activation agreement**: happy(high) + AU12(high) → 둘 다 활성이므로 일치. `a*b`가 적합
2. **Co-absence agreement**: neutral(low) + AU12(low) → 둘 다 비활성이므로 일치. `(1-a)*(1-b)` 또는 `1-|a-b|`가 적합

**제안: `1 - |a - b|`를 기본으로 하되, activation gate를 분리.** Agreement 함수는 "두 값이 얼마나 비슷한가"만 측정하고, "그 일치가 의미 있는가"는 별도의 **activation level** `(a+b)/2`을 PairMiner에 전달하여 triplet 구성 시 활용한다. **관심사의 분리(separation of concerns)** 원칙에 부합한다.

## 3. Temporal Coherence와 시계열 관점

### 프레임 독립 처리의 정보 손실

비디오 프레임은 30fps 기준 33ms 간격으로, 인접 프레임 간 상관계수(autocorrelation)가 매우 높다 (일반적으로 ρ > 0.95).

- **유효 샘플 수 과대 추정**: 실질적으로 독립인 샘플 수는 N/(1+2Σρ_k) 정도로, N보다 10~50배 작을 수 있다
- **시간적 구조 자체가 의미론적 정보**: "smile이 0.3에서 0.9로 2초간 상승"하는 것과 "이미 0.9에서 유지 중"인 것은 완전히 다른 이벤트이나 프레임 독립 처리에서는 구분 불가

### 옵저버 간 Phase 차이

- **Facial Action Unit**: 근육 움직임의 직접적 관측. onset이 빠르다
- **Expression classifier**: AU 조합의 패턴 매칭으로, 개별 AU가 충분히 활성화된 후에야 분류 확신이 높아진다. AU보다 수 프레임 **지연(lag)**
- **Head pose**: 물리적 움직임으로 관성이 있어 expression보다 느리게 변화

이 phase 차이 하에서 동시 프레임의 agreement를 계산하면, onset 시점에서 "AU는 이미 활성인데 expression은 아직 비활성"인 시간 창이 존재하여 **false conflict**가 발생한다.

### Temporal Smoothing 전략 비교

| 전략 | 장점 | 단점 | 적합성 |
|------|------|------|--------|
| **EMA** | 상태 불필요, O(1), momentscan에서 이미 사용 | α 선택이 도메인 의존 | Agreement score의 short-term noise 제거에 적합. α=0.3이면 약 3프레임 lookback |
| **Kalman Filter** | process/measurement noise 분리, 최적 추정 | 선형 가정이 비선형 agreement dynamics에 부적합 | 개별 신호의 noise filtering에 적합 (agreement 전 단계) |
| **Sliding Window** | 비모수적, 분포 가정 불필요 | 윈도우 크기 선택, 경계 효과 | "이 구간 전체의 agreement가 높았는가" 판단에 적합 |

### 표정 동역학 모델링

- **Onset**: 0.3~1.0초. AU가 순차적으로 활성화. false conflict 발생 구간
- **Apex**: 0.5~4.0초. 표정 최대 강도. Agreement가 가장 신뢰할 수 있는 구간
- **Offset**: 0.5~2.0초. 비대칭적 이완 — 어떤 AU는 빨리, 어떤 것은 천천히

**제안**: Temporal consistency bonus. 현재 프레임의 agreement가 이전 K프레임과 일관되면 보너스, 갑자기 변하면 페널티. EMA smoothing보다 더 정교하게 "지속적 일치"와 "순간적 일치"를 구분.

## 4. 다중 옵저버 결합 이론

### 옵저버 독립성 분석

| 쌍 | 입력 공유 | 독립성 |
|----|----------|--------|
| expression ↔ AU | 같은 얼굴 crop | **부분 독립**: 같은 물리 현상(근육 움직임)을 다른 추상 수준에서 관측. 조건부 독립 위반 |
| expression ↔ pose | 같은 얼굴 crop | **약한 의존**: yaw가 크면 expression 신뢰도 하락이라는 간접 의존 |
| AU ↔ pose | 같은 얼굴 crop | **약한 의존**: pose 변화 시 AU 관측 난이도 변화 |

expression과 AU는 동일한 얼굴 근육의 **서로 다른 레벨의 추상화**이다: AU는 개별 근육의 활성도이고, expression은 그 조합의 의미론적 해석이다. 따라서 이들의 agreement가 높은 것은 "독립적 검증"이 아니라 **"동어반복(tautology)에 가까운 확인"**이다.

**시사점**: expression↔AU agreement에 weight 2.0을 주는 것은, 이것이 가장 "쉬운" agreement이기 때문에 과대 평가될 위험이 있다. 진정으로 독립적인 검증(body.pose와 expression, CLIP aesthetic과 AU)에 더 높은 weight를 줘야 할 수 있다.

### Conflict Detection 정량화 제안

```
conflict(a, b, relation) = {
    positive: max(a, b) * (1 - min(a,b)/max(a,b))  when max > τ
    negative: a * b  when both > τ
}
```

- 한쪽이 강하게 활성인데 다른 쪽이 비활성 → conflict
- 둘 다 비활성 → conflict가 아님 (quiet)
- Conflict가 높은 프레임은 **hard negative**로서 특히 가치

## 5. Fisher Ratio ↔ Agreement 통합

### Fisher Ratio의 신호 이론적 해석

`fisher(d) = (μ_i(d) - μ_j(d))^2 / (σ_i^2(d) + σ_j^2(d) + ε)`

이것은 정확히 **Signal-to-Noise Ratio (SNR)**의 한 형태. 분자는 두 카테고리 간 신호 차이의 제곱, 분모는 각 카테고리 내 노이즈의 합. LDA의 1D 특수 형태와 동치.

### 핵심 통찰: 직교하는 두 정보

| 측면 | Fisher ratio (Catalog) | Agreement (VisualBind) |
|------|----------------------|----------------------|
| 대상 | 차원별 판별력 | 옵저버 간 일관성 |
| 단위 | 카테고리 간 (inter-class) | 프레임 내 (intra-frame) |
| 시간 | 전체 데이터셋 통계 (static) | 실시간 프레임별 (dynamic) |
| 역할 | "어떤 차원이 중요한가" | "이 프레임이 신뢰할 만한가" |

### 통합 제안

```
CrossCheck.weight = base_weight * sqrt(fisher_ratio(signal_a) * fisher_ratio(signal_b))
```

두 신호의 Fisher ratio의 기하 평균으로 weight 조정. 판별력이 높은 신호 쌍의 agreement가 더 중요하게 반영. sqrt는 catalog_scoring.py에서 이미 사용 중인 dynamic range 압축과 일관.

## 종합 권고 (우선순위순)

1. **Agreement 함수 교체**: `a*b` → `1-|a-b|` 기본, 별도 activation level 신호
2. **Softmax 이중 적용 제거**: expression이 이미 확률이면 `normalize="none"`
3. **MinMax range 데이터 적응형 전환**: face.baseline의 Welford stats 활용
4. **Conflict/quiet/missing 3-way 분류**: activation level 기반
5. **Temporal EMA smoothing**: α=0.3, onset 지연으로 인한 false conflict 완화
6. **Fisher-weighted CrossCheck**: Catalog build 시 Fisher ratio를 weight prior로

---

# Round 2: 뇌신경학자

## Multisensory Integration (MSI)과 Inverse Effectiveness

뇌의 다감각 통합에서 가장 중요한 발견:

**약한 단일 감각 신호일수록 다감각 통합의 이득(gain)이 크다.**

시각 신호만으로는 불확실하고, 청각 신호만으로도 불확실할 때 — 두 신호가 **일치**하면 반응 이득이 가장 크다. 강한 단일 감각 신호는 이미 충분한 정보를 가지고 있어 통합 이득이 작다.

VisualBind에서 이 원리의 적용: expression은 미약하고 AU도 미약하지만, 둘이 일치하면(both low) 그것은 "조용한 neutral 상태"라는 유의미한 정보이다. **현재 `a * b` 함수는 inverse effectiveness를 정면으로 위반한다** — both-low를 무시함으로써 가장 통합 이득이 큰 상황을 버린다.

## FACS Temporal Dynamics

표정은 정적이 아니다. 자발적(spontaneous) 미소에서:

- **AU12** (lip corner puller)가 먼저 활성화
- **AU6** (cheek raiser, Duchenne marker)가 **67-170ms 후** 활성화

이 시간차는 30fps 비디오에서 **2-5 프레임**이다. 프레임 단위로 AU6과 AU12를 비교하면 onset 구간에서 false conflict가 발생한다.

### 표정의 생애주기

- **Onset** (~500ms): 근육 활성화 시작. AU가 순차적으로 활성화
- **Apex** (가변): 최대 강도 유지. 모든 AU가 안정적
- **Offset** (~1000ms): 이완. 비대칭적 — 어떤 AU는 빨리, 어떤 것은 천천히

각 phase에서 AU 비율이 다르다. 프레임 독립 비교는 이 dynamics를 놓친다.

## Duchenne Smile과 Conjunction 문제

진짜 미소(Duchenne smile)는 AU6(orbicularis oculi, 안윤근) **AND** AU12(zygomaticus major, 큰광대근)의 **동시** 활성화로 정의된다. 사회적 미소(social smile)는 AU12만 활성화.

현재 CrossCheck 구조는 1:1 관계만 표현 가능하여 이런 conjunction을 직접 표현할 수 없다. `CrossCheck(expression, happy, au, AU12)` + `CrossCheck(expression, happy, au, AU6)`을 별도로 정의하면 각각의 일치는 확인할 수 있지만, "AU6 AND AU12가 동시에 활성일 때만 진짜 미소"라는 conjunction 규칙은 표현할 수 없다.

## Superior Temporal Sulcus (STS) — 다중 감각 얼굴 처리

뇌의 STS 영역은 얼굴의 다중 감각 정보(표정, 시선, 입 움직임)를 통합 처리한다.

흥미로운 점: STS가 **conflict detection**과 **quiet-state recognition**을 **분리 처리**한다. "모순"과 "조용함"은 다른 신경 경로를 탄다. 이것은 VisualBind의 conflict/quiet/missing 3-way 분류 필요성의 신경학적 근거이다.

## Valence-Arousal Dimensional Model

Russell(1980)의 차원적 감정 모델은 모든 감정을 두 축에 배치한다:

- **Valence**: 쾌(pleasant) ↔ 불쾌(unpleasant)
- **Arousal**: 각성(activated) ↔ 이완(deactivated)

```
        High Arousal
            ↑
    Fear    |    Excitement
    Anger   |    Happy
  ←─────────┼──────────→ Valence
    Sad     |    Calm
    Disgust |    Relaxed
            ↓
        Low Arousal
```

VisualBind의 embedding space에서 이 V-A 축이 자연스럽게 나타나는지 post-hoc 분석으로 검증 가능. 나타난다면 embedding의 해석 가능성과 이론적 정당성을 동시에 확보할 수 있다.

## Pose-dependent Processing

시각 피질의 viewpoint-dependent processing: 얼굴 인식 뉴런 자체가 pose-specific tuning을 갖는다. yaw > |30도|에서 expression/AU 측정 신뢰도가 급감하는 것은 단순히 모델의 한계가 아니라 입력 자체의 정보량이 감소하기 때문.

---

# Round 2: 시스템 아키텍트

## Visual* 시리즈 아키텍처 철학

### 계층 구조의 일관성

visual* 시리즈는 Unix 파이프라인 철학을 잘 체현한다. 각 레이어가 명확한 동사를 갖는다:

- **visualbase**: "읽는다" (Read) — 미디어 소스로부터 Frame을 생성
- **visualpath**: "분석한다" (Process) — Frame을 Module로 처리하여 Observation 생산
- **visualbind**: "묶는다" (Bind) — 여러 Observation의 합의를 학습

이 동사의 흐름이 `Frame → Observation → HintFrame → Embedding`이라는 데이터 변환 파이프라인과 정확히 대응한다. 각 레이어의 출력이 다음 레이어의 입력이 되는 composability가 잘 지켜지고 있다.

### 책임 경계 평가

**visualbase**: 경계가 명확하다. Frame, Source, Buffer, IPC라는 4가지 관심사를 다루며, 모두 "미디어 데이터의 획득과 전달"이라는 단일 책임에 속한다.

**visualpath**: Module ABC가 핵심 추상화이며, 잘 설계되어 있다. `process(frame, deps) -> Observation` 시그니처가 간결하면서도 DAG 기반 의존성 해결을 지원한다.

**visualbind**: 가장 좁은 책임. "여러 옵저버의 시그널이 일치하는지 판단하고, 그 합의를 학습 가능한 임베딩으로 변환한다." 4개 클래스, numpy만 의존. **이 간결함 자체가 설계의 강점이다.**

### 범용 vs 도메인 경계

경계가 올바르게 설정. 핵심 증거:

- visualbase의 `Frame`에는 "얼굴"이라는 개념이 없다
- visualpath의 `Module`은 어떤 종류의 분석이든 수용한다
- visualbind의 `CrossCheck`는 AU/expression이라는 도메인 지식을 **주입**받지, **내장**하지 않는다

도메인 지식이 진입하는 지점이 정확하다: vpx 플러그인(face-detect 등)과 momentscan-plugins(face-classify 등)에만 도메인 로직이 있고, visual* 레이어는 이를 모른다.

**재사용 가능성**: visualbase와 visualpath는 portrait981 외 도메인(산업 비전 검사, 스포츠 분석, 의료 영상 등)에서 즉시 재사용 가능하다. visualbind도 "multi-observer agreement를 contrastive learning으로 학습"이라는 패턴은 도메인 독립적. 음성 분석(ASR + 감정 인식), IoT 센서 융합 등에 적용 가능.

### 빠진 레이어 분석

현재 3-레이어(base/path/bind)가 적절하다:

- **visualstore (저장/캐싱)**: momentbank가 앱 레이어에서 수행. 도메인 특화이므로 현 위치가 맞다
- **visualsync (시간 동기화)**: 필요해질 수 있으나 지금은 시기상조
- **visualflow (오케스트레이션)**: visualpath의 FlowGraph + Backend가 이미 수행

"빠진 레이어"를 예방적으로 만드는 것은 과잉 설계. **필요가 증명될 때 추출하는 것이 Unix 철학에 부합.**

## VisualBind 설계 품질

### 4-Stage Pipeline의 SRP

| Stage | 입력 | 출력 | 책임 |
|-------|------|------|------|
| Collect | `Sequence[Observation]` | `HintFrame` | 추출 + 정규화 |
| Agree | `HintFrame` | `AgreementResult` | 교차 검증 |
| Pair | `[HintFrame] + [AgreementResult]` | `PairMiningResult` | 학습 데이터 구성 |
| Encode | `np.ndarray` triplets | `TrainHistory` + 학습된 `_W` | 임베딩 학습 |

SRP가 잘 지켜진다. 각 stage를 독립적으로 테스트하고 교체할 수 있다. Collect만 사용하여 시그널 모니터링, Agree까지만 사용하여 품질 지표 활용 — 이 "부분 사용 가능성"이 좋은 composability의 증거.

### 네이밍 평가

| 이름 | 평가 |
|------|------|
| `HintCollector` | 좋다. "Hint"가 "약한 신호"라는 의미를 전달. "Signal"이면 지나치게 확정적 |
| `AgreementEngine` | 적절. "Engine"은 약간 과장이지만, "Checker"보다 능동적 |
| `PairMiner` | 좋다. Contrastive learning 커뮤니티 관례와 일치 |
| `TripletEncoder` | 약간 혼란. "학습 + 인코딩" 양쪽인데 이름은 인코딩만 암시 |
| `CrossCheck` | 우수. "교차 검증 규칙"이라는 의미가 즉시 전달 |

### 확장 포인트 평가

현재 확장 포인트가 *암묵적*:

- **Agreement 함수**: `_positive_agreement`, `_negative_agreement`가 모듈 레벨 함수로 하드코딩. Strategy 패턴으로 전환 가능
- **Pairing 전략**: positive 선택 로직이 클래스 내부에 고정
- **Encoder 아키텍처**: linear projection 하드코딩

PoC 단계에서 이 경직성은 수용 가능. **먼저 올바른 기본 전략을 확보**하고, 그 다음 교체 가능성을 열어주는 순서.

## Catalog → VisualBind 전환 설계

### extract.py 대체 범위 (정밀 분석)

| 함수 | 줄수 | 대체 가능 | 이유 |
|------|------|----------|------|
| `_extract_face_expression` | 16 | O | 시그널 수치 추출 → HintCollector |
| `_extract_face_au` | 28 | O | 시그널 수치 추출 → HintCollector |
| `_extract_head_pose` | 23 | O | 시그널 수치 추출 → HintCollector |
| `_extract_portrait_score` | 18 | O | CLIP 축 → HintCollector |
| `_compute_composites` | 22 | O | 수동 조합 → 학습된 임베딩이 대체 |
| `_extract_face_detect` | 36 | X | ArcFace identity는 별도 관심사 |
| `_extract_quality` | 7 | X | 기본 품질은 gate 판단용 |
| `_extract_face_classify` | 6 | X | 역할 분류는 binary 결정 |
| `_extract_face_gate` | 23 | X | gate는 binary 결정 |
| `_extract_face_baseline` | 10 | X | 통계 축적은 별도 관심사 |

대체 가능: ~130줄 (스코어링 경로). 잔여: ~230줄 (gate, classify, identity).

### 전환 단계

**Phase A: Shadow Mode** — 기존 catalog scoring 그대로 + visualbind 병렬 실행 + 양쪽 결과 로그 비교. 구현: `_compute_composites()` 이후에 visualbind 결과를 `FrameRecord.metadata`에 추가.

**Phase B: Hybrid Mode** — `final_score = (1-α) × catalog_score + α × bind_score`. α를 config로 외부 주입. 모니터링에서 bind_score가 catalog_score보다 highlight 품질을 개선하는 것이 확인되면 α를 올린다.

**Phase C: Full Replacement** — catalog scoring 코드 제거. HintCollector → AgreementEngine → TripletEncoder.encode(). 결과를 `FrameRecord.bind_embedding`으로 저장.

### 운영 관점

- **모델 배포**: `_W` 행렬 ~1KB numpy 파일. S3 또는 Docker 이미지에 포함
- **A/B 테스트**: Shadow/Hybrid mode가 자연스러운 A/B 프레임워크
- **모니터링**: `AgreementResult.score`, `coverage`, `TrainHistory.separation`이 핵심 메트릭
- **롤백**: α=0.0으로 즉시 catalog-only 모드 복귀

### 최종 판단

> visual* 시리즈는 **"점진적 발견을 통한 프레임워크 추출"**이라는 건전한 방법론으로 만들어졌다. 먼저 momentscan이라는 구체적 앱에서 패턴을 발견하고, 그 패턴을 visualbase → visualpath → visualbind로 한 층씩 추출했다. 이것이 사전에 설계된 "범용 프레임워크"보다 훨씬 실용적이고 정확한 추상화를 만든다.
>
> 전문가 리뷰에서 지적된 3개 즉시 수정 사항(both-low, zero-padding, positive 선택)이 해결되면, catalog → visualbind 전환의 기술적 기반이 갖춰진다. Shadow mode 도입이 리스크 없이 검증을 시작할 수 있는 다음 단계다.

---

# 교차 검증: 전문가 간 합의 매트릭스

## 4/4 만장일치 — 즉시 수정

| # | 이슈 | ML | Signal | Neuro | Arch | 근거 |
|---|------|:--:|:------:|:-----:|:----:|------|
| 1 | `_positive_agreement` both-low | ✓ | ✓ | ✓ | ✓ | neutral 상태의 시스템적 배제. Inverse effectiveness 위반 |
| 2 | `flat_vector()` 차원 불일치 | ✓ | ✓ | ✓ | ✓ | missing source에서 crash. zero-padding 필수 |
| 3 | Positive 선택 기준 오류 | ✓ | ✓ | ✓ | ✓ | agreement proximity ≠ semantic similarity. Co-training 가정 위반 |
| 4 | Pose-dependent confidence | ✓ | ✓ | ✓ | ✓ | yaw > 30도에서 expression/AU 신뢰도 급감 |

## 3/4 강한 권장

| # | 이슈 | ML | Signal | Neuro | Arch |
|---|------|:--:|:------:|:-----:|:----:|
| 5 | Conflict/quiet 3-way 분류 | ✓ | ✓ | ✓ | — |
| 6 | FACS CrossCheck 자동 생성 | ✓ | ✓ | ✓ | — |
| 7 | Temporal dynamics 보정 | — | ✓ | ✓ | ✓ |
| 8 | Softmax double-application | ✓ | ✓ | ✓ | — |

## 2/4 고려

| # | 이슈 | ML | Signal | Neuro | Arch |
|---|------|:--:|:------:|:-----:|:----:|
| 9 | Triplet → InfoNCE | ✓ | — | — | ✓ |
| 10 | Fisher → CrossCheck weight | — | ✓ | — | ✓ |
| 11 | `_W` save/load | — | — | — | ✓ |
| 12 | V-A derived signals | — | — | ✓ | — |

---

# 확정 로드맵 v2 (12-item)

| # | 작업 | 영향 | 난이도 | 합의 |
|---|------|------|--------|------|
| 1 | `_positive_agreement` → 2-channel (similarity + activation) | 매우 높음 | 낮음 | 4/4 |
| 2 | `flat_vector` zero-padding (missing source) | 높음 | 낮음 | 4/4 |
| 3 | Positive 선택 → signal cosine similarity | 높음 | 중간 | 4/4 |
| 4 | Pose-dependent confidence gating | 높음 | 중간 | 4/4 |
| 5 | Conflict/quiet/missing 3-way 분류 | 중간 | 중간 | 3/4 |
| 6 | FACS 매핑 기반 CrossCheck 자동 생성 | 중간 | 낮음 | 3/4 |
| 7 | Temporal smoothing + onset lag 보정 | 중간 | 중간 | 3/4 |
| 8 | Softmax double-application 방지 | 중간 | 낮음 | 3/4 |
| 9 | Triplet → InfoNCE 전환 | 중간 | 중간 | 2/4 |
| 10 | Fisher ratio → CrossCheck weight prior | 낮음 | 낮음 | 2/4 |
| 11 | `_W` save/load interface | 낮음 | 낮음 | 2/4 |
| 12 | V-A derived signal / embedding 해석 | 낮음 | 중간 | 2/4 |
