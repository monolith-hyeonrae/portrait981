# visualbind — 어떻게 풀 것인가

> `why-visualbind.md`에서 다룬 문제의식을 visualbind가 어떤 설계로 해결하는지,
> 구체적인 아키텍처, API, 그리고 PoC 계획을 정리.
> 문제의식은 `why-visualbind.md`, 해법은 이 문서.

---

# Part 1: 아키텍처

## 전체 구조

```
visualbase (I/O)
  → visualpath (추론: Module[] → DAG → Observation[])
      → visualbind (학습: Observation[] → Agreement → Embedding)
```

visualbind는 visualpath의 **소비자**입니다. 기존 인프라를 수정하지 않고 그 위에 학습 계층을 추가합니다.

### 패키지 내부 구조

```
libs/visualbind/
├── src/visualbind/
│   ├── __init__.py
│   ├── _version.py
│   │
│   ├── fusion/              # ① 신호 수집·정규화
│   │   ├── __init__.py
│   │   ├── collector.py     # HintCollector
│   │   ├── hint.py          # HintVector, HintFrame
│   │   └── normalizer.py    # 정규화 전략들
│   │
│   ├── agreement/           # ② 합의 계산
│   │   ├── __init__.py
│   │   ├── engine.py        # AgreementEngine
│   │   ├── strategy.py      # SoftConsensus, CrossCorrelation, ...
│   │   └── map.py           # AgreementMap (프레임별 합의 결과)
│   │
│   ├── pairing/             # ③ 대조 쌍 구성
│   │   ├── __init__.py
│   │   ├── sampler.py       # ContrastiveSampler
│   │   ├── pair.py          # ContrastivePair, PairDataset
│   │   └── mining.py        # HardNegative, SemiHard 전략
│   │
│   ├── encoding/            # ④ 임베딩 학습
│   │   ├── __init__.py
│   │   ├── trainer.py       # EngramTrainer
│   │   ├── loss.py          # SoftAgreementContrastiveLoss
│   │   ├── backbone.py      # Backbone 레지스트리
│   │   └── regularizer.py   # EntropyRegularizer
│   │
│   └── pipeline/            # 전체 파이프라인 오케스트레이션
│       ├── __init__.py
│       └── runner.py        # BindPipeline
│
├── tests/
│   ├── test_fusion/
│   ├── test_agreement/
│   ├── test_pairing/
│   └── test_encoding/
│
└── pyproject.toml
```

---

## 네 단계의 파이프라인

### ① Fusion: 신호 수집

**해결하는 문제**: visualpath Observation은 모듈별로 분산되어 있고, 신호의 스케일과 의미가 제각각

visualpath의 Observation.signals를 **hint**로 재해석합니다. "재해석"이지 "변환"이 아닙니다 — Observation의 구조를 그대로 읽고, 정규화만 적용합니다.

```python
from visualbind.fusion import HintCollector, HintFrame

# Observation.signals dict를 직접 소비
collector = HintCollector(
    sources={
        "face.expression": {
            "signals": ["happy", "neutral", "sad", "angry", "surprise"],
            "normalize": "softmax",      # 이미 확률이면 identity
        },
        "face.au": {
            "signals": ["AU1", "AU2", "AU4", "AU6", "AU12", "AU25"],
            "normalize": "sigmoid",       # 0~1로 클리핑
        },
        "head.pose": {
            "signals": ["yaw", "pitch", "roll"],
            "normalize": "minmax",        # 각도 → 0~1
            "range": {"yaw": (-90, 90), "pitch": (-45, 45), "roll": (-30, 30)},
        },
    },
)

# 프레임 하나에 대한 전체 hint 수집
# observations: Dict[str, Observation] — visualpath가 이미 수집한 것
hint_frame: HintFrame = collector.collect(frame_id=42, observations=observations)
```

#### HintFrame 구조

```python
@dataclass
class HintVector:
    """단일 모듈의 정규화된 출력."""
    source: str                    # 모듈 이름 (e.g., "face.expression")
    values: np.ndarray             # 정규화된 신호 벡터
    confidence: float              # 모듈 자체의 신뢰도 [0, 1]
    signal_names: tuple[str, ...]  # 각 차원의 이름

@dataclass
class HintFrame:
    """한 프레임에 대한 모든 hint의 묶음."""
    frame_id: int
    t_ns: int
    hints: Dict[str, HintVector]   # source_name → HintVector

    @property
    def sources(self) -> list[str]:
        return list(self.hints.keys())

    def get(self, source: str) -> Optional[HintVector]:
        return self.hints.get(source)
```

#### 왜 Observation을 직접 안 쓰는가

Observation은 범용 컨테이너이고 signals의 의미와 스케일이 모듈마다 다릅니다. HintVector는 정규화된 벡터로 **비교 가능한 형태**를 보장합니다. 하지만 Observation → HintVector 변환은 `collector.collect()` 한 줄이며, 새로운 타입 체계를 강요하지 않습니다.

---

### ② Agreement: 합의 계산

**해결하는 문제**: 어떤 프레임이 "신뢰할 수 있는" 학습 샘플인지 판단

같은 프레임에 대한 여러 hint가 **일관된 방향**을 가리키는지 측정합니다. 이것이 visualbind의 핵심 지능입니다.

```python
from visualbind.agreement import AgreementEngine, SoftConsensus

engine = AgreementEngine(
    strategy=SoftConsensus(
        # 어떤 hint 쌍이 서로를 검증하는지 정의
        cross_checks=[
            # expression의 "happy"와 AU의 "AU6", "AU12"는 같은 방향이어야 함
            CrossCheck(
                source_a="face.expression", signal_a="happy",
                source_b="face.au", signals_b=["AU6", "AU12"],
                relation="positive",  # 둘 다 높거나 둘 다 낮아야 합의
            ),
            # head.pose의 yaw가 극단적이면 face.quality는 낮아야 정상
            CrossCheck(
                source_a="head.pose", signal_a="yaw",
                source_b="face.quality", signal_b="blur",
                relation="positive",  # 고개 돌림 ↑ → 블러 ↑
            ),
        ],
    ),
    confidence_weighted=True,  # 각 hint의 자체 confidence 반영
)

# 단일 프레임 합의 계산
agreement: AgreementMap = engine.evaluate(hint_frame)
# agreement.score  → float [0, 1] (전체 합의도)
# agreement.details → Dict[str, float] (cross_check별 합의도)
# agreement.confident → bool (학습에 쓸 만한 수준인가)
```

#### AgreementMap

```python
@dataclass
class AgreementMap:
    """프레임 하나에 대한 합의 계산 결과."""
    frame_id: int
    score: float                    # 전체 합의도 [0, 1]
    details: Dict[str, float]      # cross_check별 점수
    hint_frame: HintFrame          # 원본 hint 참조

    @property
    def confident(self) -> bool:
        """학습에 사용할 만한 합의 수준인가."""
        return self.score > self._threshold

    @property
    def direction(self) -> np.ndarray:
        """합의가 가리키는 방향 벡터. pairing에서 사용."""
        ...
```

#### 교차 검증(CrossCheck)의 의미

CrossCheck는 **도메인 지식의 주입점**입니다. "happy 표정이면 AU6, AU12가 높아야 한다"는 지식은 얼굴 분석 도메인에서 온 것이지만, CrossCheck 구조 자체는 범용입니다:

```python
# 자율주행: 차량 감지기와 레이더의 거리가 일치해야 함
CrossCheck(source_a="object.detect", signal_a="distance",
           source_b="radar", signal_b="range",
           relation="positive")

# 의료: 병변 크기와 텍스처 이상도가 비례해야 함
CrossCheck(source_a="lesion.detect", signal_a="area",
           source_b="texture.analyze", signal_b="irregularity",
           relation="positive")
```

도메인이 바뀌어도 **CrossCheck를 교체하면** 같은 엔진이 동작합니다.

#### 선언적 전략 교체

AgreementEngine은 strategy를 주입받습니다. SoftConsensus 외에도:

| 전략 | 방식 | 적합한 상황 |
|------|------|------------|
| `SoftConsensus` | 선언된 cross_check의 가중 평균 | 도메인 지식이 있을 때 (기본값) |
| `CrossCorrelation` | hint 벡터 간 상관계수 | 사전 지식 없이 탐색적으로 |
| `MutualInformation` | hint 분포 간 상호 정보량 | 비선형 관계 포착 필요 시 |
| `Unanimous` | 모든 hint가 같은 방향 (엄격) | 고품질 소수 샘플 필요 시 |

전략은 `AgreementStrategy` 프로토콜을 구현하면 됩니다:

```python
class AgreementStrategy(Protocol):
    def evaluate(self, hint_frame: HintFrame) -> AgreementMap: ...
```

---

### ③ Pairing: 대조 쌍 구성

**해결하는 문제**: 합의 결과로부터 contrastive learning에 쓸 (positive, negative) 쌍 구성

합의가 강한 프레임들 중에서, **유사한 방향의 합의** → positive pair, **반대 방향** → negative pair를 구성합니다. 합의가 약한 프레임은 **버립니다**.

```python
from visualbind.pairing import ContrastiveSampler

sampler = ContrastiveSampler(
    positive_threshold=0.8,   # agreement.score >= 0.8인 프레임만 positive 후보
    negative_threshold=0.3,   # agreement.score >= 0.3이지만 방향이 반대
    drop_ambiguous=True,      # 0.3 < score < 0.8인 프레임은 제외
    mining_strategy="semi_hard",  # semi-hard negative mining
)

# 비디오 전체의 agreement 결과로부터 pair 구성
agreements: list[AgreementMap] = [engine.evaluate(hf) for hf in hint_frames]
pairs: PairDataset = sampler.sample(agreements)
```

#### PairDataset

```python
@dataclass
class ContrastivePair:
    """학습용 대조 쌍."""
    anchor_frame_id: int
    positive_frame_id: int       # anchor와 합의 방향이 유사
    negative_frame_id: int       # anchor와 합의 방향이 반대
    anchor_agreement: float      # anchor의 합의 점수
    pair_similarity: float       # anchor-positive 간 합의 유사도
    pair_distance: float         # anchor-negative 간 합의 거리

class PairDataset:
    """ContrastivePair의 컬렉션. PyTorch Dataset 호환."""
    pairs: list[ContrastivePair]

    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    # → (anchor_image, positive_image, negative_image)

    @property
    def stats(self) -> PairStats:
        """pair 구성 통계: 사용률, 양성/음성 비율, 합의 분포."""
        ...
```

#### 왜 "중간"을 버리는가

```
합의 분포:

    강한 합의 (>0.8)     약한 영역 (0.3~0.8)      강한 불일치 (<0.3)
    ┌───────────┐       ┌─────────────┐          ┌──────────────┐
    │ positive  │       │   DROP      │          │  negative    │
    │ candidates│       │ (ambiguous) │          │  candidates  │
    └───────────┘       └─────────────┘          └──────────────┘
```

Noisy contrastive learning 연구가 일관되게 보여주는 것: **확실한 쌍만 쓰는 것이 전체를 쓰는 것보다 낫습니다**. 중간 영역의 샘플은 학습 신호에 noise를 추가할 뿐 정보를 추가하지 않습니다.

데이터가 부족하지 않습니다. 981파크에서 하루에 수천 명, 초당 30프레임이면 프레임은 수백만 개입니다. 90%를 버려도 충분합니다. **Less is more.**

---

### ④ Encoding: 임베딩 학습

**해결하는 문제**: 구성된 pair로부터 의미 있는 임베딩 공간 학습

```python
from visualbind.encoding import EngramTrainer, SoftAgreementContrastiveLoss

trainer = EngramTrainer(
    backbone="adaface_ir50",       # 또는 "dinov2_vits14", "resnet50", ...
    loss=SoftAgreementContrastiveLoss(
        temperature=0.07,
        agreement_weight=True,     # 합의 강도를 loss weight로 사용
    ),
    regularizer=EntropyRegularizer(
        lambda_=0.1,               # 임베딩 분포의 균일성 유지
    ),
    embedding_dim=512,             # 최종 임베딩 차원
    projection_head=True,          # 학습 시 projection head 사용
)

# 학습
trainer.fit(
    pair_dataset=pairs,
    image_loader=image_loader,     # frame_id → image array
    epochs=50,
    batch_size=64,
)

# 추론: 학습된 임베딩 추출
embedding = trainer.encode(image)  # → np.ndarray (512,)
```

#### SoftAgreementContrastiveLoss

전통적 contrastive loss (InfoNCE)에서 **합의 강도가 soft label**로 작용합니다:

```
L = -Σ w_i · log(exp(sim(a, p_i) / τ) / Σ exp(sim(a, n_j) / τ))

w_i = agreement_score(pair_i)    ← 합의가 강한 pair일수록 loss 기여 ↑
```

합의가 0.95인 pair는 0.80인 pair보다 더 큰 가중치를 받습니다. 이것이 "soft" label입니다 — 이진 판단이 아니라 연속적 신뢰도가 학습에 반영됩니다.

#### EntropyRegularizer

임베딩 공간이 소수 클러스터로 붕괴(collapse)하는 것을 방지합니다. 미니배치 내 임베딩 분포의 엔트로피를 최대화하여 **고르게 분포된 임베딩 공간**을 유지합니다.

```
L_reg = -λ · H(p(z))

H = -Σ p_i · log(p_i)    ← 미니배치 임베딩의 분포 엔트로피
```

SimCLR의 uniformity loss, Barlow Twins의 cross-correlation regularizer와 같은 맥락입니다.

#### Backbone 레지스트리

```python
# 사전 등록된 백본
BACKBONE_REGISTRY = {
    "adaface_ir50":   AdaFaceBackbone,     # face-specific
    "dinov2_vits14":  DINOv2Backbone,      # general vision
    "resnet50":       ResNetBackbone,      # lightweight
    "siglip_base":    SigLIPBackbone,      # vision-language
}

# 커스텀 백본 등록
@register_backbone("my_model")
class MyBackbone(BackboneBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @property
    def output_dim(self) -> int: ...
```

도메인에 따라 백본만 교체하면 됩니다. 얼굴이면 AdaFace, 일반 비전이면 DINOv2, 의료면 도메인 특화 모델.

---

## visualpath 연동 상세

### Observation → Hint 변환 흐름

```
visualpath FlowGraph
    │
    ├── face.detect     → Observation(signals={"face_count": 2, ...})
    ├── face.expression → Observation(signals={"happy": 0.8, ...})
    ├── face.au         → Observation(signals={"AU6": 0.7, ...})
    ├── head.pose       → Observation(signals={"yaw": -15, ...})
    └── face.quality    → Observation(signals={"blur": 12.3, ...})
         │
         │  HintCollector.collect()
         ▼
    HintFrame(
        hints={
            "face.expression": HintVector([0.8, 0.15, 0.05], conf=0.92),
            "face.au":         HintVector([0.1, 0.0, 0.1, 0.7, 0.9, 0.3], conf=0.88),
            "head.pose":       HintVector([0.42, 0.56, 0.53], conf=0.95),
        }
    )
```

### visualpath App과의 통합

visualbind는 visualpath의 App 라이프사이클에 자연스럽게 끼워 넣을 수 있습니다:

```python
import visualpath as vp
from visualbind.fusion import HintCollector
from visualbind.agreement import AgreementEngine

class BindCollectorApp(vp.App):
    """visualpath App으로 hint를 수집하는 예시."""

    def setup(self):
        self.collector = HintCollector(sources={...})
        self.hint_frames = []

    def on_frame(self, frame, results: Dict[str, Observation]):
        # visualpath가 이미 수집한 Observation dict를 그대로 사용
        hint_frame = self.collector.collect(
            frame_id=frame.frame_id,
            observations=results,
        )
        self.hint_frames.append(hint_frame)

    def after_run(self, result):
        # 비디오 전체 수집 완료 후 agreement 계산
        engine = AgreementEngine(strategy=SoftConsensus(...))
        agreements = [engine.evaluate(hf) for hf in self.hint_frames]
        # → 이후 pairing, encoding으로 전달
        return agreements
```

하지만 visualbind가 visualpath.App을 **강제하지는 않습니다**. Observation dict만 있으면 독립적으로 사용할 수 있습니다.

### FlowGraph에 영향 없음

visualbind는 FlowGraph의 **출력을 소비**할 뿐, FlowGraph 자체에 노드를 추가하지 않습니다. 추론(visualpath)과 학습(visualbind)은 분리된 단계입니다:

```
[추론 단계]  FlowGraph.run(video) → Observation[]   ← visualpath
[수집 단계]  HintCollector(Observation[]) → HintFrame[]  ← visualbind
[학습 단계]  Agreement → Pairing → Encoding  ← visualbind (오프라인)
```

이 분리는 의도적입니다. 추론은 실시간/배치 모두 가능하지만, 학습은 반드시 오프라인(배치)입니다. 두 관심사를 섞지 않습니다.

---

# Part 2: 설계 결정

## 1. Observation을 래핑하지 않고 읽는다

**선택지**:
- (A) Observation을 상속한 HintObservation 타입 정의
- (B) Observation.signals를 읽어서 HintVector로 변환 (선택)

**(B)를 선택한 이유**: Observation은 visualpath의 핵심 타입이고, 이것을 상속하거나 래핑하면 visualpath와의 결합도가 올라갑니다. HintCollector가 signals dict를 **읽기만** 하면 visualpath 버전 변경에 영향을 받지 않습니다.

## 2. CrossCheck는 선언적이되 필수가 아니다

**선택지**:
- (A) CrossCheck 없이 순수 통계적 합의만 (CrossCorrelation)
- (B) CrossCheck를 필수로 요구
- (C) CrossCheck는 있으면 좋지만 없어도 동작 (선택)

**(C)를 선택한 이유**: 도메인 지식이 있으면 훨씬 좋은 합의를 계산할 수 있지만, 없어도 통계적 방법(CrossCorrelation)으로 시작할 수 있어야 합니다. 새 도메인에 적용할 때 처음부터 CrossCheck를 정의하라고 하면 진입 장벽이 높아집니다.

```python
# 도메인 지식 없이도 시작 가능
engine = AgreementEngine(strategy=CrossCorrelation())

# 도메인 지식이 있으면 정밀도 향상
engine = AgreementEngine(strategy=SoftConsensus(cross_checks=[...]))
```

## 3. 추론과 학습의 명시적 분리

**선택지**:
- (A) FlowGraph에 학습 노드를 추가 (end-to-end)
- (B) 추론 출력을 저장 → 오프라인 학습 (선택)

**(B)를 선택한 이유**: 학습은 비디오 전체(또는 다수 비디오)를 본 후에야 의미 있습니다. FlowGraph는 프레임 단위 실행에 최적화되어 있고, 비디오 전체 집계는 이미 App.after_run()에서 처리합니다. 학습 노드를 FlowGraph에 넣으면 FlowGraph의 실행 모델(프레임 단위)과 학습의 실행 모델(배치)이 충돌합니다.

## 4. 프레임워크 의존성 최소화

**선택지**:
- (A) PyTorch 필수
- (B) PyTorch 선택적, numpy 기반 코어 (선택)

**(B)를 선택한 이유**: fusion, agreement, pairing은 순수 numpy로 구현 가능합니다. PyTorch는 encoding 단계에서만 필요합니다. 이렇게 하면:

```toml
# pyproject.toml
[project]
dependencies = ["numpy"]

[project.optional-dependencies]
train = ["torch>=2.0"]
```

hint 수집과 합의 계산만 하고 학습은 다른 도구로 하겠다는 사용자도 수용할 수 있습니다.

## 5. 이미지 로딩은 외부 책임

**선택지**:
- (A) visualbind가 frame_id → 이미지 로딩을 직접 수행
- (B) image_loader를 주입받음 (선택)

**(B)를 선택한 이유**: 이미지 소스가 다양합니다 — visualbase.FileSource에서 직접 읽을 수도, 디스크에 저장된 crop을 읽을 수도, momentbank에서 가져올 수도 있습니다. visualbind가 이 모든 경우를 알 필요가 없습니다.

```python
# image_loader Protocol
class ImageLoader(Protocol):
    def load(self, frame_id: int) -> np.ndarray: ...
```

---

# Part 3: portrait981 적용

## 적용 시나리오

portrait981에서 visualbind를 처음 적용할 대상은 **Portrait Embedding**입니다 — "이 사람의 이 순간"을 하나의 벡터로 표현하는 임베딩.

### 현재 상태

momentscan Phase 3이 사용하는 임베딩:

| 임베딩 | 모델 | 용도 |
|--------|------|------|
| Face-ID | InsightFace ArcFace | 동일 인물 판정 (identity) |
| General Vision | DINOv2 / SigLIP (TBD) | 다양성/참신함 (novelty) |

이 두 임베딩은 **사전 학습된 모델의 출력을 그대로 사용**합니다. 우리 도메인(981파크 고객 초상화)에 특화되어 있지 않습니다.

### visualbind로 달라지는 것

```
현재: face_embed = ArcFace(face_crop)        ← 범용 face-ID, 우리 도메인 무시
     vision_embed = DINOv2(upper_crop)       ← 범용 vision, 우리 도메인 무시

이후: portrait_embed = BindEncoder(face_crop) ← 우리의 14개 모듈이 합의한 품질 기준 반영
```

Portrait Embedding이 반영하는 것:
- 진짜 미소 vs 사회적 미소 (expression + AU 합의)
- 자연스러운 포즈 vs 어색한 포즈 (head.pose + body.pose 합의)
- 기술적 품질 (blur + exposure + face.quality 합의)
- **이 세 축의 조합** (개별 모듈로는 불가능한 통합 판단)

### 구체적 적용 계획

```python
# portrait981 전용 설정
collector = HintCollector(
    sources={
        "face.expression": {
            "signals": ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"],
            "normalize": "identity",    # 이미 softmax 출력
        },
        "face.au": {
            "signals": ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7",
                        "AU9", "AU10", "AU12", "AU14", "AU15",
                        "AU17", "AU20", "AU23", "AU25", "AU26"],
            "normalize": "sigmoid",
        },
        "head.pose": {
            "signals": ["yaw", "pitch", "roll"],
            "normalize": "minmax",
            "range": {"yaw": (-90, 90), "pitch": (-45, 45), "roll": (-30, 30)},
        },
        "face.quality": {
            "signals": ["blur", "exposure", "face_area_ratio"],
            "normalize": "minmax",
            "range": {"blur": (0, 500), "exposure": (0, 255), "face_area_ratio": (0, 0.5)},
        },
    },
)

engine = AgreementEngine(
    strategy=SoftConsensus(
        cross_checks=[
            # 뒤셴 미소: happy + (AU6 + AU12) → 합의
            CrossCheck("face.expression", "happy",
                       "face.au", ["AU6", "AU12"],
                       relation="positive"),
            # 놀람: surprise + (AU1 + AU2 + AU5 + AU25 + AU26) → 합의
            CrossCheck("face.expression", "surprise",
                       "face.au", ["AU1", "AU2", "AU5", "AU25", "AU26"],
                       relation="positive"),
            # 정면일수록 선명해야 함
            CrossCheck("head.pose", "yaw",
                       "face.quality", "blur",
                       relation="positive"),  # yaw↑ (측면) → blur 허용
            # 얼굴이 클수록 품질이 좋아야 함
            CrossCheck("face.quality", "face_area_ratio",
                       "face.quality", "blur",
                       relation="negative"),  # 큰 얼굴인데 블러 → 비정상
        ],
    ),
)

trainer = EngramTrainer(
    backbone="adaface_ir50",
    loss=SoftAgreementContrastiveLoss(temperature=0.07),
    regularizer=EntropyRegularizer(lambda_=0.1),
    embedding_dim=512,
)
```

### momentbank과의 연동

학습된 Portrait Embedding은 momentbank의 MemoryNode에서 기존 Face-ID 임베딩과 **병렬로** 사용됩니다:

```
MemoryNode:
    vec_id: ArcFace embedding     ← 동일 인물 매칭 (변경 없음)
    vec_portrait: Portrait embed  ← 초상화 품질/스타일 매칭 (NEW)
```

- `vec_id`: "같은 사람인가?" → identity 판정
- `vec_portrait`: "이 사람의 가장 좋은 순간인가?" → quality/style 판정

reportrait가 참조 이미지를 선택할 때, vec_id로 인물을 필터하고 vec_portrait으로 최적 이미지를 선택합니다.

---

# Part 4: PoC 결과 (2026-03-12 완료)

## PoC 구현 — numpy-only, 4-stage pipeline

설계 문서의 API 이름은 PoC 과정에서 더 간결하게 조정됨:

| 설계 문서 | 실제 구현 | 이유 |
|-----------|----------|------|
| `ContrastiveSampler` | `PairMiner` | 역할이 명확 — pair를 "채굴"함 |
| `EngramTrainer` | `TripletEncoder` | PoC는 linear projection만 (backbone 없음) |
| `SoftConsensus` | `AgreementEngine` + `CrossCheck` | strategy 패턴 대신 직접 CrossCheck 목록 |
| `AgreementMap` | `AgreementResult` | 네이밍 간소화 |

### 구현 현황

```
libs/visualbind/
├── src/visualbind/
│   ├── __init__.py          # 공개 API re-export
│   ├── types.py             # SourceSpec, HintVector, HintFrame, CrossCheck, AgreementResult
│   ├── collector.py         # HintCollector (4개 normalizer: none/minmax/sigmoid/softmax)
│   ├── agreement.py         # AgreementEngine (positive/negative relation, weighted scoring)
│   ├── pairing.py           # PairMiner (threshold 기반 triplet mining)
│   └── encoder.py           # TripletEncoder (linear projection, triplet margin loss, numpy GD)
├── demos/
│   └── demo_pose_expression.py   # 4 states × 200 samples, 3 observers, 11D signals
├── tests/                    # 58 tests
└── pyproject.toml
```

### Demo 결과

`demo_pose_expression.py`: Pose + Expression + AU → 4D unified embedding

- 4 states: happy_frontal, happy_turned, neutral_frontal, surprise_open
- 200 samples/state, 3 observers (expression 4D + AU 4D + pose 3D = 11D)
- 4 CrossChecks: smile↔AU12, smile↔AU6, surprise↔AU26, surprise↔AU25
- **kNN accuracy (k=5): 94.2%** — 합의 기반 pseudo-label만으로 상태 분리 성공

### 성공 기준 대비

| 기준 | 목표 | 결과 | 상태 |
|------|------|------|------|
| kNN accuracy | - | 94.2% | ✅ PoC 수준 충분 |
| 합의-품질 상관 | Kendall τ ≥ 0.5 | 실제 비디오 미적용 | 🔄 momentscan 통합 후 검증 |
| pair 사용률 | ≥ 5% | synthetic 100% | ✅ (실제 비디오에서 재검증 필요) |
| 클러스터 분리 | Silhouette ≥ 0.3 | 4-state 분리 확인 | ✅ |

---

# Part 5: 전문가 리뷰 결과 (2026-03-12)

2라운드 전문가 리뷰 수행. Round 1 (ML, Sensor Fusion, Systems), Round 2 (ML/AI, Signal Processing, Neuroscience, Architecture).

## 만장일치 이슈 (4/4 전문가 동의)

### 1. `_positive_agreement`의 both-low 문제

현재 `a * b` 함수는 co-activation만 탐지. neutral/calm 상태(`0.1 * 0.1 = 0.01`)가 체계적으로 negative pool에 빠짐.

**뇌신경학적 근거**: MSI(Multisensory Integration)의 "inverse effectiveness" — 약한 신호의 통합이 최대 이득. 현재 구현은 정반대.

**대안 (신호처리 응답곡선 분석)**:
- `a * b`: (0.1,0.1)→0.01, (0.5,0.5)→0.25, (0.9,0.9)→0.81 — 비선형, low-bias
- `1-|a-b|`: (0.1,0.1)→1.0, (0.5,0.5)→1.0, (0.9,0.9)→1.0 — 선형, activation 무시
- `2ab/(a+b)`: (0.1,0.1)→0.1, (0.5,0.5)→0.5 — 조화평균, 부분 보정

**결론**: 단일 함수로 일치도와 활성화 수준을 동시에 표현 불가. 2-channel 접근 권장:
- Channel 1: similarity = `1 - |a - b|`
- Channel 2: activation = `(a + b) / 2`

### 2. `flat_vector()` 차원 불일치

present한 source만 포함 → source 누락 시 벡터 차원 변동 → `_W` 행렬 crash. zero-padding 필수.

### 3. Positive 선택 기준 오류

Agreement score의 수치적 근접성으로 positive 선택 중 → 다른 상태끼리 pair 가능.
**ML 관점**: co-training(Blum & Mitchell 1998) 핵심 가정 위반. Signal vector cosine similarity로 교체.

### 4. Pose-dependent confidence gating

yaw > |30°|에서 expression/AU 신뢰도 급감. CrossCheck의 static relation으로는 조건부 규칙 불가.

## 강한 권장 (3/4)

| # | 이슈 | 핵심 |
|---|------|------|
| 5 | Conflict vs quiet 구분 | 3-way 분류 필요 (conflict/quiet/missing) |
| 6 | FACS 기반 CrossCheck 자동 생성 | 현재 happy/surprise만 → 6 emotions 커버 가능 |
| 7 | Temporal dynamics | AU6-AU12 onset lag (67-170ms) → frame-level false conflict |
| 8 | Softmax double-application | HSEmotion 확률 출력에 softmax 재적용 → MI 50%↓ |

## 학술적 포지셔닝

- **논문**: "Few-shot prototype matching in agreement-supervised metric spaces" (FG/ECCV)
- **이론적 연결**: Co-training (Blum & Mitchell 1998), Prototypical Networks (Snell 2017)
- **Catalog = hand-crafted prototypical network** → visualbind = data-driven 전환
- **뇌신경학적 기반**: Binding Problem, MSI inverse effectiveness, STS multimodal processing

## 개선 로드맵 v2 (12-item)

| # | 작업 | 영향 | 난이도 |
|---|------|------|--------|
| 1 | `_positive_agreement` → 2-channel (similarity + activation) | 매우 높음 | 낮음 |
| 2 | `flat_vector` zero-padding | 높음 | 낮음 |
| 3 | Positive 선택 → signal cosine similarity | 높음 | 중간 |
| 4 | Pose-dependent confidence gating | 높음 | 중간 |
| 5 | Conflict/quiet/missing 3-way 분류 | 중간 | 중간 |
| 6 | FACS 매핑 기반 CrossCheck 자동 생성 | 중간 | 낮음 |
| 7 | Temporal smoothing + onset lag 보정 | 중간 | 중간 |
| 8 | Softmax double-application 방지 | 중간 | 낮음 |
| 9 | Triplet → InfoNCE 전환 | 중간 | 중간 |
| 10 | Fisher ratio → CrossCheck weight prior | 낮음 | 낮음 |
| 11 | `_W` save/load interface | 낮음 | 낮음 |
| 12 | V-A derived signal / embedding 해석 | 낮음 | 중간 |

---

# Part 6: Catalog → VisualBind 전환 전략

## 결정 (2026-03-12)

**즉시 흡수하지 않고 점진적 교체.**
visualbind의 기능과 퍼포먼스를 비교하며, 충분히 catalog를 대체할 수 있을 때 완전 교체.

## 흡수 가능 영역 (catalog_scoring.py)

| Catalog 기능 | VisualBind 대응 |
|-------------|----------------|
| Signal normalization (21D vector) | `HintCollector` + `SourceSpec` 선언적 전환 |
| Distance calculation (weighted Euclidean) | `TripletEncoder` embedding distance |
| Category matching (Fisher-weighted similarity) | Embedding space nearest neighbor |
| `SIGNAL_FIELDS`, `SIGNAL_RANGES` | `SourceSpec.signals`, `SourceSpec.range` |
| `compute_importance_weights()` (Fisher ratio) | CrossCheck weight 초기값으로 활용 가능 |

## 유지 영역 (visualbind 대상 아님)

- face_gate (binary decision)
- face_classify (역할 분류)
- identity (ArcFace embedding)
- Category YAML definitions (도메인 설정)
- Reference image management (momentbank)

## 전환 단계

```
Phase A — Shadow mode: 두 시스템 병렬 실행, 결과 로깅만
Phase B — Hybrid: α × visualbind + (1-α) × catalog 블렌딩
Phase C — Full replacement: catalog 제거
```

### 전환 가교

- Fisher ratio → CrossCheck weight 초기값
- Category centroids → embedding space prototype 초기값
- `SIGNAL_RANGES` → `SourceSpec.range` 직접 매핑

---

# Part 7: 남은 질문 (업데이트)

## 해결된 결정들

| 질문 | 결정 | 이유 |
|------|------|------|
| PoC 프레임워크 | numpy-only | PyTorch 없이 4-stage 검증 완료 |
| agreement의 temporal 고려 | 프레임 독립으로 시작 (PoC) | 전문가 리뷰 후 temporal 보정 로드맵 #7로 편입 |
| multi-person 처리 | 1명 | momentscan Phase 3과 동일 |
| catalog 흡수 시점 | 점진적 전환 | Shadow → Hybrid → Full |

## 열린 결정들

| 질문 | 선택지 | 현재 생각 |
|------|--------|----------|
| backbone 구조 | linear vs MLP vs pretrained | linear PoC 완료, MLP 실험 후 판단 |
| 임베딩 차원 | 4 (PoC) / 8~16 (production) / 512 | PoC에서 4D 충분, production은 신호 수에 따라 |
| InfoNCE 전환 시점 | 로드맵 #9 | triplet 한계 체감 시 전환 |
| 학습 빈도 | 일일 야간 배치 vs on-demand | 야간 배치 권장 (architecture 전문가) |
| agreement function | 2-channel | 로드맵 #1에서 구현 후 A/B 검증 |

## visualpath 진화가 필요한 부분

| 필요 | 현재 상태 | 영향 |
|------|----------|------|
| Observation 저장/로드 | App에서 직접 구현 | visualbind 오프라인 학습 시 직렬화 필요 |
| 비디오 배치 처리 | App에서 직접 루프 | 다수 비디오에서 hint 수집 시 편의성 |
