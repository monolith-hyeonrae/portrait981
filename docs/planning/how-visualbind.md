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

# Part 4: PoC 계획

## PoC 범위

전체 파이프라인을 한 번 관통하는 최소 구현:

### Stage 1: Fusion + Agreement (numpy만)

**목표**: momentscan의 기존 분석 결과에서 hint를 수집하고 합의도를 계산

**입력**: momentscan 출력 (이미 있는 비디오 분석 결과)
**출력**: 프레임별 agreement score 분포

**검증**: 수동으로 확인한 "좋은 프레임"이 높은 agreement score를 받는지

```
구현: HintCollector, HintVector, HintFrame
      AgreementEngine, SoftConsensus, CrossCheck
      AgreementMap
```

### Stage 2: Pairing (numpy만)

**목표**: agreement 결과로부터 contrastive pair를 구성하고 품질 확인

**입력**: Stage 1의 agreement 결과
**출력**: ContrastivePair 리스트, 사용률/분포 통계

**검증**: positive pair의 프레임들이 시각적으로 유사한 품질/분위기인지, negative pair가 명확히 다른지

```
구현: ContrastiveSampler, ContrastivePair, PairDataset
```

### Stage 3: Encoding (PyTorch)

**목표**: pair로 AdaFace 백본을 fine-tune하고 임베딩 품질 평가

**입력**: Stage 2의 PairDataset + 원본 이미지
**출력**: 학습된 임베딩 모델, t-SNE 시각화

**검증**:
- t-SNE에서 "좋은 초상화" 클러스터가 형성되는지
- 기존 ArcFace 대비 초상화 품질 구분력 향상 여부
- momentbank select_refs()에 적용 시 선택 품질 변화

```
구현: EngramTrainer, SoftAgreementContrastiveLoss
      EntropyRegularizer, AdaFaceBackbone
```

## PoC 일정 (비순차적으로 진행 가능한 부분 표시)

```
Stage 1 (Fusion + Agreement)
   ├── HintCollector 구현
   ├── AgreementEngine + SoftConsensus 구현
   └── momentscan 출력으로 검증
         │
Stage 2 (Pairing)              ← Stage 1 완료 후
   ├── ContrastiveSampler 구현
   └── pair 품질 시각적 검증
         │
Stage 3 (Encoding)             ← Stage 2 완료 후
   ├── EngramTrainer 구현
   ├── AdaFace fine-tune 실험
   └── t-SNE 시각화 + momentbank 연동 테스트
```

## 성공 기준

| 기준 | 측정 방법 |
|------|----------|
| agreement score와 주관적 품질의 상관 | Kendall τ ≥ 0.5 |
| pair 사용률 (전체 프레임 중 pair에 포함된 비율) | ≥ 5% |
| positive pair 내 시각적 일관성 | 수동 검증 80% 이상 동의 |
| t-SNE 클러스터 분리도 | Silhouette score ≥ 0.3 |
| momentbank 선택 품질 | 기존 대비 MOS(Mean Opinion Score) ≥ +0.5 |

---

# Part 5: 남은 질문

## 열린 결정들

| 질문 | 선택지 | 현재 생각 |
|------|--------|----------|
| 학습 데이터 단위 | 비디오 1개 vs 다수 비디오 | 1개로 시작, 효과 확인 후 확장 |
| backbone freeze 여부 | 전체 fine-tune vs head만 | head만으로 시작 (데이터 적을 때) |
| agreement의 temporal 고려 | 프레임 독립 vs 시간 연속성 | 프레임 독립으로 시작 |
| multi-person 처리 | 프레임당 주인공 1명 vs 다수 | 1명 (momentscan Phase 3과 동일) |
| 임베딩 차원 | 128 / 256 / 512 | 512 (ArcFace 호환) |

## visualpath 진화가 필요한 부분

visualbind PoC는 visualpath의 **현재 기능만으로** 진행 가능합니다. 하지만 장기적으로:

| 필요 | 현재 상태 | 영향 |
|------|----------|------|
| Observation 저장/로드 | App에서 직접 구현 | visualbind가 오프라인으로 Observation을 읽으려면 직렬화 필요 |
| 비디오 배치 처리 | App에서 직접 루프 | 다수 비디오에서 hint를 수집할 때 편의성 |

이들은 visualbind 없이도 유용한 기능이므로, visualpath 자체의 로드맵에 포함시키는 것이 자연스럽습니다.
