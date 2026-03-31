# MomentBank — 고객 기억 시스템

> 최종 업데이트: 2026-03-27

## 핵심 정의

personmemory는 **member 단위의 장기 기억 저장소**이자 **person-conditioned distribution의 구현체**다.
단순한 이미지 DB가 아니라, 고객에 대한 이해를 축적하고 압축하는 시스템.

```
personmemory의 이중 역할:

  서비스 인프라 (product):
    reportrait에 참조 이미지 제공
    고객 프로필 관리 (앱 대표 프로필, 경기 대시보드)
    어트랙션 추억 저장

  person-conditioned distribution (research):
    per-member μ, Σ 축적
    표현 다양성 측정
    cross-person 비교 가능
    서비스 운영 = 연구 데이터 축적 (플라이휠)
```

```
personmemory의 질문: "이 사람을 얼마나 잘 이해하고 있는가?"
personmemory의 답변: 대표 이미지, 표정 분포, 빈 공간, 수집 전략
```

## portrait981 서비스의 세 기둥

```
personmemory  = 기억 (이 사람을 안다)
reportrait  = 창작 (이 사람을 표현한다)
gallery     = 전달 (이 사람에게 보여준다)
```

## 저장소 분리 원칙

```
data/
├── datasets/portrait-v1/     ← 학습용 작업 공간 (개발 단계)
│   ├── images/               # 레퍼런스 + 운영 이미지 (member 무관)
│   ├── labels.csv            # workflow 중심 라벨
│   ├── signals.parquet       # 43D signal 벡터
│   └── videos/               # 압축 비디오
│
├── personmemory/               ← 고객 기억 (서비스 단계)
│   └── members/
│       └── park_042/
│           ├── profile.json  # 프로필 (분포, gap, 이력)
│           ├── observed/     # 관찰 대표 이미지 (자체 복사본)
│           ├── embeddings.npz# face embedding history
│           └── history.json  # 생산 이력 + 레시피
│
└── gallery/                  ← 생산물 (고객 전달)
    └── park_042/
        ├── 2026-03-23_warm_portrait.jpg
        └── 2026-03-23_action_poster.jpg
```

| 저장소 | 목적 | member_id 필요? | 독립성 |
|--------|------|----------------|--------|
| dataset | XGBoost 학습, 도구 개발 | 불필요 | 독립 |
| personmemory | 고객 이해 축적 | **필수** | 독립 |
| gallery | 생성물 서빙 | 필수 | 독립 |

**dataset에는 있지만 personmemory에는 없는 것:**
- 인터넷 레퍼런스 이미지 (member 없음)
- member_id 미부여 테스트 비디오
- cut 프레임

**personmemory 노드 생성 조건:**
- member_id가 부여됨
- SHOOT된 프레임 (expression + pose 라벨 있음)
- member_id 없으면 → 노드 없음 (정상)

## 기억의 세 층위

```python
member.memory = {
    "episodic":  관찰된 장면 (대표 이미지),
    "semantic":  이 사람에 대한 이해 (분포, gap, signal profile),
    "creative":  생산 이력 (레시피 + gallery 참조 + 고객 반응),
}
```

### Episodic — 관찰 기억
```python
# expression×pose 버킷당 best K장
member.observed = {
    ("cheese", "front"):  [Frame(quality=0.92), Frame(quality=0.88)],
    ("hype", "front"):    [Frame(quality=0.78)],
}
```

### Semantic — 의미 기억
```python
member.profile = {
    "signal_mean": [43D],
    "signal_std":  [43D],
    "expression_distribution": {"cheese": 0.45, "hype": 0.20, ...},
    "pose_distribution": {"front": 0.60, "angle": 0.30, ...},
    "gaps": ["goofy", ("edge", "side")],
    "visit_count": 5,
}
```

### Creative — 생산 기억
```python
member.history = {
    "generations": [
        {
            "date": "2026-03-23",
            "style": "warm_portrait",
            "references": ["observed/cheese_front_001.jpg"],
            "workflow": "i2i_portrait_v2.json",
            "output": "gallery/park_042/2026-03-23_warm_portrait.jpg",
            "reaction": "purchased",
        },
    ]
}
```

생성물 이미지는 gallery에 저장, personmemory는 레시피와 반응만 기억.
→ "이 고객은 warm_portrait를 좋아했다" → 재방문 시 같은 스타일 우선 제안.

## 아키텍처 위치

```
중앙 시스템 → member_id ─┐
                          │
momentscan → workflow 분석 ┤
                          ├→ personmemory ←→ reportrait → gallery
annotator  → 수동 라벨   ┤       │
                          │       ├ episodic: 관찰 대표 이미지
face_detect → face_id ───┘       ├ semantic: 분포 + gap + signal profile
                                 └ creative: 생산 레시피 + 반응
```

### 다른 컴포넌트와의 관계

| 컴포넌트 | 역할 | personmemory와의 관계 |
|----------|------|-------------------|
| momentscan | per-frame 분석 (43D signal) | SHOOT 프레임 + signal → ingest |
| annotator | 수동 라벨링 + 리뷰 | 라벨 확정 후 → ingest |
| visualbind | signal → 판단 (XGBoost) | personmemory가 학습 데이터 제공 |
| reportrait | AI 초상화 생성 | personmemory에서 참조 조회, 결과는 gallery로 |
| gallery | 생성물 서빙 | personmemory가 gallery 위치 참조 |
| dataset | 학습용 작업 공간 | personmemory와 독립, 학습 데이터 소스 |

## 핵심 식별자

```
member_id   = 중앙 시스템 부여 (티켓/QR), ground truth
workflow_id = 1회 탑승 = 1 비디오
face_id     = face embedding (512D), 보조 식별
```

member_id가 authority. face_id는 보조:
- 같은 비디오 내 주탑승자 vs 동승자 구분
- member_id 없는 고객의 이전 방문 매칭 시도
- 동일인 검증 (member_id + face_id 교차 확인)

## 핵심 인터페이스

```python
class MomentBank:
    # --- 축적 ---
    def ingest(self, workflow_id: str, member_id: str,
               frames: list[Frame], labels: dict,
               face_embedding: np.ndarray | None = None):
        """workflow 결과를 member 기억에 축적.
        대표셋 업데이트, 분포 갱신, gap 재계산."""

    # --- 조회 ---
    def lookup(self, member_id: str,
               expression: str | None = None,
               pose: str | None = None,
               top_k: int = 3) -> list[Frame]:
        """member의 특정 조건에 맞는 베스트 프레임 반환.
        reportrait가 참조 이미지를 요청할 때."""

    # --- 프로필 ---
    def profile(self, member_id: str) -> MemberProfile:
        """member의 전체 프로필: 분포, gap, 방문 이력."""

    # --- 검증 ---
    def verify(self, member_id: str,
               face_embedding: np.ndarray) -> float:
        """face embedding으로 member 동일인 검증."""

    # --- 생산 기록 ---
    def record_generation(self, member_id: str,
                          style: str, references: list[str],
                          output_path: str, reaction: str | None = None):
        """reportrait 생산 결과를 creative memory에 기록."""

    # --- 수집 전략 ---
    def suggest(self, member_id: str) -> CollectionStrategy:
        """이 member에 대해 다음에 무엇을 수집해야 하는지."""

    # --- 압축 ---
    def compact(self, member_id: str):
        """기억 정리: 중복 제거, 대표셋 갱신, 오래된 데이터 압축."""
```

## 기억의 4 Level

### Level 1: Representative Selection (코어셋)
expression×pose 버킷당 best K장만 유지.
새 프레임 → 기존 worst보다 좋으면 교체.
**구현**: 지금 바로 가능. → reportrait 연결 즉시 가능.

### Level 2: Distribution Understanding
43D signal의 mean/std 추적 (Welford online stats).
Gap 인식: 부족한 expression×pose 조합 식별.
face_id 인덱스 (FAISS).
**구현**: Level 1 + 기존 face-baseline 활용.

### Level 3: Embedding Space Compression
이미지가 아닌 "이 사람이 어떻게 변하는지"를 기억.
Face State Embedding: z = [z_id | z_state | z_pose | z_quality].
새 프레임이 "새로운 정보인지" 판단 → 중복이면 버림.
**구현**: Face State Embedding 이후.

### Level 4: Active Collection
personmemory가 능동적으로 수집 전략을 조정.
Gap 기반 → momentscan에 전달 → 다음 탑승 시 우선 수집.
**구현**: VisualGrow 이후.

## 구현 로드맵

```
Phase A (단기): Level 1 — 대표셋 + 기본 프로필
  - ingest, lookup, profile
  - reportrait 연결 테스트

Phase B (중기): Level 2 — 분포 이해 + face_id
  - Welford stats, gap 인식
  - face embedding 인덱스
  - creative memory (생산 이력)

Phase C (장기): Level 3-4 — Embedding + Active Collection
  - Face State Embedding 통합
  - VisualGrow 연동
  - 능동적 수집 전략
```

## 단기 기억(momentscan) ↔ 장기 기억(personmemory)

```
momentscan (단기 기억 — 해마):
  한 영상에서 SHOOT 프레임 생산
  절대적 판단: f(frame) → quality + expression + pose
  느슨하게 수집 → 좋은 프레임을 놓치지 않는 것이 목표

personmemory.consolidate() (장기 기억 — 피질):
  여러 방문에 걸쳐 축적
  상대적 판단: g(signal, memory) → marginal_value
  엄격하게 정리 → 중복 줄이고 다양성 유지
```

momentscan이 생산한 signal을 재분석 없이 재사용.
새로운 분석기가 필요한 것이 아니라 **다른 함수**(기존 기억 대비 가치 판단).

→ person-conditioned distribution 상세: `research/person-conditioned-distribution.md`

## 서비스 경로: 즉시 + 축적

```
탑승 완료 시점:

  ┌─────────────────────────────────────────────┐
  │ 즉시 경로 (Fast Path)                        │
  │                                              │
  │ momentscan → best frames (이번 workflow)     │
  │   + personmemory.lookup(member_id)             │
  │   → 이전 기억이 있으면 맥락 보충              │
  │   → reportrait → 어트랙션 추억 AI portrait   │
  │   → 고객 앱 전송                              │
  └─────────────────────────────────────────────┘
                     │ (비동기)
  ┌─────────────────────────────────────────────┐
  │ 축적 경로 (Accumulation Path)                │
  │                                              │
  │ personmemory.ingest(                           │
  │   member_id, workflow_id,                    │
  │   shoot_frames + signals + signal_summary,   │
  │   face_embeddings,                           │
  │ )                                            │
  │   → 기억 노드 생성/갱신                       │
  │   → μ, Σ Welford merge                      │
  │   → 대표 프레임 갱신                           │
  │   → 고객 프로필 portrait 재생성 가능           │
  └─────────────────────────────────────────────┘
```

서비스하는 두 가지 컨텐츠:
- **어트랙션 추억** (per-workflow, 즉시): 방금 탑승한 순간의 AI portrait
- **고객 프로필** (per-member, 축적): 앱 대표 프로필, 경기 대시보드 디스플레이

## 기억 노드 구조

Phase 1에서는 expression × pose 이산 버킷을 노드 키로 사용.
Phase 2에서 signal distance 검증 후 연속 coverage로 전환.

```python
@dataclass
class MemoryNode:
    expression: str              # cheese, hype, ...
    pose: str                    # front, angle, side
    signal_mean: np.ndarray      # 43D, 이 노드의 signal 중심
    signal_std: np.ndarray       # 43D, 이 노드의 signal 분산
    n_observed: int              # 관찰 횟수
    best_frame_path: str         # 대표 이미지 경로
    best_confidence: float
    last_seen: str
    workflows: list[str]         # 관찰된 workflow 목록

@dataclass
class MemberMemory:
    member_id: str
    nodes: list[MemoryNode]           # 동적 기억 노드 (사람마다 다름)
    signal_mean: np.ndarray           # 43D 전체 분포 중심 (μ)
    signal_cov: np.ndarray            # 43×43 전체 공분산 (Σ)
    n_total_frames: int
    n_visits: int
    dominant_embedding: np.ndarray    # 512D 주 face embedding
    expression_dist: dict             # {"cheese": 0.45, ...}
    pose_dist: dict                   # {"front": 0.60, ...}
```

노드 수는 사람마다 동적:
- 표현 풍부한 사람: 4~8개 노드
- 표현 절제된 사람: 1~3개 노드

## 네 가지 Primitive — 프레임워크 없이 조합

personmemory가 하는 일을 정확히 지원하는 기존 프레임워크는 없음.
기존 vectorDB(ChromaDB, Pinecone 등)는 "retrieval 최적화"를 목적으로 설계되었고,
personmemory의 핵심인 **"새 관찰이 기존 분포에 얼마나 marginal value를 추가하는가"**를 다루지 않음.
이것은 ML 연구의 coreset selection / active learning 문제인데,
실시간 서비스 인프라로 구현한 사례가 없음.

```
personmemory = 네 가지 primitive의 오케스트레이션:

1. Entity Store      → member_id 기준 profile (JSON)
                        member별 기억 노드, 메타데이터, 생산 이력

2. Online Stats      → Welford (μ, Σ incremental update)
                        라이브러리 불필요, numpy만으로 구현
                        person-conditioned distribution의 핵심

3. ANN Index (identity) → FAISS (512D face embedding)
                           "이 사람 누구야?"
                           비디오 내 주 탑승자 추적, member 매칭

4. ANN Index (signal)   → FAISS (43D visualbind signal)
                           "이런 표정/포즈 상태의 프레임 찾아줘"
                           이미지 프롬프트로 signal 공간 검색
```

프레임워크를 쓰는 순간 그 프레임워크의 추상화 모델에 맞게 설계를 타협해야 함.
이 네 가지를 MomentBank 클래스가 직접 오케스트레이션하는 구조가 의미론적 정확성을 보존.

### 기존 프레임워크와의 비교

| 프레임워크 | 유사한 점 | 부족한 점 |
|-----------|----------|----------|
| Mem0 | entity 기준 memory 축적/검색 | LLM 대화용, signal 수치 연산 미지원 |
| Zep | entity memory + 시간 축적 | 텍스트 특화 |
| LanceDB | columnar vector, per-member 파티션 | μ/Σ update 직접 구현 필요 |
| ChromaDB | metadata 필터 + embedding 검색 | marginal value 개념 없음 |

## 다중 인덱스 검색 아키텍처

### FAISS 인덱스 세 가지 (Phase 2)

```
FAISS (identity)  512D InsightFace   "이 사람 누구야?"
FAISS (signal)    43D  visualbind    "이런 표정 상태의 프레임"
FAISS (semantic)  768D CLIP/DINOv2   "이 텍스트/이미지와 의미적으로 가까운 프레임"
```

각각이 다른 질문에 답함:

```python
# 1. Identity 검색 — "이 사람의 모든 프레임"
bank.search_by_identity(face_image)

# 2. Signal 검색 — "이런 AU/표정/포즈 상태의 프레임"
bank.search_by_signal(signal_vector)
# 또는 이미지 프롬프트: query_image → 14 models → 43D → FAISS
bank.search_by_image_signal(query_image)

# 3. Semantic 검색 — "이 설명에 맞는 프레임" (CLIP text-image joint space)
bank.search_by_text("gentle smile, looking at camera")

# 4. Visual similarity — "이 이미지와 비슷한 느낌" (DINOv2)
bank.search_by_visual(reference_image)

# 5. 조합 — identity 필터 + semantic 검색 + metadata 필터
bank.search(member_id="test_3", text="laughing", pose="front")
# → "test_3의 기억 중에서, 정면이면서, '웃고 있는' 프레임"
```

### CLIP 활용 — portrait.score와의 차이

이전에 제거한 portrait.score(CLIP 4축)는 미리 정의된 고정 축 점수였음:
```
제거한 것: CLIP → warm_smile: 0.7, cool_gaze: 0.3 (경직된 카테고리)
새 활용:   CLIP → 768D embedding 그대로 보존 → FAISS 인덱스
           → 자유로운 텍스트로 검색 ("laughing", "surprised", "calm" 등)
           → 열린 어휘, 카테고리 경계 없음
```

CLIP은 텍스트 쿼리용, DINOv2는 이미지 쿼리용 (시각적 유사도가 더 정확).

### reportrait 연동 시나리오

```
시나리오 1: 어트랙션 추억 (즉시)
  → 이번 workflow의 best frames (signal 기반 선택)

시나리오 2: 프로필 portrait (축적)
  → 가장 대표적 노드의 best frame (n_observed × confidence)

시나리오 3: 스타일 지정 생성
  → "이 사람의 웃는 정면 사진" → signal 검색 또는 text 검색

시나리오 4: 레퍼런스 이미지 기반
  → 원하는 표정의 참고 이미지 → CLIP/signal 공간에서 가장 가까운 프레임

시나리오 5: 부족한 표현 보충
  → coverage gap에서 가장 가까운 노드의 프레임 활용
  → "이 사람은 goofy가 없지만 hype가 비슷할 것"
```

## 구현 Phase

```
Phase 1 (현재, 구현 완료):
  Entity Store: JSON (PersonMemory 클래스)
  Online Stats: Welford (μ, Σ, per-node + global)
  Retrieval: 이산 카테고리 매칭 (expression × pose)
  Portrait quality score: lighting_ratio × brightness_std × conf × sharpness
  face.lighting analyzer: 47D signal (lighting_ratio, brightness_std, highlight/shadow ratio)
  대표 프레임: 버킷별 portrait quality best
  → momentscan run --ingest → personmemory 축적 확인

Phase 2 (다음):
  + FAISS identity index (512D face embedding)
  + FAISS signal index (47D, 이미지 프롬프트 검색)
  + 연속 signal 공간 기반 검색 (카테고리 → 연속)
  + marginal value 기반 프레임 선택 고도화
  + base-portrait 선정 → reportrait 연동

Phase 3 (중기):
  + FAISS semantic index (768D CLIP embedding)
  + IC-Light latent space → lighting descriptor (pixel 통계 대체)
  + 텍스트 프롬프트 검색
  + DINOv2 visual similarity 검색
  + reportrait 연동 (다중 쿼리 조합)

Phase 4 (장기):
  + Face State Embedding (통합 모델)
  + 단일 인덱스로 identity × state × semantic × lighting 통합
  + VisualGrow 연동 (자율 재학습)
  + Active collection (gap 기반 수집 전략)
```

## 통합 모델의 학습 인프라

personmemory는 서비스 인프라이면서 동시에 **통합 모델(Face State Embedding)의 학습 데이터를 자동 생산하는 인프라**.

### 통합 모델들의 한계

```
Florence-2, CLIP, DINO, TWLV-I 등:
  학습 데이터에 person-level 구조가 없음
  "이 이미지에 사람이 있다" (detection) ← 있음
  "이 이미지는 웃고 있다" (captioning) ← 있음
  "이것이 저것과 같은 사람의 다른 표정이다" ← 없음!
```

### personmemory가 제공하는 것: person-grounded training signal

```
통합 모델이 없는 것:            personmemory가 가진 것:
  person identity 연결           member_id (물리적 보장)
  같은 사람의 다양한 상태        per-member 다중 노드
  상태 변화의 연속성             43D signal trajectory
  개인별 표현 범위               per-member μ, Σ
  "다른 표정이지만 같은 사람"    member_id로 positive pair 보장
```

### 자동 생성되는 학습 데이터

```
1. Contrastive Learning Pairs:
   Positive: 같은 member_id, 다른 표정 → "같은 사람"
   Negative: 다른 member_id, 비슷한 표정 → "다른 사람"
   → ArcFace 학습 데이터보다 풍부 (표정 변화 포함)

2. Conditional Generation Targets:
   Input: member_id + "cheese|front"
   Target: 해당 노드의 대표 프레임
   → person-conditioned generation의 ground truth

3. Diversity Supervision:
   같은 member의 노드 간 거리 = 표현 다양성
   → "이 사람의 표현 범위"를 학습 가능

4. Temporal Grounding:
   workflow 내 signal trajectory
   → Video-LLM의 face event grounding 학습 데이터
```

### 실현 경로

```
현재:
  14 frozen models → 43D signals → personmemory (저장/관리)

미래:
  14 frozen models → 43D signals → personmemory (축적)
                                       ↓
                              person-grounded training data
                                       ↓
                              Face State Embedding 학습
                              (통합 모델, 14 models → 1 model)
                                       ↓
                              추론 시 14 models → 1 model 교체
```

Florence-2가 다양한 task annotation으로 학습되었듯이,
Face State Embedding은 personmemory의 person-grounded annotation으로 학습.
차이는 **수동 annotation 0건, 서비스 운영에서 자동 축적**.

→ 상세: `research/vision-face-state-embedding.md`

## Florence-2와의 비교 — Data Engine의 진화

### Florence-2의 구조

```
Florence-2 (Microsoft, 2024):
  Specialist models → 5.4B auto-annotations (FLD-5B "data engine")
  → Unified Seq2Seq backbone 학습
  → 1 model이 detection, captioning, segmentation, grounding 처리

핵심: "다양한 task의 annotation을 자동 생성하고,
       하나의 통합 모델을 학습하면
       개별 specialist보다 풍부한 representation"
```

### 구조적 유사성

```
Florence-2:                      portrait981:
  specialist models              14 frozen vpx models
  → auto-annotation (5.4B)       → 43D signals (매일 자동)
  → unified backbone 학습        → Face State Embedding (계획)
  → 1 model replaces all         → 1 model replaces 14
```

Florence-2가 검증한 것: specialist → auto-annotation → unified model **패턴이 작동한다**.
우리의 14 frozen models → 43D signals → 통합 모델 방향의 정당성.

### 핵심 차이: Image-centric vs Person-centric

```
Florence-2:
  Image-centric — "이 이미지를 이해해라"
  annotation 1회 생성 → 학습 → 배포 → 끝
  entity 개념 없음
  data engine이 학습 전에만 작동

portrait981:
  Person-centric — "이 사람을 이해해라"
  annotation 지속 축적 → 학습 → 재축적 → 재학습
  member_id 기반 entity memory
  data engine이 서비스와 함께 영구 작동 (플라이휠)
```

### Florence-2에서 참고할 점

1. **Data engine 패턴**: specialist → auto-annotation → unified model (실증됨)
2. **Task diversity**: 다양한 task를 함께 학습하면 representation 품질 향상
3. **Unified sequence**: 모든 task를 하나의 형식으로 통일 (Florence: text, 우리: 43D vector)

### Florence-2에 없는 것 = 우리의 고유 가치

| Florence-2에 없는 것 | portrait981이 가진 것 |
|---------------------|---------------------|
| Entity memory | personmemory (per-member μ, Σ) |
| Temporal accumulation | consolidate (방문 누적) |
| Diversity measurement | person-conditioned distribution |
| Conditional generation | "이 사람 기준" 참조 이미지 |
| Operational flywheel | 서비스 운영 = 데이터 축적 |
| Person-grounded pairs | member_id로 positive/negative pair 보장 |

Florence-2는 **우리 방향의 앞 절반을 검증** (specialist → unified model).
portrait981은 **뒷 절반을 추가** (entity memory + continuous accumulation + person-conditioned distribution).

## 데이터 흐름 원칙

### 두 경로의 분리

```
자동 축적 (서비스 운영):
  momentscan run --ingest member_id → personmemory.ingest()
  모델이 판단한 결과 그대로 축적 (signal + expression + pose + embedding)

수동 수정 (모델 품질 개선):
  annotator → datasets/labels.csv 수정
  → signal 재추출 → XGBoost 재학습
  → 더 나은 모델 → momentscan 재실행 → personmemory 재축적
```

personmemory에 수동 라벨을 직접 넣지 않음 — signal 없이 라벨만 있으면 분포(μ, Σ) 계산 불가.

### personmemory 편집 정책

personmemory는 시스템의 기억. 사람의 직접적인 편집은 지양.

```
✓ 자동 축적 (momentscan → ingest)
✓ 문제 프레임 제거 (심각한 오류만)
✗ 수동 라벨 편집 (expression/pose 직접 수정)
✗ 수동 프레임 추가 (signal 없는 프레임)
```

### 삭제 시 피드백 루프

```
사람이 personmemory에서 문제 프레임 제거
  → datasets/labels.csv에 해당 프레임 CUT으로 자동 반영
  → 다음 재학습 때 XGBoost가 "이런 프레임은 CUT" 학습
  → 이후 momentscan이 비슷한 프레임을 자동으로 걸러냄
```

사람의 1회 피드백 → 시스템의 영구 개선. visualgrow의 가장 단순한 형태.

### datasets vs personmemory

| | datasets | personmemory |
|---|---|---|
| 목적 | 모델을 개선 | 고객을 기억 |
| 입력 | 수동 라벨 | momentscan 자동 |
| 편집 | 자유 (annotator) | 삭제만 |
| 누적 | 수동 추가 | 서비스 운영으로 자동 |
| 출력 | XGBoost 모델 | 참조 이미지 + 분포 |

datasets가 좋아지면 → 모델이 좋아지고 → personmemory가 좋아짐.

## 핵심 원칙

1. **member_id가 기준** — face_id는 보조, 중앙 시스템이 authority
2. **member_id 없으면 노드 없음** — dataset과 분리, 서비스 데이터만
3. **이미지를 쌓지 않고 이해를 축적** — 대표셋 + 분포(μ, Σ)
4. **빈 공간을 안다** — "이 사람에 대해 모르는 것"을 인식 (coverage gap)
5. **세 층위의 기억** — episodic(관찰) + semantic(이해) + creative(생산)
6. **시스템의 기억** — 사람의 직접 편집 지양, 삭제 → datasets CUT 피드백
7. **독립 저장소** — dataset/personmemory/gallery 각각 독립
