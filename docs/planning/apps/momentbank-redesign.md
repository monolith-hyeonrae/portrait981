# Momentbank Redesign — 고객 기억 시스템

## 핵심 정의

momentbank는 **member 단위의 장기 기억 저장소**다.
단순한 이미지 DB가 아니라, 고객에 대한 이해를 축적하고 압축하는 시스템.

```
momentbank의 질문: "이 사람을 얼마나 잘 이해하고 있는가?"
momentbank의 답변: 대표 이미지, 표정 분포, 빈 공간, 수집 전략
```

## portrait981 서비스의 세 기둥

```
momentbank  = 기억 (이 사람을 안다)
reportrait  = 창작 (이 사람을 표현한다)
gallery     = 전달 (이 사람에게 보여준다)
```

## 저장소 분리 원칙

```
data/
├── datasets/portrait-v1/     ← 학습용 작업 공간 (개발 단계)
│   ├── images/               # 레퍼런스 + 운영 이미지 (member 무관)
│   ├── labels.csv            # workflow 중심 라벨
│   ├── signals.parquet       # 49D signal 벡터
│   └── videos/               # 압축 비디오
│
├── momentbank/               ← 고객 기억 (서비스 단계)
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
| momentbank | 고객 이해 축적 | **필수** | 독립 |
| gallery | 생성물 서빙 | 필수 | 독립 |

**dataset에는 있지만 momentbank에는 없는 것:**
- 인터넷 레퍼런스 이미지 (member 없음)
- member_id 미부여 테스트 비디오
- cut 프레임

**momentbank 노드 생성 조건:**
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
    "signal_mean": [49D],
    "signal_std":  [49D],
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

생성물 이미지는 gallery에 저장, momentbank는 레시피와 반응만 기억.
→ "이 고객은 warm_portrait를 좋아했다" → 재방문 시 같은 스타일 우선 제안.

## 아키텍처 위치

```
중앙 시스템 → member_id ─┐
                          │
momentscan → workflow 분석 ┤
                          ├→ momentbank ←→ reportrait → gallery
annotator  → 수동 라벨   ┤       │
                          │       ├ episodic: 관찰 대표 이미지
face_detect → face_id ───┘       ├ semantic: 분포 + gap + signal profile
                                 └ creative: 생산 레시피 + 반응
```

### 다른 컴포넌트와의 관계

| 컴포넌트 | 역할 | momentbank와의 관계 |
|----------|------|-------------------|
| momentscan | per-frame 분석 (49D signal) | SHOOT 프레임 + signal → ingest |
| annotator | 수동 라벨링 + 리뷰 | 라벨 확정 후 → ingest |
| visualbind | signal → 판단 (XGBoost) | momentbank가 학습 데이터 제공 |
| reportrait | AI 초상화 생성 | momentbank에서 참조 조회, 결과는 gallery로 |
| gallery | 생성물 서빙 | momentbank가 gallery 위치 참조 |
| dataset | 학습용 작업 공간 | momentbank와 독립, 학습 데이터 소스 |

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
49D signal의 mean/std 추적 (Welford online stats).
Gap 인식: 부족한 expression×pose 조합 식별.
face_id 인덱스 (FAISS).
**구현**: Level 1 + 기존 face-baseline 활용.

### Level 3: Embedding Space Compression
이미지가 아닌 "이 사람이 어떻게 변하는지"를 기억.
Face State Embedding: z = [z_id | z_state | z_pose | z_quality].
새 프레임이 "새로운 정보인지" 판단 → 중복이면 버림.
**구현**: Face State Embedding 이후.

### Level 4: Active Collection
momentbank가 능동적으로 수집 전략을 조정.
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

## 핵심 원칙

1. **member_id가 기준** — face_id는 보조, 중앙 시스템이 authority
2. **member_id 없으면 노드 없음** — dataset과 분리, 서비스 데이터만
3. **이미지를 쌓지 않고 이해를 축적** — 100장 → 15장 대표셋 + 분포
4. **빈 공간을 안다** — "이 사람에 대해 모르는 것"을 인식
5. **세 층위의 기억** — episodic(관찰) + semantic(이해) + creative(생산)
6. **생성물은 gallery** — momentbank는 레시피만, 이미지는 gallery
7. **독립 저장소** — dataset/momentbank/gallery 각각 독립
