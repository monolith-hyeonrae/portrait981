# Momentbank Redesign — 고객 기억 시스템

## 핵심 정의

momentbank는 **member 단위의 장기 기억 저장소**다.
단순한 이미지 DB가 아니라, 고객에 대한 이해를 축적하고 압축하는 시스템.

```
momentbank의 질문: "이 사람을 얼마나 잘 이해하고 있는가?"
momentbank의 답변: 대표 이미지, 표정 분포, 빈 공간, 수집 전략
```

## 아키텍처 위치

```
중앙 시스템 → member_id ─┐
                          │
momentscan → workflow 분석 ┤
                          ├→ momentbank ←→ reportrait
annotator  → 수동 라벨   ┤       │
                          │       ├ member별 기억 저장소
face_detect → face_id ───┘       ├ face embedding 인덱스
                                 ├ 다양성 분포 이해
                                 └ 수집 gap 인식
```

### 다른 컴포넌트와의 관계

| 컴포넌트 | 역할 | momentbank와의 관계 |
|----------|------|-------------------|
| momentscan | per-frame 분석 (49D signal) | 분석 결과를 momentbank에 축적 |
| annotator | 수동 라벨링 + 리뷰 | 라벨 결과를 momentbank에 축적 |
| visualbind | signal → 판단 (XGBoost) | momentbank가 학습 데이터 제공 |
| reportrait | AI 초상화 생성 | momentbank에서 참조 이미지 조회 |

## 핵심 식별자

```
member_id   = 중앙 시스템 부여 (티켓/QR), ground truth, 교차 방문 연결
workflow_id = 1회 탑승 = 1 비디오, member의 한 에피소드
face_id     = face embedding (InsightFace 512D), 비전 기반 보조 식별
```

member_id가 기준. face_id는 보조:
- 같은 비디오 내 주탑승자 vs 동승자 구분
- member_id 없이 방문한 고객의 이전 방문 매칭
- 동일인 검증 (member_id + face_id 교차 확인)

## 기억 구조: 4 Level

### Level 1: Representative Selection (코어셋)

expression×pose 버킷당 best K장만 유지.

```python
member.representatives = {
    ("cheese", "front"):  [Frame(quality=0.92), Frame(quality=0.88)],
    ("cheese", "angle"):  [Frame(quality=0.85)],
    ("hype", "front"):    [Frame(quality=0.78)],
}
# 새 프레임 → 기존 worst보다 좋으면 교체
# 100장 방문 기록 → 15장 대표셋으로 압축
```

**구현**: 지금 바로 가능. quality score = XGBoost confidence 또는 face quality signal.

### Level 2: Distribution Understanding

이 사람의 signal 공간 분포를 기억.

```python
member.profile = {
    "signal_mean": [49D],           # 이 사람의 평균 상태
    "signal_std":  [49D],           # 변동 범위
    "expression_distribution": {
        "cheese": 0.45,             # 이 사람은 주로 웃음
        "hype": 0.20,
        "chill": 0.15,
        "edge": 0.05,
    },
    "pose_distribution": {
        "front": 0.60,
        "angle": 0.30,
        "side": 0.10,
    },
    "visit_count": 5,
    "total_frames": 47,
    "first_visit": "2026-03-15",
    "last_visit": "2026-03-23",
}
```

**빈 공간(gap) 인식**:
```python
member.gaps = {
    "missing_expressions": ["goofy", "occluded"],  # 한번도 수집 안 됨
    "weak_poses": ["side"],                         # 수집됐지만 quality 낮음
    "underrepresented": [("edge", "angle")],        # 이 조합 부족
}
```

**구현**: Level 1 + Welford online statistics (이미 momentscan-face-baseline에 있음)

### Level 3: Embedding Space Compression

이미지가 아닌 "이 사람이 어떻게 변하는지"를 기억.

```python
member.appearance = {
    "face_centroid": [512D],           # 평균 얼굴 embedding
    "expression_directions": {
        "smile":   [512D delta],       # 웃을 때 embedding 변화 방향
        "excited": [512D delta],       # 흥분할 때
    },
    "state_embedding": [z_state],      # Face State Embedding 학습 후
}
```

새 프레임이 들어오면:
- centroid 업데이트 (online mean)
- 기존 direction과의 거리로 "새로운 정보인지" 판단
- 새로운 방향이면 저장, 중복이면 버림

**구현**: Face State Embedding (Level 2 이후)

### Level 4: Active Collection

momentbank가 능동적으로 수집 전략을 조정.

```python
strategy = momentbank.suggest_collection(member_id="042")
→ {
    "priority": "side pose",           # 가장 부족한 것
    "avoid": "cheese_front",           # 이미 충분
    "threshold_adjustment": {
        "hype": 0.3,                   # hype 임계값 낮춰서 더 수집
    }
}
# → momentscan에 전달 → 다음 탑승 시 반영
```

**구현**: VisualGrow Level 1 이후

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
        """member의 특정 조건에 맞는 베스트 프레임 반환."""

    # --- 프로필 ---
    def profile(self, member_id: str) -> MemberProfile:
        """member의 전체 프로필: 분포, gap, 방문 이력."""

    # --- 검증 ---
    def verify(self, member_id: str,
               face_embedding: np.ndarray) -> float:
        """face embedding으로 member 동일인 검증. similarity 반환."""

    # --- 수집 전략 ---
    def suggest(self, member_id: str) -> CollectionStrategy:
        """이 member에 대해 다음에 무엇을 수집해야 하는지."""

    # --- 압축 ---
    def compact(self, member_id: str):
        """기억 정리: 중복 제거, 대표셋 갱신, 오래된 데이터 압축."""
```

## 저장 구조

```
momentbank/
├── members/
│   ├── park_042/
│   │   ├── profile.json          # 프로필 (분포, gap, 이력)
│   │   ├── representatives/      # 대표 이미지 (expression×pose별 best K)
│   │   │   ├── cheese_front_001.jpg
│   │   │   └── hype_front_001.jpg
│   │   ├── embeddings.npz        # face embedding history
│   │   └── signal_stats.json     # 49D signal mean/std (Welford)
│   └── park_087/
│       └── ...
├── face_index/                   # face embedding → member_id 매핑
│   └── faiss_index.bin           # FAISS or annoy index
└── bank.db                       # SQLite: member 메타, workflow 이력
```

## 구현 로드맵

```
Phase A (지금): Level 1 — 대표셋 선별 + 기본 프로필
  - ingest: workflow 결과에서 expression×pose별 best K 유지
  - lookup: 조건에 맞는 대표 프레임 반환
  - profile: 분포 + gap 계산
  → reportrait 연결 즉시 가능

Phase B: Level 2 — 분포 이해 + face_id 연결
  - Welford online stats로 signal 분포 추적
  - face embedding 인덱스 (FAISS)
  - verify: 동일인 검증
  - compact: 중복/저품질 제거

Phase C: Level 3 — Embedding Space (Face State Embedding 이후)
  - appearance space 학습
  - "새로운 정보인가?" 판단
  - 이미지 없이 member를 표현하는 compact representation

Phase D: Level 4 — Active Collection (VisualGrow 이후)
  - suggest: 수집 전략 생성
  - momentscan과 연동하여 실시간 수집 조정
```

## 기존 코드와의 관계

현재 `apps/momentbank/`:
- `ingest.py`: lookup_frames, ingest_collection — **유지, 확장**
- `bank.py`: MemoryBank — **Level 1 기반으로 리팩토링**
- `identity.py`: face embedding 관리 — **Level 2에서 확장**

현재 `apps/annotator/`:
- labels.csv, signals.parquet — momentbank의 **입력 소스**
- 수동 라벨은 highest confidence source로 취급

현재 `scripts/extract_signals.py`:
- momentbank.ingest 시 자동 signal 추출 호출 가능
- 또는 momentscan 결과에서 직접 수신

## 핵심 원칙

1. **member_id가 기준** — face_id는 보조, 중앙 시스템이 authority
2. **이미지를 쌓지 않고 이해를 축적** — 100장 → 15장 대표셋 + 분포 프로필
3. **빈 공간을 안다** — "이 사람에 대해 모르는 것"을 인식
4. **점진적 압축** — 방문할수록 더 적은 데이터로 더 정확한 이해
5. **외부 요청에 즉시 응답** — reportrait가 물으면 바로 최적 참조 반환
