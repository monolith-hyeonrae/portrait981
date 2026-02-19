# portrait_pivots — 포트레이트 생성 기반 Pivot 수집 시스템

> **identity_builder 리팩터링 설계**. 기존 수동 격자(binning)를
> reportrait 관점의 pivot 기반 수집으로 전환.
>
> 관련 문서: [identity_builder.md](identity_builder.md), [identity_memory.md](identity_memory.md), [reportrait.md](reportrait.md)

## 1. 동기

### 현재 구조의 문제

현재 identity_builder는 yaw(7) x pitch(3~5) x expression(4) = 84~140개 수동 격자로
프레임을 분류하고 버킷별 top-quality를 선택한다. E2E 검증 결과:

- **84개 버킷 중 17개만 사용 (20%)** — 나머지는 구조적 사각지대
- 상위 4개 버킷에 76% 프레임 집중 — 밀집 영역의 세밀한 구분 부족
- quality 순 선택 → **같은 영역에서 유사 프레임 중복 선택**
- 버킷이 분류만 하고 대표점(pivot)을 정의하지 않음

### reportrait 관점의 요구

momentscan의 수집 목적은 카메라 분포 기록이 아니라
**reportrait 디퓨전 모델의 가이드 얼굴 확보**이다.

- 특정 각도/표정의 가이드 얼굴을 요청할 수 있어야 함
- "정면 미소", "측면 프로필", "놀란 표정" 등 포트레이트 생성 조건이 기준
- 수집 시점에 이 조건들을 pivot으로 정의하고, 각 pivot의 최적 대표를 확보

## 2. 핵심 개념: Portrait Pivot

### 정의

**Portrait Pivot**: reportrait가 요청할 수 있는 가이드 얼굴 조건의 원형(prototype).
연속적인 포즈/표정 공간에서 사전 정의된 대표점.

```
Pivot = {
    name: "frontal-smile",       # reportrait가 참조하는 이름
    pose: (yaw=0, pitch=0),      # 포즈 공간의 좌표
    expression: "smile",         # 표정 조건
    r_accept: 15.0,              # 수용 반경 (degrees)
    priority: "high",            # 수집 우선순위
}
```

### 수동 격자와의 차이

| | 수동 격자 (현재) | Portrait Pivot |
|---|---|---|
| 정의 기준 | 각도 균등 분할 | 포트레이트 생성 요구 |
| 경계 | 직사각형 hard edge | 원형 soft radius |
| 대표 선택 | quality 최고 | pivot 중심에 가장 가까운 (quality 최소 기준 후) |
| 빈 영역 의미 | 카메라 사각지대 | 해당 가이드 미확보 |
| 커버리지 | 버킷 채움 수 | pivot별 대표 확보 여부 |
| 중복 방지 | 사후 필터 3종 | pivot당 1대표 → 구조적 중복 불가 |

### SOM과의 관계

Pivot 시스템은 Self-Organizing Map (SOM)의 변형이다:

| | SOM | Portrait Pivot | MemoryBank |
|---|---|---|---|
| 노드 위치 | 학습 (자기조직화) | 수동 정의 (도메인 지식) | 동적 생성 |
| 토폴로지 | 격자 이웃 연결 | 독립 | 독립 |
| 대표 선택 | BMU (Best Matching Unit) | 가장 가까운 프레임 | EMA merge |
| 가장 유사한 패턴 | — | Growing Neural Gas | Growing Neural Gas |

Pivot 위치를 수동 정의하는 이유: **해석 가능성**. "왜 이 프레임을 골랐는가"에
"frontal-smile pivot에 가장 가까우므로"라고 설명할 수 있다.

## 3. Pivot 카탈로그

두 축의 직교 조합: **Pose Pivot x Expression Pivot**.

### Pose Pivots

포트레이트 생성에서 의미 있는 얼굴 각도 조건.

| Name | (yaw, pitch) | 용도 | 비고 |
|------|-------------|------|------|
| `frontal` | (0, 0) | ID 전달 기본, InstantID/PuLID 최적 | 최고 우선순위 |
| `three-quarter` | (30, 0) | 클래식 포트레이트 앵글 | 자연스러운 인물 사진 |
| `side-profile` | (60, 0) | 실루엣, 프로필 샷 | identity 약하지만 분위기 |
| `looking-up` | (15, 20) | 역동적 앵글 | 981파크 롤러코스터 특유 |
| `three-quarter-up` | (30, 15) | 올려보는 3/4 뷰 | 981파크 카메라 위치 반영 |

> 좌우 대칭: yaw 음수도 동일 pivot name에 매핑 (카메라 위치에 따라 좌/우가 뒤집힘).
> `side-profile`은 yaw=60이든 yaw=-60이든 같은 pivot.

### Expression Pivots

포트레이트 생성에서 의미 있는 표정 조건.

| Name | 판정 기준 | 용도 |
|------|----------|------|
| `neutral` | smile < 0.3, mouth < 0.3, eyes open | 깨끗한 베이스라인, 합성 최안정 |
| `smile` | smile_intensity >= 0.4 | 기본 포트레이트, 가장 많이 요청 |
| `excited` | mouth_open >= 0.5 AND smile >= 0.3 | 리액션 하이라이트, 981파크 핵심 |
| `surprised` | mouth_open >= 0.5 AND smile < 0.3 | 놀란 순간, 재미있는 포트레이트 |

> `eyes_closed`, `confident` 등은 데이터 충분 시 추후 추가.

### 조합 (Pose x Expression)

```
5 pose x 4 expression = 20 pivots
```

모든 조합이 동등하지 않다. Priority:

| Priority | 조합 | 이유 |
|----------|------|------|
| **must** | frontal-neutral, frontal-smile | ID 전달 필수 |
| **high** | three-quarter-neutral, three-quarter-smile | 클래식 포트레이트 |
| **high** | any-excited, any-surprised | 981파크 핵심 순간 |
| **medium** | looking-up-*, three-quarter-up-* | 981파크 특유 앵글 |
| **low** | side-profile-* | 확보되면 좋지만 필수 아님 |

## 4. 수용 반경 (r_accept)

### Pose 거리

유클리드 거리 (yaw, pitch 공간):

```python
d_pose = sqrt((yaw - pivot.yaw)^2 + (pitch - pivot.pitch)^2)
```

좌우 대칭 적용:

```python
d_pose = min(
    sqrt((yaw - pivot.yaw)^2 + (pitch - pivot.pitch)^2),
    sqrt((-yaw - pivot.yaw)^2 + (pitch - pivot.pitch)^2),
)
```

### Expression 매칭

범주형 → 정확 매칭 (같은 expression이면 거리 0, 다르면 ∞).
향후 soft distance 확장 가능 (smile과 excited의 거리 < smile과 neutral의 거리).

### 통합 할당 로직

```python
for frame in all_frames:
    # 1. Expression 매칭
    expr_pivot = classify_expression(frame)

    # 2. Pose 거리 계산 → 최근접 pivot
    best_pose_pivot = argmin(d_pose(frame, p) for p in pose_pivots)
    best_dist = d_pose(frame, best_pose_pivot)

    # 3. 수용 판정
    if best_dist <= r_accept:
        assign(frame, best_pose_pivot, expr_pivot)
    else:
        mark_unassigned(frame)
```

### r_accept 기본값

E2E 검증 결과: r_accept=15°에서 698프레임 중 697개 할당 (99.9%).

```
r_accept = 15°  (기본값)
```

## 5. 대표 선택

### 원칙

**각 pivot에 1장의 대표 (representative)**. pivot 중심에 가장 가까운 프레임을 선택하되,
최소 품질 기준을 통과해야 한다.

```python
def select_representative(pivot, assigned_frames):
    # 품질 게이트
    candidates = [f for f in assigned_frames if f.quality >= q_min]
    if not candidates:
        return None  # 이 pivot 미확보

    # pivot 중심에 가장 가까운 프레임
    return min(candidates, key=lambda f: d_pose(f, pivot))
```

### Anchor vs Representative

기존 identity_builder의 anchor/coverage/challenge 3세트 구분을 pivot 시스템으로 재정의:

| 기존 개념 | pivot 시스템 대응 |
|----------|-----------------|
| Anchor (정면 고품질 top-k) | `frontal-*` pivot의 대표 (must priority) |
| Coverage (버킷별 best) | 각 pivot의 대표 1장 |
| Challenge (극단 + 안정) | `side-profile-*` 등 low priority pivot의 대표 |

3세트 구분이 pivot priority로 자연스럽게 대체된다.

## 6. MemoryBank 연동

### 저장

pivot별 대표가 선택되면 MemoryBank에 등록:

```python
for pivot, representative in pivot_representatives.items():
    bank.update(
        e_id=representative.e_id,
        quality=representative.quality_score,
        meta={"pivot": pivot.name},  # pivot name을 메타로 저장
        image_path=representative.crop_path,
    )
```

### 검색 (reportrait → MemoryBank)

MemoryBank의 `select_refs()` 쿼리가 pivot 이름을 받는다:

```python
# 현재: bucket 기반 coverage_score
refs = bank.select_refs(RefQuery(target_buckets={"yaw": "[-5,5]"}))

# 변경: pivot 이름 기반
refs = bank.select_refs(RefQuery(target_pivot="frontal-smile"))
```

## 7. Query Resolver (reportrait → pivot)

reportrait가 가이드 얼굴을 요청하는 세 가지 경로:

### ① 직접 지정

```python
refs = bank.query_pivot("frontal-smile")
# → frontal-smile pivot의 대표 이미지 반환
```

### ② 텍스트 맥락

```python
# "자신만만한 미소로 정면을 보는 포트레이트"
pivot = resolve_text("자신만만한 미소로 정면을 보는")
# → text → (pose=frontal, expression=smile) → "frontal-smile"
refs = bank.query_pivot(pivot)
```

텍스트 → pivot 매핑은 키워드 룩업 또는 LLM 기반:

| 텍스트 키워드 | Pose Pivot | Expression Pivot |
|-------------|-----------|-----------------|
| 정면, 앞을 보는 | frontal | — |
| 측면, 프로필 | side-profile | — |
| 미소, 웃는 | — | smile |
| 놀란, 당황한 | — | surprised |
| 신나는, 즐거운 | — | excited |
| 올려보는, 위를 보는 | looking-up | — |

### ③ 참조 이미지

```python
# 샘플 포트레이트 이미지에서 pose/expression 추출
ref_img_features = extract_features("sample_portrait.jpg")
# → yaw=25, pitch=5, smile=0.6
# → nearest pivot: three-quarter-smile
pivot = resolve_image(ref_img_features)
refs = bank.query_pivot(pivot)
```

## 8. 시각화

### Pivot Coverage Map

identity report.html에 추가할 시각화:

```
  pitch
   20° ·  ·  ·  ·  ◉  ·  ·      ◉ = pivot (확보됨, 대표 있음)
       ·  ·  ·  ·  ·  ◉  ·      ○ = pivot (미확보)
    0° ·  ·  ◉  ·  ·  ◉  ○      · = 프레임
       ·  ·  ·  ·  ·  ·  ·      --- = r_accept 경계
  -10° ·  ·  ·  ·  ·  ·  ·
      -60°     0°     60° yaw
```

Plotly scatter:
- 전체 프레임을 회색 점으로 표시
- Pivot 위치를 큰 마커로 표시 (확보=채움, 미확보=빈 원)
- r_accept 원을 반투명으로 표시
- 선택된 대표 프레임을 강조 마커로 표시
- expression별 컬러/탭 분리

### Pivot Status Table

```
| Pivot              | Status | Dist  | Quality | Frame |
|--------------------|--------|-------|---------|-------|
| frontal-neutral    | ● OK   | 0.7°  | 0.72    | #203  |
| frontal-smile      | ● OK   | 2.3°  | 0.68    | #415  |
| frontal-excited    | ● OK   | 0.4°  | 0.65    | #278  |
| three-quarter-*    | ● OK   | 0.6°  | 0.70    | #243  |
| side-profile-*     | ○ MISS | —     | —       | —     |
```

### 카메라 설치 진단 (부산물)

pivot coverage 데이터는 카메라 설치 품질의 간접 지표가 된다:

| 측정 | 의미 |
|------|------|
| `frontal-*` pivot 점유율 | 정면 확보율 (포트레이트 핵심 KPI) |
| 전체 pivot 확보율 | 설치 품질 (높을수록 좋은 카메라 위치) |
| 프레임 밀집 pivot | 카메라의 실효 각도 |
| 분포 엔트로피 | 다양성 (높을수록 좋은 설치) |

여러 카메라의 pivot coverage를 비교하면 설치 표준화 기준을 도출할 수 있다.

## 9. 구현 계획

### Phase A: Pivot 정의 + 할당

- [ ] `PosePivot`, `ExpressionPivot` 타입 정의 (`identity/pivots.py`)
- [ ] `POSE_PIVOTS`, `EXPRESSION_PIVOTS` 카탈로그 상수
- [ ] `assign_pivot(frame) -> (PosePivot, ExpressionPivot, distance)` 할당 함수
- [ ] 기존 `classify_frame()` 대체

### Phase B: 대표 선택 리팩터링

- [ ] `select_representative(pivot, frames)` — pivot 중심 최근접 선택
- [ ] `IdentityBuilder._select_anchors/coverage/challenge` → `_select_by_pivots()` 통합
- [ ] anchor/coverage/challenge 분류를 pivot priority로 대체
- [ ] 기존 사후 필터 (temporal, visual similarity) 제거 — pivot당 1대표로 불필요

### Phase C: MemoryBank 연동

- [ ] `NodeMeta.pivot_name` 필드 추가 (기존 histogram 대체 또는 병행)
- [ ] `bank.query_pivot(pivot_name)` API 추가
- [ ] `RefQuery.target_pivot` 필드 추가

### Phase D: 시각화

- [ ] Pivot Coverage Map (Plotly scatter + acceptance circles)
- [ ] Pivot Status Table
- [ ] identity report.html에 통합

### Phase E: Query Resolver (reportrait)

- [ ] 텍스트 → pivot 매핑 (키워드 룩업)
- [ ] 이미지 → pivot 매핑 (feature extraction → nearest pivot)
- [ ] reportrait comfy_bridge 연동
