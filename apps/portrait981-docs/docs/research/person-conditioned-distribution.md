# Person-Conditioned Distribution — 개인별 표현 분포 기반 다양성 모델

> "같은 미소도 사람마다 다르다."
> Foundation model은 일관성(consistency)을 학습하지만,
> portrait981은 일관성과 다양성(diversity)을 동시에 표현해야 한다.

---

## 핵심 아이디어

기존 foundation model의 학습 목표:

```
ArcFace:  같은 사람 → 가까운 embedding  (identity consistency)
CLIP:     같은 의미 → 가까운 embedding  (semantic consistency)
DINO:     같은 장면 → 가까운 embedding  (visual consistency)
V-JEPA:   같은 동작 → 가까운 embedding  (motion consistency)
```

모두 **같은 것을 같다고 말하는** 모델. consistency 최적화.

portrait981이 풀어야 하는 문제:

```
같은 사람(identity 고정) 안에서,
  - 어떤 표정이 다른 표정과 "다른가"
  - 어떤 포즈가 다른 포즈와 "다른가"
  - 수집된 프레임들이 전체적으로 "충분한 다양성을 갖추었는가"
  - 새 프레임이 기존 set에 "얼마만큼의 새로운 가치를 추가하는가"
```

이것은 **conditional diversity** — identity를 고정한 상태에서의 다양성 측정.

---

## Person-Conditioned Distribution

### 표현 방식

기존 모델:
```
f(image) → point in embedding space
```

제안:
```
f(person, observation_history) → distribution in signal space

  μ_person  = 이 사람의 기본 상태 (resting state)
  Σ_person  = 이 사람의 표현 범위 (expressive range)
  coverage  = 수집된 프레임이 이 분포를 커버하는 정도
```

### 왜 person-conditioned인가

같은 "smile" 카테고리라도:

```
A씨 (표현 풍부):
  neutral → smile: AU6=0.2→0.9, AU12=0.3→0.95  (큰 변화)
  Σ_A의 smile 방향 분산이 큼
  → d_mahal=2는 "보통 수준의 표현"

B씨 (표현 절제):
  neutral → smile: AU6=0.1→0.3, AU12=0.2→0.4  (작은 변화)
  Σ_B의 smile 방향 분산이 작음
  → d_mahal=2는 "이 사람 기준 극적인 표현"
```

절대적 카테고리 모델은 A의 smile과 B의 smile을 같은 "smile"로 처리.
Person-conditioned distribution에서는 **각각이 자기 분포 안에서 상대적 의미를 가짐**.

### Mahalanobis Distance의 의미

```
새 프레임의 signal vector: x
이 사람의 분포: N(μ, Σ)

상대적 편차: z = x - μ
Mahalanobis distance: d = √(z^T Σ^{-1} z)

d의 의미:
  d ≈ 0: 이 사람의 평소 상태
  d ≈ 1: 이 사람 기준 1σ 수준의 변화
  d ≈ 2: 이 사람 기준 2σ — 상당히 다른 표현
  d ≈ 3: 이 사람 기준 극단적 변화 (드문 표현)
```

**사람마다 동일한 d가 다른 절대적 크기의 변화를 의미.** 표현이 풍부한 사람의 2σ는 절제된 사람의 2σ보다 signal 공간에서 훨씬 넓은 범위.

---

## Momentbank as Person Memory System

member_id별 signal statistics가 축적되면, personmemory는 단순 프레임 저장소를 넘어 **AI의 사람에 대한 기억 체계**가 된다.

### 저장 구조

```
personmemory/
  member_id: "test_3"
  ├── distribution:
  │     μ: [43D mean signal vector]
  │     Σ: [49×49 covariance matrix]
  │     N: 1247  (관찰 횟수)
  │     principal_axes: [PCA of Σ]
  │       PC1: smile↔neutral 방향 (이 사람의 최대 변화 축)
  │       PC2: pose variation 방향
  │       PC3: quality variation 방향
  ├── coverage:
  │     collected_frames: [selected frames with signals]
  │     hull_volume: 0.73  (signal 공간 convex hull 커버리지)
  │     bucket_fill: {cheese|front: ✓, chill|angle: ✗, ...}
  └── meta:
        first_seen: 2026-01-15
        last_seen: 2026-03-25
        total_visits: 12
```

### Face VectorDB로서의 활용

person-conditioned distribution이 축적되면, personmemory는 강력한 face vectorDB:

```
Query: "이 사람의 표현 범위를 연속적으로 정렬"
  → PC1 (smile↔neutral) 축으로 수집된 프레임 정렬
  → 가장 중립적인 표정에서 가장 밝은 표정까지 연속 스펙트럼

Query: "이 표현(smile)에 대한 여러 사람의 방식들"
  → 각 member의 μ→smile 방향 벡터 비교
  → "A씨는 주로 눈으로 웃고, B씨는 입으로 웃고, C씨는 전체 얼굴이 변한다"
  → 같은 감정의 개인별 표현 방식 차이를 signal 수준에서 설명

Query: "이 사람과 비슷한 표현 패턴을 가진 다른 사람"
  → Σ_person1과 Σ_person2의 구조적 유사도 비교
  → 표현 방식 자체가 비슷한 사람끼리 그룹핑

Query: "이 사람에게 아직 수집되지 않은 표현 영역"
  → coverage의 빈 영역 식별
  → 다음 방문 시 우선 수집할 방향 제안
```

---

## 연속 다양성 레벨

```
Level 0 (현재): 이산 버킷
  expression × pose grid coverage
  select_frames()의 greedy bucket fill
  한계: "cheese|front"와 "hype|front"이 얼마나 다른지 표현 불가

Level 1: 연속 coverage (engineering)
  43D signal 공간에서의 convex hull 또는 k-DPP
  새 프레임의 marginal diversity gain = hull 확장 정도
  한계: 절대적 signal 공간, person-specific이 아님

Level 2: Person-conditioned coverage (이 문서의 핵심)
  per-member μ, Σ 축적
  Mahalanobis distance 기반 상대적 다양성
  PCA 축으로 개인별 표현 스펙트럼 정의
  coverage = 분포의 각 방향에 대한 커버리지

Level 3: Adaptive diversity (visualgrow 영역)
  데이터가 쌓일수록 μ, Σ가 정교해짐
  방문 횟수에 따라 prior → posterior 업데이트
  새 member의 cold start에 population prior 활용
  "이 사람은 12번 방문, 표현 범위의 73%가 커버됨"
```

---

## Cold Start: Bayesian Prior

새 member (첫 방문)에서는 개인 분포를 추정할 데이터가 부족.

```
Prior (population level):
  μ_0 = 전체 member 평균 signal
  Σ_0 = 전체 member의 pooled covariance
  ν_0 = prior strength (pseudo-count)

첫 방문 (N=50 frames):
  μ_test3 ≈ (ν_0 × μ_0 + N × x̄) / (ν_0 + N)
  → population prior에 가까움, 개인 특성 약함

12번째 방문 (N=1200 frames):
  μ_test3 ≈ x̄  (개인 데이터가 지배)
  Σ_test3 ≈ sample covariance (개인 특성 명확)
```

Normal-Inverse-Wishart prior → Bayesian update:
- 적은 데이터: "평균적인 사람"에 가까운 분포 사용
- 많은 데이터: 완전히 개인화된 분포
- 중간: 자연스러운 보간 (shrinkage)

이것은 evidence accumulation과 정확히 같은 구조:
> "관찰이 쌓일수록 belief가 prior에서 data-driven으로 이동"

---

## XGBoost와의 관계

현재 TreeStrategy(XGBoost)는 절대적 분류기:
```
signals → P(cheese), P(chill), P(edge), P(hype)
```

Person-conditioned 방향에서는 두 단계로 분리:

```
Step 1: Person-relative transform (per-member)
  z = Σ^{-1/2} (signals - μ_person)
  "이 사람 기준으로 얼마나, 어느 방향으로 다른가"

Step 2: Classification/scoring (shared or per-person)
  Option A: z → shared XGBoost → expression category
            공통 분류기, 입력만 정규화
  Option B: z → person-specific threshold
            개인별 적응 (visualgrow Level 3)

Step 1이 person-specific, Step 2는 공유 가능.
```

### 기존 시스템과의 호환성

현재 visualbind 파이프라인에 자연스럽게 삽입 가능:

```
현재:
  observations → bind_observations() → signals → TreeStrategy → category

확장:
  observations → bind_observations() → signals
    → person_transform(signals, μ_person, Σ_person) → z (relative signals)
    → TreeStrategy(z) → category (person-normalized input)
```

TreeStrategy 자체를 바꾸지 않고, **입력을 person-normalize**하는 것만으로 첫 단계 구현 가능.

---

## 응용 시나리오

### 1. 연속 표현 정렬 (Expression Spectrum)

```
member "test_3"의 PC1 (smile↔neutral) 축으로 정렬:

  [neutral] -------- [slight smile] -------- [broad smile]
  frame#42           frame#187              frame#891
     ↑                   ↑                      ↑
  PC1=-1.2σ          PC1=0.0σ              PC1=+2.1σ
```

이산 카테고리("chill", "cheese")가 아닌, **연속 스펙트럼**으로 표현의 변화를 시각화.

### 2. Cross-Person Expression Comparison

```
"smile" 방향 벡터 비교:

  A씨: smile = [AU6↑↑, AU12↑↑, AU25↑]    → 눈+입 동시 (Duchenne)
  B씨: smile = [AU6↑,  AU12↑↑, AU25=]    → 입 위주, 눈 덜 변화
  C씨: smile = [AU6↑↑, AU12↑,  AU25↑↑]   → 눈 위주, 입 크게 벌림

같은 감정, 다른 표현 방식 → signal 수준에서 설명 가능
```

### 3. Coverage Gap 식별 + 수집 가이드

```
member "test_7" (3회 방문):
  coverage: PC1 73%, PC2 45%, PC3 21%
  gap: "측면 포즈에서의 다양한 표정" 영역이 비어있음
  → 다음 방문 시 이 영역 프레임 우선 수집
```

### 4. 개인 표현 범위 (Expressive Range) 프로파일링

```
member "test_3": Σ trace = 12.7 (표현 풍부)
member "test_5": Σ trace = 3.2  (표현 절제)

→ 생성(reportrait) 시 다른 전략 적용:
  test_3: 다양한 스타일 시도 가능
  test_5: subtle variation에 집중
```

### 5. 유사 표현 패턴 그룹핑

```
Σ의 구조적 유사도로 member 클러스터링:
  Group A: 눈으로 표현하는 사람들 (PC1 ≈ AU6 방향)
  Group B: 입으로 표현하는 사람들 (PC1 ≈ AU12 방향)
  Group C: 전체 얼굴이 변하는 사람들 (PC1이 다축 혼합)

→ 그룹별 생성 템플릿, 포즈 가이드 최적화
```

---

## 단기 기억(momentscan) ↔ 장기 기억(personmemory)

### 뇌의 기억 시스템과의 대응

```
해마 (단기 기억):
  깨어있는 동안 경험을 일단 다 기록
  정제 안 됨, 중복 많음, 용량 제한

수면 중 피질 (장기 기억):
  시냅스 가지치기 — 불필요한 연결 제거
  기억 통합 — 구조적 이해로 변환
  → 다음 날 더 효율적으로 판단 가능

momentscan (단기 기억 — 해마):
  한 영상에서 수십~수백 개 SHOOT 프레임 생산
  절대적 판단: "이 프레임 자체가 좋은가?"
  다른 프레임을 모름 → 좋으면 다 통과

personmemory (장기 기억 — 피질):
  여러 방문에 걸쳐 축적
  상대적 판단: "기존 기억 대비 이 프레임이 필요한가?"
  정제되고, 구조화되고, 검색 가능
```

### 핵심 통찰: 같은 분석기가 아니라 다른 함수

```
momentscan:   f(frame) → quality + expression + pose          (절대 판단)
personmemory:   g(frame_signal, memory) → marginal_value        (상대 판단)
```

momentscan은 프레임 하나를 보고 "좋다/나쁘다"를 판단.
personmemory는 기존 기억(μ, Σ, coverage)을 보고 "이미 가진 것 대비 새로운 가치가 있는가"를 판단.

**g에 새로운 분석기가 필요하지 않다.** momentscan이 이미 계산한 signal vector를 재사용하면 된다.

```
momentscan → [SHOOT frames + 43D signals] → personmemory.ingest()
                                              ├─ μ, Σ 업데이트 (signal 재사용, 재분석 불필요)
                                              ├─ marginal_value = mahalanobis(signal, μ, Σ)
                                              ├─ pruning: value가 낮은 기존 프레임 교체
                                              └─ coverage gap 업데이트
```

### 상호 보완적 밸런스

momentscan이 너무 완벽하게 선별하려 하면 두 가지 문제가 발생:
1. 좋은 프레임을 놓칠 수 있음 (false negative)
2. "좋은"의 기준이 person-specific이어야 하는데, momentscan은 member 기억이 없음

반대로 personmemory consolidate 없이 momentscan에만 의존하면:
1. SHOOT 300개가 그대로 쌓임 (중복)
2. 방문 횟수가 늘수록 파일만 증가, 구조적 이해 없음

**역할 분리가 답:**

```
momentscan (느슨하게 수집, recall 최대화):
  gate pass + SHOOT → 전부 personmemory로 전달
  "좋은 프레임을 놓치지 않는 것"이 목표
  select_frames()는 가벼운 중복 제거 수준으로 단순화

personmemory.consolidate() (엄격하게 정리, precision 최대화):
  per-member μ, Σ 기반 marginal value 계산
  value가 낮은 기존 프레임 교체 (시냅스 가지치기)
  coverage gap 식별 → 다음 방문 수집 가이드
  "중복을 줄이고 다양성을 유지하는 것"이 목표
```

이 밸런스가 뇌의 해마(일단 다 기억) ↔ 수면 중 피질(정리해서 장기 저장) 관계와 같다.
해마가 너무 엄격하면 중요한 기억을 놓치고, 피질이 정리를 안 하면 기억이 과부하.

### consolidate의 타이밍

```
실시간 (momentscan 실행 중):
  → 단기 기억 생산만. consolidate 하지 않음.
  → 뇌도 깨어있을 때는 수집에 집중.

비실시간 (방문 종료 후, 배치):
  → personmemory.consolidate(member_id)
  → 시냅스 가지치기 + 기억 통합 + coverage 업데이트
  → 뇌도 잠잘 때 정리.

이 consolidate가 visualgrow의 첫 번째 구현체가 될 수 있음:
  — 데이터가 쌓일수록 기억이 성장하는 것이므로.
```

### consolidate 구체 알고리즘

```
personmemory.consolidate(member_id):

  1. μ, Σ 업데이트
     현재 축적된 모든 signal vectors → online Welford update
     방문 횟수에 따라 prior → posterior 자연 전환

  2. Marginal value 계산 (모든 프레임)
     for each frame in member's collection:
       value = mahalanobis_distance(frame.signal, μ, Σ)
     → 분포 중심에서 먼 프레임 = 높은 가치 (희소한 표현)
     → 분포 중심에 가까운 프레임 = 낮은 가치 (흔한 표현)

  3. 시냅스 가지치기 (pruning)
     같은 expression×pose 버킷 내에서:
       - 너무 가까운 프레임끼리 (signal distance < ε) → 품질 높은 것만 유지
       - 버킷당 최대 K개 유지 (K = f(방문 횟수))

  4. Coverage gap 업데이트
     PCA of Σ → 주요 변화 축 식별
     각 축 방향으로 수집된 프레임의 커버리지 계산
     빈 영역 = 다음 방문에서 우선 수집할 표현

  5. 메타데이터 업데이트
     total_frames, unique_expressions, coverage_ratio
     expressive_range = trace(Σ)  (이 사람의 표현 풍부도)
     last_consolidated = timestamp
```

---

## Cross-Model Agreement — 오탐을 이용한 신뢰도 판단

### 단일 모델의 한계

14개 frozen model은 각각 고유한 실패 모드를 가진다.
예: 완전한 측면 얼굴에서:

```
6DRepNet (head pose):  yaw=28° → "약간 측면" (오추정, 실제는 75°+)
BiSeNet (face parse):  seg_face=0.0 → "얼굴 영역 없음" (정직한 실패)
LibreFace (AU):        AU 합계=0.05 → "근육 활동 없음" (정직한 실패)
InsightFace (detect):  confidence=0.81 → "얼굴 있음" (측면도 잘 잡음)
```

단일 모델에 의존하면 오탐을 감지할 수 없다.
6DRepNet만 보면 yaw=28°는 정상이다.

### 핵심 통찰: 오탐 시점이 모델마다 다르다

```
모델 A가 실패할 때 → 모델 B, C는 정상일 수 있음
모델 B가 실패할 때 → 모델 A, C는 정상일 수 있음

→ 다수의 모델이 동시에 같은 방향으로 실패할 확률은 낮음
→ 모델 간 불일치(disagreement) 자체가 "무언가 이상하다"는 신호
```

이것은 crowds consensus의 핵심 원리와 동일:
> "14개 frozen model = domain-shifted된 noisy observer.
>  한 observer가 실패해도 다른 observer들이 교차 검증."

### 실제 동작: 3단 gate의 cross-validation

```
일관성 높음 (신뢰 가능):
  head_pose: yaw=30°    →  "약간 측면"
  seg_face:  0.12       →  "얼굴 보임"
  AU 합계:   0.4        →  "근육 활동 있음"
  → 3개 모두 "정상 얼굴" → 판단 신뢰

일관성 낮음 (신뢰 불가):
  head_pose: yaw=28°    →  "약간 측면" (?)
  seg_face:  0.00       →  "얼굴 안 보임" (!)
  AU 합계:   0.05       →  "근육 활동 없음" (!)
  → 1개만 정상, 2개가 이상 → head pose 신뢰 불가 → gate fail
```

**모델 하나가 틀리는 것이 문제가 아니라, 틀린다는 것을 감지하는 것이 핵심.**
감지는 다른 모델들의 출력이 서로 일치하지 않을 때 가능.

### Heuristic → Learned Agreement

현재 gate는 이 cross-validation을 개별 threshold로 구현 (heuristic):
```
if seg_face < 0.01: fail     ← BiSeNet이 "얼굴 없음"
if au_sum < 0.05: fail       ← LibreFace가 "근육 없음"
if |yaw| > 55: fail          ← 6DRepNet이 "극단적 각도"
```

장기적으로는 모델 간 일관성 판단 자체를 학습하는 것이 visualbind의 방향:
```
Level 0 (현재): 개별 threshold (heuristic gate)
Level 1: 모델 간 불일치 feature 추가 (|pose_yaw - pose_from_seg_ratio| 등)
Level 2: XGBoost가 모델 간 일관성 패턴을 학습 (gate를 흡수)
Level 3: Vision Student가 raw image에서 직접 신뢰도 판단
```

이것이 why-visualbind.md에서 정의한 crowds consensus가 gate 레벨에서 실현되는 과정.

---

## ArcFace Embedding 실험 — identity와 state의 얽힘

### 실험 (2026-03-26, test_3 영상)

```
Embedding 수:  전체=324  gate_pass=286  SHOOT=11

Centroid 코사인 유사도:
  all vs gate_pass: 0.998  → 거의 동일
  all vs SHOOT:     0.848  → 상당히 다름!
  gate vs SHOOT:    0.829  → 상당히 다름!

Intra-group 분산 (자기 centroid까지 거리):
  all:       0.200 ± 0.061  → 넓게 분포
  gate_pass: 0.189 ± 0.056  → 약간 좁아짐
  SHOOT:     0.095 ± 0.076  → 매우 타이트

전체 프레임의 대표성:
  all centroid (324개):  0.200
  SHOOT centroid (11개): 0.322  → 전체를 대변 못함
```

### 해석

SHOOT의 타이트한 분산(0.095)은 "같은 사람이라서"가 아니라
**"비슷한 표정(cheese+hype) × 비슷한 시간대(#253~300)"이라서** 타이트한 것.

ArcFace 512D에 identity와 expression state가 얽혀있기 때문에:
- 표정이 크게 달라지면 같은 사람이어도 embedding 거리가 벌어짐
- SHOOT(미적으로 뛰어난 특정 표정)의 centroid는 이 사람의 전체 상태를 대변하지 못함

이것은 `vision-face-state-embedding.md`에서 정의한 문제와 정확히 일치:
```
ArcFace: identity와 state를 분리하지 못함
  → "다른 표정"을 "다른 사람 방향"으로 해석
  → z_id와 z_state가 얽혀있는 512D
```

### member_id의 역할

identity를 embedding으로 풀 필요 없음 — member_id가 물리적 앵커:

```
member_id:
  "이 비디오의 주 탑승자는 처음부터 끝까지 같은 사람"
  → ArcFace가 표정 변화로 거리를 벌려도 identity는 확정

member_id + face embedding 조합:
  → 비디오 내 오검출/다른 사람 필터링 (승강장 근처 등)
  → 주 탑승자의 dominant embedding cluster 추적
  → 동승자(passenger)와의 분리
```

face embedding의 역할은 identity 판정이 아니라 **비디오 내 주 탑승자 추적**.
member_id가 cross-session identity를 보장하고,
face embedding은 single-session 내 face tracking을 보조.

### 이전 실험과의 모순 해소

이전 실험에서는 ArcFace가 표정 변화에 강인하다고 판단:
```
이전: 개별 프레임 간 비교
  frame_A(smile) vs frame_B(neutral) → cosine distance 아주 작음
  → "표정이 바뀌어도 거의 안 변하네"
```

이번 실험에서는 centroid가 상당히 다르다고 관찰:
```
이번: centroid 간 비교
  SHOOT centroid(11) vs 전체 centroid(324) → cosine sim 0.848
  → "꽤 다르네"
```

**모순이 아니라 스케일의 차이:**
```
개별 프레임: cos_dist ≈ 0.02 (아주 작음) → "강인하다"
축적:        0.02씩 324번 → centroid가 이동
             SHOOT 11개는 특정 방향(smile/hype)에 몰림
             전체 324개는 모든 방향으로 분산
             → centroid 간 거리 발생
```

한 걸음은 작지만 324걸음이면 꽤 먼 곳에 도달.
ArcFace가 표정 변화에 "강인"한 것은 맞지만 **완전히 불변은 아님**.
미세한 drift가 존재하고, 분포 수준에서 축적되면 관찰 가능.

**43D signal space가 더 적합한 이유:**
- ArcFace: 표정 drift가 암묵적 (512D 어딘가에 숨어있음, 해독 어려움)
- 43D signals: AU/emotion/pose가 명시적으로 분리 → "어디서 얼마나 변했는지" 바로 읽힘

### person-conditioned distribution과의 연결

```
기존 접근 (ArcFace 의존):
  identity: ArcFace embedding distance → 표정 변화에 민감, 부정확
  diversity: 측정 불가 (identity와 얽혀있으므로)

우리 접근 (member_id + 43D signal):
  identity: member_id (물리적 보장, 사업 구조)
  diversity: 43D signal space에서 per-member μ, Σ
  → ArcFace 한계를 우회
```

이 실험이 확인한 것:
1. ArcFace는 표정 변화에 민감 → identity와 state 분리 실패
2. SHOOT은 특정 상태에 편향 → 전체 분포를 대변 못함
3. member_id + 43D signal이 person-conditioned distribution의 올바른 기반

---

## TWLV-I 논문과의 연결

TWLV-I (Twelve Labs, 2024)가 제시한 교훈:

| TWLV-I 관점 | portrait981 적용 |
|-------------|-----------------|
| Appearance + Motion 이중 축 | Expression(appearance) + Pose(spatial) + Motion(temporal) |
| Distillation + Masked modeling 결합 | 14 frozen teacher + self-supervised signal prediction |
| Linear/Attentive/KNN 다층 평가 | XGBoost/Catalog/Heuristic 전략 비교 |
| Directional motion distinguishability | Expression onset/apex/offset temporal signal |
| Multi-Clip Embedding | Per-frame signal + temporal delta (on_frame에서) |

핵심 차이: TWLV-I는 **범용 video 표현**을 목표로 하지만,
portrait981은 **person-specific face state 표현**을 목표로 함.
TWLV-I가 "모든 동영상의 appearance+motion"이라면,
portrait981은 "특정 인물의 표현 다양성".

---

## 기존 시스템 기반

이미 존재하는 컴포넌트와의 관계:

| 기존 모듈 | 현재 역할 | 확장 방향 |
|-----------|----------|----------|
| `face.baseline` (Welford) | per-track online mean/var | → per-member persistent distribution |
| `personmemory` | member_id별 프레임 저장 | → member별 signal distribution 축적 |
| `select_frames()` | 이산 bucket coverage | → 연속 Mahalanobis coverage |
| `visualbind` 43D signals | 프레임별 절대 signal | → person-relative signal |
| `TreeStrategy` | 절대적 카테고리 분류 | → person-normalized input 분류 |
| `visualgrow` | XGBoost 재학습 | → per-member distribution update |
| `Face State Embedding` | 통합 face embedding 비전 | → person-conditioned subspace |

---

## 구현 우선순위

### Phase 1: Signal Statistics 축적 (engineering, 지금 가능)

```
personmemory에 member별 signal statistics 저장:
  - μ (43D mean), Σ (49×49 covariance), N (count)
  - momentscan v2의 on_frame()에서 signals 수집 시 동시 업데이트
  - Welford online algorithm (이미 face.baseline에 구현됨) 확장
```

### Phase 2: Person-Relative Scoring (engineering)

```
select_frames()에서 연속 diversity 사용:
  - Mahalanobis distance 기반 프레임 가치 산출
  - Convex hull 또는 k-DPP 기반 coverage 최적화
  - 이산 버킷 → 연속 coverage 전환
```

### Phase 3: Person-Normalized Classification (research)

```
TreeStrategy 입력을 person-normalize:
  z = Σ^{-1/2} (signals - μ_person)
  검증: person-normalized XGBoost vs absolute XGBoost
  cold start: NIW prior from population
```

### Phase 4: Expressive Range Profiling (research)

```
Σ의 PCA → 개인별 표현 축 식별
Coverage gap 분석 → 수집 가이드
Cross-person comparison → 그룹핑
```

### Phase 5: Face VectorDB (product)

```
personmemory → queryable face vector database:
  - "이 사람의 smile spectrum" → PC1 정렬
  - "smile이 비슷한 다른 사람" → inter-person Σ 비교
  - "아직 수집되지 않은 영역" → coverage gap
```

---

## 데이터셋 문제와 visual*의 해법

### 왜 이 문제를 남들이 못 푸는가

Person-conditioned diversity 모델을 학습하려면 이론적으로 필요한 라벨:

```
  - 인물 identity (누구인가)
  - 표정 카테고리 + 강도 (어떤 표현인가, 얼마나 강한가)
  - 포즈 (어떤 각도인가)
  - 품질 (선명한가, 조명은 적절한가)
  - 다양성 판단 (이 프레임이 기존 set에 새로운 가치를 추가하는가)

  × 수백~수천 명의 인물
  × 인물당 충분한 다양성 (수십~수백 프레임)
  × 일관된 기준으로 라벨링
```

이것을 사람이 라벨링하는 것은 사실상 불가능.
표정의 "강도"나 "다양성 가치"는 annotator 간 합의도 어렵고, 규모도 안 됨.
**기존 연구들이 이 문제를 안 푸는 이유 — 데이터셋을 만들 방법이 없으니 문제 자체를 정의하지 않는 것.**

### 막혀있는 길 vs 열려있는 길

```
기존 접근 (막혀있는 길):
  사람이 라벨링 → 데이터셋 → 모델 학습
  문제: 라벨링 불가능 (규모, 기준, 비용)

visual* 접근 (우리의 길):
  영상 → visualpath (14 frozen models) → 43D signals    (자동)
       → visualbind (crowds consensus) → pseudo-labels  (자동)
       → personmemory (member_id별 축적) → per-person μ, Σ (자동)
       → visualgrow (재학습) → 점진적 개선               (자동)

데이터셋 구축 비용 = 0.
서비스가 운영되는 것 자체가 데이터셋을 만드는 과정.
```

### visual*이 각 라벨을 자동 생산하는 방법

```
identity     ← member_id (사업 운영에서 자연 발생)
               + 비디오 상수 (같은 영상 = 같은 사람, 물리적 보장)
               → 수동 라벨 불필요, ArcFace보다 강인한 identity supervision

expression   ← face.au (12D) + face.expression (8D)
               → visualbind crowds consensus로 통합
               → TreeStrategy가 카테고리 + 확률 분포 출력

pose         ← head.pose (yaw/pitch/roll) + body.pose
               → TreeStrategy가 분류

quality      ← face.quality (mask-based blur/exposure)
               + face.parse (segmentation ratios)
               → HeuristicStrategy가 gate

다양성 가치   ← person-conditioned distribution에서 자동 산출
               μ, Σ가 축적되면 marginal gain = Mahalanobis 기반 자동 계산
```

### Student 학습 시 supervision 구조

Face State Embedding의 Student 모델이 학습할 때:

```
입력:   Raw face image
출력:   person-relative embedding

Supervision (전부 자동):
  14 frozen specialists → 43D signals (per-frame)
  member_id → identity grouping (per-video)
  축적된 μ, Σ → person-relative signal z = Σ^{-1/2}(x - μ)

  → Student는 "이 사람 기준 어떤 상태인가"를 raw image에서 직접 학습
  → 추론 시 14 frozen models 불필요, Student 하나로 판단
```

### 사업 구조 = 연구 인프라

```
portrait981 서비스 운영:
  매일 N건 비디오 × M명 고객 → 자동 축적

  방문 1회차: μ, Σ 초기화 (population prior 기반)
  방문 3회차: 개인 특성 윤곽 형성
  방문 10회차: 안정적 person distribution
  방문 30회차: 연구 수준의 개인별 표현 모델

  → 서비스가 돌수록 데이터셋이 풍부해짐
  → 풍부한 데이터셋 → 더 나은 모델 → 더 나은 서비스
  → 플라이휠 (flywheel)
```

### 데이터 획득의 근본적 난점 — 왜 이런 데이터가 세상에 없는가

person-conditioned diversity 모델에 필요한 데이터의 핵심 조건:
**같은 사람이 자연스럽게 여러 표정을 짓는 상황을 반복적으로 관찰할 수 있는 환경.**

기존 데이터 획득 방법은 전부 이 조건을 충족하지 못함:

```
연구실 촬영:
  "웃어주세요", "화난 표정 해주세요" → 의도된 연출
  Duchenne smile vs social smile 구분 불가
  실제 감정이 아닌 연기 → person distribution이 왜곡됨
  1회 방문, 반복 데이터 없음

공개 데이터셋 (AffectNet, FER, BP4D):
  인물당 프레임 수 적음 (대부분 1~수 장)
  자연스러운 표정 전환 과정 없음 (스냅샷)
  같은 사람의 반복 방문 데이터 없음
  → per-person distribution 추정에 필요한 N이 절대적으로 부족

CCTV/거리 촬영:
  무작위 → 같은 사람을 다시 만날 수 없음
  한 순간의 한 표정만 포착
  프라이버시 이슈

영상 통화/회의 녹화:
  표정 범위가 좁음 (대화 맥락에 제한)
  주로 neutral + mild smile 범위
  강한 감정 표현 거의 없음
```

### 테마파크 어트랙션 — 고유한 데이터 환경

어트랙션 탑승 1~2분 동안 자연스럽게 발생하는 표정 전환:

```
출발 전:    긴장/기대    (neutral → anticipation)
탑승 중:    놀람/흥분/공포 (surprise, excitement, fear)
하이라이트: 최고조 표정   (peak expression)
종료 후:    안도/웃음     (relief, joy)

→ 한 번의 탑승에서 자연스러운 표정 전환 2~3가지
→ 의도된 연출이 아닌 진짜 감정 반응
→ 자율적이므로 사람마다 반응 패턴이 다름
```

여러 어트랙션을 보유한 테마파크 환경의 구조적 이점:

```
어트랙션 A (롤러코스터): 흥분/공포 → 환희
어트랙션 B (워터라이드):  기대 → 놀람 → 웃음
어트랙션 C (다크라이드):  호기심 → 긴장 → 안도

같은 사람, 다른 맥락, 다른 표정:
  → 어트랙션이 추가될수록 관찰 가능한 표정 다양성 증가
  → 같은 사람이 여러 어트랙션을 타면 다양한 감정 맥락에서의 반응 수집
  → 재방문할수록 개인 분포가 정교해짐

데이터 획득 조건 (전부 자연스럽게 충족):
  ✓ 동일 인물 반복 관찰  — member_id, 연간 패스
  ✓ 자연스러운 감정      — 연기가 아닌 실제 반응
  ✓ 다양한 감정 맥락     — 어트랙션별 다른 자극
  ✓ 표정 전환 과정 포착  — 1~2분 연속 촬영
  ✓ 통제된 카메라 환경   — 고정 위치, 일정한 화각
  ✓ 고객 동의 기반       — 촬영 고지 + 서비스 동의
  ✓ 확장 가능           — 어트랙션 추가 = 데이터 채널 추가
```

### 타 회사/연구팀과의 비교

```
얼굴 분석 회사 (Face++, Naver Clova 등):
  → 정적 이미지 기반, 인물당 1장
  → 같은 사람의 여러 표정? 데이터 없음

감정 인식 스타트업:
  → 연구실 환경 또는 영상 통화 데이터
  → 표정 범위 제한적 (neutral/smile 위주)
  → 자연스러운 강한 감정 데이터 획득 불가

자율주행 회사:
  → 운전자 모니터링, 졸음/부주의 위주
  → 표정 다양성이 목표가 아님

우리 (981파크):
  → 같은 고객의 여러 어트랙션 탑승 = 다양한 감정 맥락
  → 1~2분 연속 = 표정 전환 과정 포착
  → 재방문 = 시간에 걸친 분포 축적
  → 어트랙션 추가 = 새로운 감정 자극원 추가
  → 사업 확장이 곧 데이터 다양성 확장
```

이 조건을 모두 갖춘 환경은 세상에 거의 없음.
**테마파크 어트랙션은 person-conditioned face diversity 연구를 위한 자연 실험실.**

### 왜 남들이 못 하는가 (종합)

| 필요 조건 | 일반 연구팀 | portrait981 |
|-----------|-----------|-------------|
| 대규모 per-person 영상 | 공개 데이터셋 없음 | 서비스 운영에서 자연 발생 |
| 자연스러운 다양한 표정 | 연구실 연기 or 제한적 범위 | 어트랙션 자극에 의한 진짜 감정 |
| 1인당 여러 감정 맥락 | 데이터 획득 불가 | 어트랙션별 다른 자극 |
| 표정 전환 과정 (temporal) | 스냅샷 데이터 | 1~2분 연속 촬영 |
| 일관된 member_id | 수동 라벨링 필요 | 사업 구조에서 자동 제공 |
| 반복 방문 데이터 | 수집 불가 (프라이버시) | 고객 동의 기반 축적 |
| 다축 face signal | 개별 모델 각각 평가 | visual* 14개 frozen model 통합 |
| 다양성의 정의 | 합의 없음 | person-conditioned distribution으로 수학적 정의 |
| 자동 라벨링 | 수동 의존 | crowds consensus pseudo-label |
| 확장성 | 고정 (추가 수집 비용 큼) | 어트랙션 추가 = 데이터 채널 추가 |

**세 가지 벽이 동시에 존재:**
1. **데이터 벽**: 같은 사람의 자연스러운 다양한 표정 데이터가 없음
2. **라벨 벽**: 있다 해도 annotation이 불가능
3. **정의 벽**: 데이터도 라벨도 없으니 문제 자체를 정의하지 않음

**portrait981은 세 벽을 모두 넘을 수 있음:**
1. 테마파크 어트랙션이 데이터를 자연 생산
2. visual* frozen specialist crowds가 라벨을 자동 생산
3. person-conditioned distribution이 다양성을 수학적으로 정의

### 학술적 기여 (데이터셋 관점 추가)

```
Contribution:
  1. Person-conditioned face diversity representation
     — identity 고정 상태에서의 다양성을 연속 공간에서 표현

  2. Annotation-free dataset construction via frozen specialist crowds
     — 14개 frozen model의 crowds consensus + 서비스 자연 발생 member_id
     — 수동 라벨 0건으로 person-conditioned diversity 학습 데이터 생산

  3. Operational flywheel
     — 서비스 운영 → 데이터 축적 → 모델 개선 → 서비스 개선의 선순환
     — 시간이 지날수록 per-person distribution이 정교해지는 자기강화 구조

  4. visual* 생태계가 제공하는 범용 프레임워크
     — visualpath (signal production) + visualbind (signal interpretation)
       + visualgrow (adaptation) 조합으로 데이터셋 자동 구축
     — portrait981 이외의 도메인에도 동일 패턴 적용 가능
```

---

## 프랜차이즈 시너지와 데이터 네트워크 효과

### 기존 사업과의 관계

981파크는 테마파크 프랜차이즈화를 준비 중.
지점이 늘어나면 person-conditioned distribution의 가치가 **초선형**으로 증가.

```
기존 매출 (티켓 판매):
  지점 1개: 매출 X
  지점 3개: 매출 ~3X
  지점 10개: 매출 ~10X
  → 선형 성장 (지점 수 × 고객 수)

데이터 자산 가치:
  지점 1개: N명 × M개 어트랙션
  지점 3개: 3N명 × 다양한 어트랙션 구성 + 교차 방문
  지점 10개: population prior 정교화 + 지역/문화 다양성
  → 초선형 성장 (네트워크 효과 + 복리 축적)
```

### 지점 증가의 데이터 시너지

```
단일 지점 (현재):
  고객 N명 × 어트랙션 M개
  per-person: 방문당 M개 감정 맥락

다중 지점:
  1. 인물 풀 확대
     → population prior가 정교해짐
     → 새 고객의 cold start가 더 정확해짐

  2. 어트랙션 종류 다양화
     → 지점별 다른 어트랙션 구성 가능
     → 같은 사람이 A지점 롤러코스터 + B지점 워터라이드 탑승
     → 관찰 가능한 감정 맥락이 어트랙션 종류만큼 확장

  3. 동일 인물의 교차 방문
     → 다른 날, 다른 장소, 다른 컨디션
     → 환경 변화에 강인한 person distribution
     → "이 사람은 어디서든 이렇게 웃는다"

  4. 지역/문화별 표현 차이
     → 한국/일본/동남아 지점 등
     → 문화권별 표현 패턴 차이를 데이터로 관찰
     → 범용 모델로의 확장 근거
```

### 데이터 플라이휠

```
┌─────────────────────────────────────────────────┐
│  테마파크 운영 (기존 사업)                        │
│    → 고객 어트랙션 탑승 → 영상 자동 수집          │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│  visual* 파이프라인 (자동)                       │
│    → 14 frozen models → 43D signals             │
│    → crowds consensus → pseudo-labels            │
│    → per-member μ, Σ 축적                        │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│  모델 개선                                       │
│    → person-conditioned distribution 정교화      │
│    → Face State Embedding Student 학습           │
│    → 더 정확한 표정/다양성 판단                    │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│  서비스 가치 향상                                 │
│    → 더 나은 portrait 선택 → 고객 만족도 ↑       │
│    → 개인화된 포트폴리오 → 재방문/재구매 ↑        │
│    → AI 초상화 품질 향상 → 부가 매출 ↑            │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│  프랜차이즈 매력 증가                             │
│    → 더 나은 서비스 → 가맹 수요 ↑                │
│    → 지점 증가 → 데이터 증가 → 모델 개선 (반복)   │
└─────────────────────────────────────────────────┘
```

### 사업적 가치 차원

| 차원 | 기존 (티켓 매출) | person-conditioned distribution |
|------|-----------------|-------------------------------|
| 성장 곡선 | 선형 (지점 × 고객) | 초선형 (네트워크 효과) |
| 자산 축적 | 없음 (매출 = 일회성) | 복리 (데이터 + 모델이 시간과 함께 성장) |
| 경쟁 해자 | 약함 (위치/시설) | 강함 (데이터 + 모델, 복제 불가) |
| 확장 비용 | 높음 (시설 투자) | 한계비용 0 (기존 인프라 활용) |
| 고객 가치 | 단발성 경험 | 누적 포트폴리오 (방문할수록 풍부) |

### 데이터 자산으로서의 경쟁 해자

```
경쟁자가 따라하려면:
  1. 테마파크를 운영해야 함 (시설 투자)
  2. 어트랙션에 카메라를 설치해야 함 (하드웨어)
  3. 고객 동의를 받아야 함 (서비스 설계)
  4. member_id 체계를 갖춰야 함 (시스템)
  5. visual* 수준의 분석 파이프라인을 구축해야 함 (기술)
  6. 수년간 데이터를 축적해야 함 (시간)

→ 1~4는 돈으로 해결 가능하지만
→ 5~6은 시간과 노하우의 문제
→ 먼저 시작한 쪽이 데이터 축적에서 되돌릴 수 없는 격차를 가짐
```

### 왜 빅테크도 쉽게 따라할 수 없는가

이 문제는 기술력으로 우회할 수 없는 **사업 구조의 문제**:

```
구글 (기술 최강):
  ✓ 최고 모델 (ViT, Gemini), 막대한 컴퓨팅, YouTube 수십억 영상
  ✗ YouTube에 "같은 사람의 반복적 자연 표정 + identity 보장" 없음
  ✗ member_id를 붙이려면 face recognition → 프라이버시 문제
  ✗ 붙인다 해도 카메라를 의식한 표정 (자연스러운 감정 아님)

  → 데이터의 양은 ∞이지만, 필요한 구조가 없음
  → 기술력으로 데이터 구조를 만들 수 없음 (테마파크를 운영해야 함)

디즈니 (인프라 최강):
  ✓ 전 세계 12개 파크, MagicBand (고객 ID 추적), PhotoPass (사진 촬영)
  ✗ PhotoPass = "사진 판매" 사업으로 고정 (찍고 → 팔고 → 끝)
  ✗ 얼굴 signal 추출 파이프라인 없음
  ✗ per-person distribution 개념 자체가 없음
  ✗ AI 기반 portrait 생성 시도하지 않음

  → 인프라는 있지만 관점이 없음
  → "사진을 파는 것"과 "사람을 이해하는 것"은 완전히 다른 방향
```

핵심 통찰: **데이터의 양이 아니라 구조가 결정적이고, 그 구조는 사업 형태에서 나옴.**

```
구글:    기술 ✓  인프라 ✗  관점 ✗  → 따라할 수 없음
디즈니:  기술 △  인프라 ✓  관점 ✗  → 할 수 있지만 하지 않음
981파크: 기술 ✓  인프라 ✓  관점 ✓  → 유일하게 실행 중
```

portrait981의 근본적 차별점은 인프라도 기술도 아닌,
**"사진을 찍는 행위"를 "사람을 이해하는 과정"으로 재정의한 관점**:

```
디즈니 PhotoPass:
  카메라 → 사진 → 판매
  데이터의 끝 = 상품 전달 시점

portrait981:
  카메라 → visual*(14 models) → 43D signals → per-person distribution → 축적
  데이터의 시작 = 촬영 시점
  매 방문이 이 사람에 대한 이해를 깊게 함
```

이 관점의 차이가 기술적 방향을 결정하고,
visual* 생태계 전체가 이 관점 위에 설계됨.

오프라인에서 사람을 상대로 하는 비즈니스이면서,
충분한 기술적 인프라(분석 파이프라인 + 축적 시스템 + 생성 엔진)를
갖춘 업체만이 이 방향을 실행할 수 있음.

### 고객 관점의 가치 전환

```
기존: "오늘 찍은 사진 사세요" (단발성 구매)

person-conditioned distribution 시대:
  "당신의 3년간 표정 포트폴리오입니다"
  "지난번에 없던 이런 표정이 새로 수집되었어요"
  "당신만의 표현 스펙트럼이 이만큼 채워졌어요"

  → 방문할수록 포트폴리오가 풍부해짐
  → 재방문 동기 (수집 완성도, coverage gap)
  → portrait 서비스가 일회성 상품 → 장기 구독 가치로 전환
  → "나의 AI 포트폴리오"라는 개인화된 자산
```

---

## 연관 문서

| 문서 | 관계 |
|------|------|
| `vision-face-state-embedding.md` | 상위 비전 (conditional multi-mode embedding) |
| `how-visualbind.md` | 현재 signal binding (43D, 전략별 판단) |
| `how-visualgrow.md` | 적응 시스템 (per-member distribution update = visualgrow 영역) |
| `pipeline-architecture.md` | 3-Layer 파이프라인에서의 위치 |

## 참고

| 개념 | 수학적 기반 |
|------|-----------|
| Person distribution | Multivariate Gaussian N(μ, Σ) per member |
| Cold start prior | Normal-Inverse-Wishart conjugate prior |
| Diversity measure | Mahalanobis distance, convex hull volume |
| Frame selection | Submodular function maximization (marginal gain) |
| Coverage optimization | k-DPP (Determinantal Point Process) |
| Online update | Welford algorithm (incremental mean/covariance) |
