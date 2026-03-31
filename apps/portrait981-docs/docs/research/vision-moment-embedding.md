# Moment Embedding — personmemory 기반 고객 이해 임베딩

## 핵심 아이디어

face_id는 "누구인가"만 답한다. moment embedding은 "어떤 사람인가"를 답한다.

```
face_id (기존):     image → z_face    → "이 얼굴 = 저 얼굴" (identity)
moment_emb (제안):  image → z_moment  → "이 사람의 portrait 잠재력" (understanding)
```

personmemory에 축적된 고객 이해 — 표정 패턴, 포즈 선호, 스타일 반응, 상태 변화 —
를 하나의 embedding space로 압축한다.

## 왜 face_id보다 강력한가

| | face_id | moment embedding |
|---|---------|-----------------|
| 학습 신호 | 얼굴 동일 여부 | 고객 경험 전체 |
| 표현 | "누구인가" (정적) | "어떤 사람인가" (동적) |
| 입력 | 단일 이미지 | 관찰 이력 + signal + 생산 반응 |
| 공간 의미 | 가까움 = 같은 사람 | 가까움 = portrait 특성 유사 |

```
쌍둥이: face_id 가까움, moment_emb 다를 수 있음 (성격이 다르니까)
남남:   face_id 멀음, moment_emb 가까울 수 있음 (사진 스타일이 비슷)
```

## member_id: 강력한 집결 지점

기존 face recognition의 근본 문제:
```
야외 어트랙션 → 조명 변화, 움직임, 가려짐
→ ArcFace embedding이 같은 사람이어도 흔들림
→ face_id 기반 cross-visit 매칭 불안정
```

portrait981은 **검증된 member_id**를 보유:
```
중앙 시스템 → 티켓/QR → member_id 확정 (100% 정확)
```

이 member_id가 학습의 anchor:
```python
# 기존 face recognition: face만으로 identity 학습
loss = ArcFace(z_face, face_label)  # face_label이 부정확할 수 있음

# moment embedding: member_id가 identity를 보장
loss = (
    identity_loss(z_moment, member_id)     # member_id로 확실한 동일인 학습
  + state_loss(z_moment, expression_label)  # 표정 상태 구분
  + style_loss(z_moment, purchase_history)  # 스타일 선호 학습
)
```

### member_id → face_id 정제

member_id가 확실한 anchor이므로, **face embedding의 noise를 줄이는 방향**으로도 학습 가능:

```python
# 같은 member_id의 모든 관찰에서 face embedding 수집
embeddings_042 = personmemory.get_face_embeddings("park_042")
# → [visit1_emb, visit2_emb, ..., visit5_emb]
# → 조명/각도/표정 변화에도 같은 사람

# member_id를 supervision으로 face encoder fine-tune
for (emb_a, emb_b) in same_member_pairs:
    loss += contrastive(emb_a, emb_b, positive=True)

for (emb_a, emb_c) in different_member_pairs:
    loss += contrastive(emb_a, emb_c, positive=False)
```

ArcFace가 일반 환경에서 학습된 반면,
이 방식은 **우리 환경(야외 어트랙션)에서 검증된 identity**로 fine-tune.
→ 우리 도메인에서 face_id 정확도가 대폭 향상.

```
일반 ArcFace: 야외 어트랙션에서 face_id 불안정
+ member_id supervision: 같은 사람을 확실히 아니까
= 도메인 특화 face encoder: 우리 환경에서 안정적인 face_id
```

## 학습 구조

### Phase 1: member_id 기반 face refinement

```
입력:  관찰 이미지 + member_id
출력:  refined face embedding
학습:  member_id contrastive loss

효과:  우리 환경에서 face_id 정확도 향상
필요:  member_id가 부여된 다수의 방문 데이터
```

### Phase 2: State-aware embedding

```
입력:  이미지 + 43D signal
출력:  z = [z_id | z_state | z_pose | z_quality]
학습:
  - identity: member_id contrastive
  - state: expression label (manual + XGBoost)
  - pose: pose label
  - quality: face quality signals

효과:  같은 사람의 다른 상태를 구분하면서 identity 유지
```

### Phase 3: Full moment embedding

```
입력:  이미지 + signal + personmemory profile
출력:  z_moment (compact representation of understanding)
학습:
  - identity: member_id anchor
  - state: expression/pose
  - style affinity: 구매 이력 기반 선호 학습
  - novelty: 새로운 정보인지 판단

효과:  고객에 대한 전체 이해를 하나의 벡터로
```

## 응용

### Portrait Affinity (정보 가치 판단)
```
새 프레임 → z_moment 추출
→ 기존 personmemory의 z_moment들과 비교
→ 거리가 멀면: "새로운 정보" → 수집
→ 거리가 가까우면: "중복" → 스킵
```

### Cross-member Transfer (콜드 스타트 해결)
```
신규 고객 (첫 방문)
→ 첫 몇 프레임으로 z_moment 추출
→ embedding 공간에서 유사한 기존 고객 검색
→ 유사 고객에게 성공한 스타일 추천
```

### Style Prediction (구매 예측)
```
z_moment → style predictor → "이 사람은 warm_portrait 좋아할 확률 0.82"
구매 이력 없이도 예측 가능 (유사 고객 기반)
```

### Active Collection (능동 수집)
```
z_moment 공간에서 현재 위치 분석
→ coverage가 부족한 방향 식별
→ 다음 탑승 시 해당 방향 우선 수집
```

## Face State Embedding과의 관계

```
Face State Embedding (기존 비전 문서):
  z = [z_id | z_state | z_pose | z_quality]
  → per-frame, 현재 상태 표현

Moment Embedding (확장):
  z_moment = f(face_state_history, signal_profile, creative_history)
  → per-member, 누적된 이해 표현
```

Face State Embedding은 moment embedding의 **입력 feature**가 된다.
매 프레임마다 face state를 추출하고, 이들의 분포가 moment embedding을 구성.

## 데이터 요구사항

```
Phase 1 (face refinement):
  - member_id 부여된 데이터 최소 100명 × 3회 방문
  - 현재: 0명 (member_id 미운영)
  → personmemory Phase A-B가 가동된 이후

Phase 2 (state-aware):
  - Phase 1 + expression/pose 라벨
  - 현재 라벨링 데이터 활용 가능 (member_id 부여 시)

Phase 3 (full moment):
  - Phase 2 + 구매 이력
  - gallery + 고객 반응 데이터 필요
```

## 핵심 통찰

1. **member_id가 anchor** — 검증된 identity가 모든 학습의 기반
2. **face_id는 출발점이지 목적지가 아님** — member_id로 정제하면 도메인 특화 face encoder
3. **personmemory = 학습 데이터 생성기** — 서비스가 돌수록 데이터가 쌓이고 모델이 진화
4. **이미지 → 숫자 → 이해 → 표현** — 관찰에서 시작해 고객 이해로 압축
5. **portrait 특성 공간** — face identity 공간과 다른, 새로운 representation
