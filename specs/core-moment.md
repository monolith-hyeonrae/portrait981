# Moment Definition

## 문서의 목적

이 문서는 moment의 정의, 최소 단위, 트리거 기준, 선정 흐름을 정리한다.

---

## 정의

moment는 고객 반응에서 **의미 있는 변화/피크가 나타나는 시간 구간**이다.

---

## 최소 단위

- time_range(start_ms, end_ms)
- label + score
- 대표 프레임(1장 또는 N장 pack)

---

## 시간 단위

- start_ms/end_ms는 ms 정수로 저장한다.
- 표시/디버그용 seconds는 필요 시 파생 계산한다.

---

## 기본 길이

- 기본 길이: 2~5초
- MVP에서는 3초 고정도 가능

---

## 입력 근거

- p981-state가 생성하는 상태 타임라인(state_timeline_ref)을 입력으로 사용한다.
- 상태 타임라인 스키마는 `specs/core-state-timeline.md`를 따른다.

---

## 트리거 유형 (MVP)

- 감정 피크: happy/angry 확률이 임계치 이상
- 상태 전이: neutral → happy, neutral → angry

---

## 후보/선정 흐름 (MVP)

1. 상태 타임라인에서 후보 구간 생성
2. 중복 제거 (동일 비디오 + 고객 히스토리)
3. 다양성 조건을 만족하는 N개 Moment 선정
4. 선정된 Moment로 media 추출 계획 생성

---

## 알고리즘 힌트 (MVP)

- 클러스터링 기반 중복 제거
- 품질 점수 + 다양성 점수 조합

---

## 필수 산출물

- keyframe_pack_ref
- moment_clip_ref
- moment_meta

moment_meta 상세 스키마는 `specs/core-asset-schema.md`를 따른다.
reason은 추천 필드이며 스켈레톤 단계에서는 생략 가능하다.

---

## 정책 경계

- Moment 분석/선정 정책은 p981-moment 내부에만 존재한다.
