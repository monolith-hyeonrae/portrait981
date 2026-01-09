# Stage Contracts

## 문서의 목적

이 문서는 Stage의 내부 입출력 계약과 실행 흐름을 정의한다.

---

## 공통 원칙

- Stage 간 데이터 전달은 asset_ref 기반으로만 이루어진다.
- 원본 비디오는 Discover 단계에서만 접근한다.

---

## Stage 1 — DISCOVER (내부 계약)

### 입력

- video_ref
- customer_id

### 출력

- moment_refs
- keyframe_pack_refs
- moment_clip_refs
- history_updated

### 내부 흐름

1. media: video_ref 접근/준비
2. state: 상태 타임라인 생성 (state_timeline_ref)
3. moment: 후보 생성
4. moment: 중복 제거 + 다양성 조건 반영하여 N개 선정
5. moment → media: extraction_plan 생성 (time_range, keyframe_timestamps)
6. media: keyframe_pack/clip 추출
7. asset: 결과 저장 및 고객 히스토리 업데이트

---

## Stage 2 — SYNTHESIZE (내부 계약)

### 입력

- style = base | closeup | fullbody | cinematic
- style별 입력 참조 (아래 흐름 참고)

### 출력

- generated_asset_ref
- reused_existing

### 비고

- base/closeup/fullbody는 단독 요청 가능하며 생성된 결과는 cinematic 입력으로 재사용된다.

### 스타일별 흐름

- base: moment_ref → asset(keyframe_pack_ref) → synthesis → asset(base_portrait_ref)
- closeup: base_portrait_ref → synthesis → asset(closeup_image_ref)
- fullbody: base_portrait_ref → synthesis → asset(fullbody_image_ref)
- cinematic: closeup_image_ref + fullbody_image_ref → synthesis → asset(cinematic_video_ref)
