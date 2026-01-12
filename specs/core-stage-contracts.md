# Stage Contracts

## 문서의 목적

이 문서는 Stage의 내부 입출력 계약과 실행 흐름을 정의한다.

---

## 공통 원칙

- Stage 간 데이터 전달은 asset_ref 기반으로만 이루어진다.
- 원본 비디오는 Discover 단계에서만 접근한다.
- metadata 세부 스키마/전달 방식은 추후 정의하며, 스켈레톤 단계에서는 임의 metadata 리스트를 허용한다.

---

## Stage 1 — DISCOVER (내부 계약)

### 입력

- video_ref (로컬 경로, file://, http(s)://, s3://, blob_ref 허용)
- customer_id (optional, 외부 엔트리포인트에서 소유자 바인딩 시 제공)

### 출력

- moment_refs
- keyframe_pack_refs
- moment_clip_refs
- moment_metadata_refs (TBD, 스켈레톤 단계에서는 임의 구조/빈 값 가능)
- history_updated (customer_id 제공 시에만 의미가 있음)

### 내부 흐름

1. media: video_ref 접근/준비
2. state: 상태 타임라인 생성 (state_timeline_ref)
3. moment: 후보 생성
4. moment: 중복 제거 + 다양성 조건 반영하여 N개 선정
5. moment → media: extraction_plan 생성 (time_range, keyframe_timestamps)
6. media: keyframe_pack/clip 추출
7. asset: 결과 저장 및 고객 히스토리 업데이트 (customer_id 제공 시)
   - media는 ObservationPort로 프레임 관측 이벤트를 보낼 수 있다.

### 스켈레톤 기본 포맷

- keyframe_pack: JPEG 프레임 zip (최종 포맷 TBD)
- moment_clip: MP4 (H.264/AAC, 재인코딩)

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
