# Stage Contracts

## 문서의 목적

이 문서는 Stage의 내부 입출력 계약과 실행 흐름을 정의한다.

---

## 공통 원칙

- Stage 간 데이터 전달은 asset_ref 기반으로만 이루어진다.
- 원본 비디오는 Discover 단계에서만 접근한다.
- metadata 세부 스키마/전달 방식은 추후 정의하며, 스켈레톤 단계에서는 임의 metadata 리스트를 허용한다.
- Stage 내부에서는 media_handle/frame_source 같은 일시적 핸들을 사용할 수 있다.
- state와 moment는 동일 media_handle/timebase(ms) 기준으로 동기화되어야 한다.
- 서로 다른 fps로 분석하더라도 timestamp_ms 기준으로 정합한다.
- Stage는 orchestration에 집중하며 알고리즘/정책은 도메인 모듈에 둔다.
- Stage 고유 로직이 필요하면 policy/planner로 분리하고 Stage는 실행만 담당한다.
- Stage 추가는 동일한 run 계약을 유지하며 StageExecutor는 공통 실행 흐름만 제공한다.
- Stage 구현은 `stage/<stage_name>/` 플러그인 구조로 둔다.
- 공용 실행 유틸(스텝 실행/진행률 등)은 executor에 두되, stage 고유 스텝은 stage 내부에 둔다.

---

## media_handle & frame_source (스켈레톤)

- media_handle은 video_ref를 정규화한 **경량 참조 + 기본 메타 + 캐시 키**다.
- frame_source는 media_handle에서 **요청 fps/구간에 대한 지연 생성 스트림**이다.
- 동일 video_ref는 handle 재사용을 기본으로 하며, 프레임 캐시는 제한적으로 유지한다(LRU 등).
- sampling 정책은 stage(또는 도메인 요구)가 결정하고, media가 실행한다.

---

## Stage 1 — DISCOVER (내부 계약)

### 입력

- video_ref (로컬 경로, file://, http(s)://, s3://, blob_ref 허용)
- member_id (optional, 고객 히스토리 기반 중복 제거/누적에 사용)

### 출력

- moment_pack (N개의 moment 묶음)
  - moments[]: moment_ref, moment_clip_ref, keyframe_pack_ref, moment_meta
- history_updated (member_id 제공 시 고객 히스토리 갱신 여부)

moment_pack은 Discover 응답용 구조이며 저장은 moment 자산 단위로 수행한다.
- history_updated (member_id 제공 시 고객 히스토리 갱신 여부)

### 내부 흐름

1. media: video_ref 정규화/등록 → media_handle 생성
2. stage → media: state용 frame_source 요청 (fps/구간)
3. state: frame_source 기반 상태 타임라인 생성 (state_timeline_ref)
4. moment: 후보 생성 및 선정 (state_timeline_ref, member_id, 동일 media_handle 기준)
5. moment → stage: extraction_plan 생성 (time_range, keyframe_timestamps)
6. stage → media: keyframe_pack/clip 추출 (media_handle 기반)
7. asset: 결과 저장 (moment meta에 clip/keyframe/meta 포함)
   - media는 ObservationPort로 프레임 관측 이벤트를 보낼 수 있다.
8. asset: 고객 히스토리 갱신 (member_id 제공 시)

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
