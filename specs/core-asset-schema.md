# Asset Schema (MVP)

## 문서의 목적

MVP 기준 Asset 최소 스키마와 moment_meta 필수 필드를 정의한다.

---

## 공통 필수 필드

- asset_ref (opaque id)
- asset_type (자산 유형)
- customer_id
- created_at
- source (video_ref 또는 upstream asset_ref)
- blob_ref (파일 기반 자산일 때)
- meta (확장 가능한 dict/json)

---

## asset_type 예시

- moment
- keyframe_pack
- moment_clip
- state_timeline
- base_portrait
- closeup_image
- fullbody_image
- cinematic_video
- debug_pack

---

## moment_meta 필수 필드

- start_ms, end_ms (ms 정수)
- label (happy/angry/neutral)
- score (0~1)
- dedupe_hash (중복 제거용)
- diversity_key (다양성 그룹핑용)
- keyframe_pack_ref
- moment_clip_ref

---

## moment_meta 추천 필드

- quality_score (0~1)
- reason (선정 근거)

---

## 원본 비디오 원칙

- 원본 비디오는 7일 뒤 삭제되며 이후 단계는 원본에 의존하지 않는다.
- time_range + clip + keyframe은 항상 저장하도록 한다.
