# Portrait Output and Styles

## 문서의 목적

이 문서는 시네마틱 포트레이트 결과물과 스타일별 산출 형태를 정의한다.

---

## 결과물 저장 원칙

- 모든 결과물은 Asset Bundle로 저장한다.
- API/Stage는 blob 대신 asset_ref만 반환한다.
- 동일 입력/스타일 요청 시 기존 결과 재사용이 가능하다.

---

## 스타일별 결과 타입

- base_portrait_ref: 리터칭/정준화된 기본 포트레이트 (단독 요청 가능)
- closeup_image_ref: 클로즈업 이미지 (단독 요청 가능)
- fullbody_image_ref: 풀바디 이미지 (단독 요청 가능)
- cinematic_video_ref: 3~6초 MP4 (MVP 필수 결과)
- debug_pack_ref: 입력/중간 결과 묶음 (옵션)

---

## 스타일별 입력/의존성

- base: moment_ref → (asset에서 keyframe_pack_ref 로딩) → base_portrait_ref
- closeup/fullbody: base_portrait_ref → image_ref
- cinematic: closeup_image_ref + fullbody_image_ref → cinematic_video_ref

---

## MVP 범위

- MVP 필수 산출물은 cinematic_video_ref 1개다.
- base/closeup/fullbody는 단독 요청 가능하며 cinematic 입력으로 재사용된다.
- debug_pack_ref는 옵션이다.
- style 키는 cinematic|closeup|fullbody|base로 열어둔다.
