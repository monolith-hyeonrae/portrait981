# TODO — 미결 작업 목록

## ~~Scoring Formula 재검토~~ ✓ 완료

**적용**: 옵션 B + C
- `final = 0.35 × quality + 0.65 × impact` (가산 방식)
- Gate-pass only EMA (gate_fail 프레임은 이전 smoothed 값 유지)
- `HighlightConfig.final_quality_blend=0.35, final_impact_blend=0.65`

---

## ~~head_aesthetic ROI 튜닝~~ ✓ 완료 → scoring 제거

**적용**: `aesthetic_expand=1.8, aesthetic_y_shift=0.3, crop_ratio=4:5`
- `face_crop()`에 `y_shift` 파라미터 추가
- 정수리 바로 위 ~ 어깨 라인까지 포트레이트 크롭

**후속 결정 (2026-02-23)**:
- `head_aesthetic` (CLIP aggregate score)는 미학적 구분 근거 부족으로 **scoring에서 제거** → info-only
- CLIP 4축 개별 점수(disney_smile, charisma, wild_roar, playful_cute)가 `portrait_best`로 Impact에 기여

---

## ~~vision-embed → portrait-score + face.quality 분리~~ ✓ 완료

- `vpx-vision-embed` → `vpx-portrait-score` 리네임 (CLIP scoring 전용)
- `face.quality` 신규 (momentscan 내부, blur/exposure)
- `face_crop()`, `BBoxSmoother` → `vpx.sdk.crop` 공유 유틸리티로 이동

---

## ~~portrait_best Impact 병합~~ ✓ 완료

- portrait.score 4축(disney_smile, charisma, wild_roar, playful_cute) 중 프레임별 max → 단일 `portrait_best` 채널
- Impact 채널: smile_intensity(0.25), head_yaw(0.15), portrait_best(0.25)

---

## ~~highlight_rules.md §6 가산 공식 업데이트~~ ✓ 완료

- §6: face.gate analyzer 조건 (confidence, blur, exposure, contrast, parsing_coverage)
- §7: Quality Score (0.30×head_blur + 0.20×face_size + 0.30×face_identity / 0.25×frontalness)
- §8: Impact Score (3채널 top-K: smile_intensity, head_yaw, portrait_best)
- §9: Final = 0.35×Quality + 0.65×Impact (가산)
- §4 feature 테이블: face.quality, portrait.score, face.au, face.gate 필드 추가

---

## ~~face.quality seg 그룹 재구성~~ ✓ 완료

- seg 그룹: face/eye/mouth/hair (이전: skin/nose/brow 등 개별 클래스)
- `_align_parse_to_crop()` 좌표 스케일링 수정 (parse crop → head crop 변환)
- 3-level mask fallback: BiSeNet → landmark ellipse → center patch

---

## 메모

- planning docs(momentscan.md, identity_builder.md, highlight_vector.md)에 `shot.quality` 참조 잔존 — 히스토리 기록으로 유지
