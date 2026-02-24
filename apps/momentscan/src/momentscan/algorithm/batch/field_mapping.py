"""Pipeline field mapping registry.

모듈 출력 -> FrameRecord 필드 매핑, delta 계산 대상, 파생 필드를
한 곳에 선언한다. info.py와 highlight.py가 같은 소스를 읽는다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldMapping:
    """모듈 출력 -> FrameRecord 필드 매핑 하나."""

    source: str  # 모듈 이름 (e.g. "face.detect")
    record_field: str  # FrameRecord 속성명 (e.g. "face_confidence")
    description: str = ""
    scoring_role: str | None = None  # "gate" | "quality" | "impact" | "info" | None
    scoring_weight_key: str | None = None  # HighlightConfig 속성명
    rationale: str = ""  # 비즈니스 의사결정 이유


@dataclass(frozen=True)
class DeltaSpec:
    """Temporal delta 계산 대상 필드."""

    record_field: str
    alpha: float = 0.1  # EMA baseline alpha


@dataclass(frozen=True)
class DerivedField:
    """다른 필드에서 파생되는 필드."""

    name: str
    source_fields: tuple[str, ...]
    description: str = ""


PIPELINE_FIELD_MAPPINGS: tuple[FieldMapping, ...] = (
    # face.detect (7)
    FieldMapping("face.detect", "face_detected", "얼굴 검출 여부", "gate",
                 rationale="얼굴이 없는 프레임은 인물 사진으로 사용 불가"),
    FieldMapping("face.detect", "face_confidence", "주 얼굴 검출 신뢰도", "gate", "gate_face_confidence",
                 rationale="오검출된 얼굴로 선별하면 의미 없는 사진이 뽑힘. 0.7 이상만 신뢰"),
    FieldMapping("face.detect", "face_area_ratio", "주 얼굴 영역 비율", "gate+quality", "quality_face_size_weight",
                 rationale="얼굴이 너무 작으면 인쇄/SNS 사진으로 부적합. 화면의 2% 이상 (gate: 0.02)"),
    FieldMapping("face.detect", "face_center_distance", "프레임 중심 거리", "info",
                 rationale="프레임 중심 거리 참고용. 현재 scoring 미사용"),
    FieldMapping("face.detect", "head_yaw", "머리 좌우 회전 (도)", "gate+quality",
                 rationale="정면에 가까운 사진이 고객 만족도 높음. gate: 70도 초과 탈락, quality: frontalness"),
    FieldMapping("face.detect", "head_pitch", "머리 상하 회전 (도)", "gate",
                 rationale="상하 회전 50도 초과 탈락 (gate). head.pose 정밀값 우선"),
    FieldMapping("face.detect", "head_roll", "머리 기울기 (도)", "info",
                 rationale="머리 기울기 참고. 현재 scoring 미사용"),
    FieldMapping("face.detect", "face_identity", "얼굴 동일성", "quality", "quality_face_identity_weight",
                 rationale="ArcFace anchor 대비 cosine similarity. 정면/선명/가림없음을 통합적으로 평가"),
    # face.quality (2)
    FieldMapping("face.quality", "head_blur", "헤드샷 선명도 (Laplacian)", "gate+quality", "quality_head_blur_weight",
                 rationale="portrait crop 내 얼굴 선명도. gate: 30 미만 탈락, quality: 가중 품질 점수"),
    FieldMapping("face.quality", "head_exposure", "헤드샷 밝기 (0-255)", "info",
                 rationale="얼굴 crop 내 노출 참고용"),
    FieldMapping("face.quality", "head_contrast", "얼굴 로컬 대비 (CV)", "gate",
                 rationale="CV = std/mean. 피부톤 무관 노출 품질 메트릭. contrast_min(0.05) 미만 탈락"),
    FieldMapping("face.quality", "clipped_ratio", "과포화 비율", "gate",
                 rationale="마스크 내 >250 픽셀 비율. clipped_max(0.3) 초과 탈락"),
    FieldMapping("face.quality", "crushed_ratio", "암전 비율", "gate",
                 rationale="마스크 내 <5 픽셀 비율. crushed_max(0.3) 초과 탈락"),
    FieldMapping("face.quality", "mask_method", "마스크 방법", "info",
                 rationale="parsing(BiSeNet) → landmark(5-point) → center_patch(50%) fallback"),
    # portrait.score (1)
    FieldMapping("portrait.score", "head_aesthetic", "포트레이트 품질 점수 (CLIP)", "info",
                 rationale="CLIP aggregate portrait quality. 4축 개별 점수(disney_smile, charisma, wild_roar, playful_cute)와 복합 시그널이 scoring 담당. 참고용"),
    # face.expression (2)
    FieldMapping("face.expression", "eye_open_ratio", "눈 개방도 proxy", "info",
                 rationale="em_neutral proxy — 신뢰도 부족으로 gate에서 제거. 참고용"),
    FieldMapping("face.expression", "smile_intensity", "미소 강도", "impact", "impact_smile_intensity_weight",
                 rationale="face.au 실행 시 LibreFace AU12(Lip Corner Puller) 직접 측정(DISFA 0-5 → min(AU12/3.0, 1.0)). face.au 없으면 HSEmotion em_happy 폴백. 표현 메트릭이므로 face.expression 패널에 표시"),
    # frame.quality (3)
    FieldMapping("frame.quality", "blur_score", "선명도 (Laplacian 분산)", "gate",
                 rationale="모션블러가 심한 프레임은 사진 품질 열화. Laplacian 50 미만 탈락 (gate_blur_min). 품질 점수는 face.quality head_blur 사용"),
    FieldMapping("frame.quality", "brightness", "평균 밝기 (0-255)", "gate",
                 rationale="너무 밝거나 너무 어두우면 후보정으로도 복구 어려움. 40-220 범위만 통과"),
    FieldMapping("frame.quality", "contrast", "대비 (표준편차)", "info",
                 rationale="대비 정보. 현재 brightness와 blur로 충분하여 미사용"),
    # face.classify (1)
    FieldMapping("face.classify", "main_face_confidence", "주탑승자 분류 신뢰도", "info",
                 rationale="주탑승자 분류 신뢰도 참고용"),
    # CLIP axes (4) — portrait.score에서 추출, portrait_best 서브시그널
    FieldMapping("portrait.score", "clip_disney_smile", "디즈니 미소 축 점수", "info",
                 rationale="따뜻한 Duchenne 미소 분위기. duchenne_smile 복합 시그널 구성 → portrait_best에 기여"),
    FieldMapping("portrait.score", "clip_charisma", "카리스마 축 점수", "info",
                 rationale="시크/당당한 분위기. portrait_best에 직접 기여"),
    FieldMapping("portrait.score", "clip_wild_roar", "포효 축 점수", "info",
                 rationale="호탕한 함성/환호 분위기. wild_intensity 복합 시그널 구성 → portrait_best에 기여"),
    FieldMapping("portrait.score", "clip_playful_cute", "장난기 축 점수", "info",
                 rationale="혀내밀기/볼부풀리기 등. portrait_best 후보 (CLIP 4축 중 max 선택)"),
    # face.au — 개별 AU (4)
    FieldMapping("face.au", "au6_cheek_raiser", "AU6 볼 올림 (Duchenne)", "info",
                 rationale="Duchenne smile 복합 시그널 구성 요소"),
    FieldMapping("face.au", "au12_lip_corner", "AU12 입꼬리 올림", "info",
                 rationale="Duchenne smile + smile_intensity 보정 근거"),
    FieldMapping("face.au", "au25_lips_part", "AU25 입술 벌림", "info",
                 rationale="wild_intensity 복합 시그널 구성 요소"),
    FieldMapping("face.au", "au26_jaw_drop", "AU26 턱 벌림", "info",
                 rationale="wild_intensity 복합 시그널 구성 요소"),
    # face.expression — neutral
    FieldMapping("face.expression", "em_neutral", "무표정 확률 (HSEmotion)", "info",
                 rationale="chill 복합 시그널 구성 요소. eye_open_ratio 산출 근거"),
    # Cross-analyzer composites (3) — portrait_best 서브시그널 (프레임별 최상위 1개만 Impact 반영)
    FieldMapping("portrait.score", "duchenne_smile", "Duchenne 미소 (CLIP×AU)", "info",
                 rationale="disney_smile × (AU6+AU12). portrait_best 후보 (프레임별 max 선택)"),
    FieldMapping("portrait.score", "wild_intensity", "포효 강도 (CLIP×AU)", "info",
                 rationale="wild_roar × max(AU25,AU26). portrait_best 후보 (프레임별 max 선택)"),
    FieldMapping("portrait.score", "chill_score", "무표정 탑승 (neutral×low_axes)", "info",
                 rationale="neutral 높음 + CLIP 축 낮음. portrait_best 후보 (프레임별 max 선택)"),
    # face.gate (2) — per-face gate analyzer 결과 (main face 기준)
    FieldMapping("face.gate", "gate_passed", "프레임 gate 통과 여부 (main face)", "gate",
                 rationale="face.gate analyzer가 DAG에서 per-face 판정한 main face gate 결과"),
    FieldMapping("face.gate", "gate_fail_reasons", "gate 실패 조건 목록", "info",
                 rationale="gate 실패 시 comma-separated 조건명. 디버깅용"),
    # face.gate — passenger (6)
    FieldMapping("face.gate", "passenger_detected", "동승자 검출 여부", "info",
                 rationale="face.gate analyzer에서 passenger role 할당된 얼굴 존재 여부"),
    FieldMapping("face.gate", "passenger_gate_passed", "동승자 gate 통과 여부", "info",
                 rationale="동승자 gate 판정 (relaxed thresholds)"),
    FieldMapping("face.gate", "passenger_gate_fail_reasons", "동승자 gate 실패 조건", "info",
                 rationale="동승자 gate 실패 시 조건명"),
    FieldMapping("face.gate", "passenger_face_area_ratio", "동승자 얼굴 영역 비율", "info",
                 rationale="동승자 얼굴 크기 (relaxed threshold)"),
    FieldMapping("face.gate", "passenger_head_blur", "동승자 헤드샷 선명도", "info",
                 rationale="동승자 얼굴 crop 선명도 (relaxed threshold)"),
    FieldMapping("face.gate", "passenger_head_exposure", "동승자 헤드샷 밝기", "info",
                 rationale="동승자 얼굴 crop 노출"),
)

PIPELINE_DELTA_SPECS: tuple[DeltaSpec, ...] = (
    DeltaSpec("smile_intensity"),
    DeltaSpec("head_yaw"),
    DeltaSpec("face_area_ratio"),
    DeltaSpec("brightness"),
    DeltaSpec("duchenne_smile"),
    DeltaSpec("wild_intensity"),
)

PIPELINE_DERIVED_FIELDS: tuple[DerivedField, ...] = (
    DerivedField("frontalness", ("head_yaw",), "1 - |yaw| / max_yaw, clamped [0,1]"),
)
