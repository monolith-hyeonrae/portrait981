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
    FieldMapping("face.detect", "face_area_ratio", "주 얼굴 영역 비율", "quality", "quality_face_size_weight",
                 rationale="얼굴이 너무 작으면 인쇄/SNS 사진으로 부적합. 화면의 1% 이상"),
    FieldMapping("face.detect", "face_center_distance", "프레임 중심 거리", "info",
                 rationale="프레임 중심 거리 참고용. 현재 scoring 미사용"),
    FieldMapping("face.detect", "head_yaw", "머리 좌우 회전 (도)", "quality",
                 rationale="정면에 가까운 사진이 고객 만족도 높음. 45도 이상 옆모습은 감점"),
    FieldMapping("face.detect", "head_pitch", "머리 상하 회전 (도)", "info",
                 rationale="상하 회전 참고용. 현재 scoring 미사용"),
    FieldMapping("face.detect", "head_roll", "머리 기울기 (도)", "info",
                 rationale="머리 기울기 참고. 현재 scoring 미사용"),
    FieldMapping("face.detect", "face_identity", "얼굴 동일성", "quality", "quality_face_identity_weight",
                 rationale="ArcFace anchor 대비 cosine similarity. 정면/선명/가림없음을 통합적으로 평가"),
    # shot.quality (3)
    FieldMapping("shot.quality", "head_blur", "헤드샷 선명도 (Laplacian)", "quality", "quality_head_blur_weight",
                 rationale="portrait crop 내 얼굴 선명도. 전체 프레임 blur보다 portrait 목적에 직접적"),
    FieldMapping("shot.quality", "head_exposure", "헤드샷 밝기 (0-255)", "info",
                 rationale="얼굴 crop 내 노출 참고용"),
    FieldMapping("shot.quality", "head_aesthetic", "헤드샷 미학 점수 (LAION)", "quality", "quality_head_aesthetic_weight",
                 rationale="LAION Aesthetic score for head_crop. quality score에 가중합 반영; impact score에도 per-video min-max 절대값으로 기여 (impact_head_aesthetic_weight). 0이면 shot.quality 미실행"),
    # face.expression (2)
    FieldMapping("face.expression", "eye_open_ratio", "눈 개방도 proxy", "gate",
                 rationale="눈 감은 사진은 인물 사진으로 부적합. 0.15 미만이면 탈락"),
    FieldMapping("face.expression", "smile_intensity", "미소 강도", "impact", "impact_smile_intensity_weight",
                 rationale="face.au 실행 시 LibreFace AU12(Lip Corner Puller) 직접 측정(DISFA 0-5 → min(AU12/3.0, 1.0)). face.au 없으면 HSEmotion em_happy 폴백. 표현 메트릭이므로 face.expression 패널에 표시"),
    # frame.quality (3)
    FieldMapping("frame.quality", "blur_score", "선명도 (Laplacian 분산)", "gate",
                 rationale="모션블러가 심한 프레임은 사진 품질 열화. Laplacian 50 미만 탈락 (gate_blur_min). 품질 점수는 shot.quality head_blur 사용"),
    FieldMapping("frame.quality", "brightness", "평균 밝기 (0-255)", "gate",
                 rationale="너무 밝거나 너무 어두우면 후보정으로도 복구 어려움. 40-220 범위만 통과"),
    FieldMapping("frame.quality", "contrast", "대비 (표준편차)", "info",
                 rationale="대비 정보. 현재 brightness와 blur로 충분하여 미사용"),
    # face.classify (1)
    FieldMapping("face.classify", "main_face_confidence", "주탑승자 분류 신뢰도", "info",
                 rationale="주탑승자 분류 신뢰도 참고용"),
)

PIPELINE_DELTA_SPECS: tuple[DeltaSpec, ...] = (
    DeltaSpec("smile_intensity"),
    DeltaSpec("head_yaw"),
    DeltaSpec("face_area_ratio"),
    DeltaSpec("brightness"),
)

PIPELINE_DERIVED_FIELDS: tuple[DerivedField, ...] = (
    DerivedField("frontalness", ("head_yaw",), "1 - |yaw| / max_yaw, clamped [0,1]"),
)
