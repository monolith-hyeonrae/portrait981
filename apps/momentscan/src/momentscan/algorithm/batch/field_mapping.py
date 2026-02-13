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
                 rationale="상하 회전 참고. head_velocity 파생 필드의 소스"),
    FieldMapping("face.detect", "head_roll", "머리 기울기 (도)", "info",
                 rationale="머리 기울기 참고. 현재 scoring 미사용"),
    # face.expression (3)
    FieldMapping("face.expression", "mouth_open_ratio", "비중립 표정 강도", "impact", "impact_mouth_open_weight",
                 rationale="환호/놀람 등 감정 표현이 큰 순간이 라이드 하이라이트"),
    FieldMapping("face.expression", "eye_open_ratio", "눈 개방도 proxy", "gate",
                 rationale="눈 감은 사진은 인물 사진으로 부적합. 0.15 미만이면 탈락"),
    FieldMapping("face.expression", "smile_intensity", "미소 강도", "impact", "impact_smile_intensity_weight",
                 rationale="미소 피크는 감정적으로 좋은 순간. 평소 무표정/부정 표정 대비 미소 급등이 특히 의미 있음"),
    # body.pose (4)
    FieldMapping("body.pose", "wrist_raise", "손목 들어올림 비율", "impact", "impact_wrist_raise_weight",
                 rationale="손을 올리는 동작은 라이드 즐기는 대표적 제스처"),
    FieldMapping("body.pose", "torso_rotation", "어깨 기울기 (도)", "impact", "impact_torso_rotation_weight",
                 rationale="상체 움직임이 큰 순간 = 활발한 반응 구간"),
    FieldMapping("body.pose", "hand_near_face", "손-얼굴 근접도", "info",
                 rationale="손으로 얼굴 가리는 순간 감지용. 향후 gate 추가 후보"),
    FieldMapping("body.pose", "elbow_angle_change", "팔꿈치 평균 각도 (도)", "info",
                 rationale="팔 동작 크기. wrist_raise와 중복도 높아 현재 미사용"),
    # frame.quality (3)
    FieldMapping("frame.quality", "blur_score", "선명도 (Laplacian 분산)", "quality", "quality_blur_weight",
                 rationale="모션블러가 심한 프레임은 사진 품질 열화. Laplacian 50 미만 탈락"),
    FieldMapping("frame.quality", "brightness", "평균 밝기 (0-255)", "gate",
                 rationale="너무 밝거나 너무 어두우면 후보정으로도 복구 어려움. 40-220 범위만 통과"),
    FieldMapping("frame.quality", "contrast", "대비 (표준편차)", "info",
                 rationale="대비 정보. 현재 brightness와 blur로 충분하여 미사용"),
    # face.classify (1)
    FieldMapping("face.classify", "main_face_confidence", "주탑승자 분류 신뢰도", "info",
                 rationale="주탑승자 분류 신뢰도 참고용"),
    # frame.scoring (1)
    FieldMapping("frame.scoring", "frame_score", "종합 프레임 점수", "info",
                 rationale="종합 프레임 점수 참고용"),
)

PIPELINE_DELTA_SPECS: tuple[DeltaSpec, ...] = (
    DeltaSpec("mouth_open_ratio"),
    DeltaSpec("smile_intensity"),
    DeltaSpec("head_yaw"),
    DeltaSpec("head_pitch"),
    DeltaSpec("wrist_raise"),
    DeltaSpec("torso_rotation"),
    DeltaSpec("face_area_ratio"),
    DeltaSpec("brightness"),
)

PIPELINE_DERIVED_FIELDS: tuple[DerivedField, ...] = (
    DerivedField("head_velocity", ("head_yaw", "head_pitch"), "\u221a(\u0394yaw\u00b2 + \u0394pitch\u00b2) / dt"),
    DerivedField("frontalness", ("head_yaw",), "1 - |yaw| / max_yaw, clamped [0,1]"),
)
