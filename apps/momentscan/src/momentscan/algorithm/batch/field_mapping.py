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
    FieldMapping("face.detect", "face_detected", "얼굴 검출 여부", "gate"),
    FieldMapping("face.detect", "face_confidence", "주 얼굴 검출 신뢰도", "gate", "gate_face_confidence"),
    FieldMapping("face.detect", "face_area_ratio", "주 얼굴 영역 비율", "quality", "quality_face_size_weight"),
    FieldMapping("face.detect", "face_center_distance", "프레임 중심 거리", "info"),
    FieldMapping("face.detect", "head_yaw", "머리 좌우 회전 (도)", "quality"),
    FieldMapping("face.detect", "head_pitch", "머리 상하 회전 (도)", "info"),
    FieldMapping("face.detect", "head_roll", "머리 기울기 (도)", "info"),
    # face.expression (3)
    FieldMapping("face.expression", "mouth_open_ratio", "비중립 표정 강도", "impact", "impact_mouth_open_weight"),
    FieldMapping("face.expression", "eye_open_ratio", "눈 개방도 proxy", "info"),
    FieldMapping("face.expression", "smile_intensity", "미소 강도", "info"),
    # body.pose (4)
    FieldMapping("body.pose", "wrist_raise", "손목 들어올림 비율", "impact", "impact_wrist_raise_weight"),
    FieldMapping("body.pose", "torso_rotation", "어깨 기울기 (도)", "impact", "impact_torso_rotation_weight"),
    FieldMapping("body.pose", "hand_near_face", "손-얼굴 근접도", "info"),
    FieldMapping("body.pose", "elbow_angle_change", "팔꿈치 평균 각도 (도)", "info"),
    # frame.quality (3)
    FieldMapping("frame.quality", "blur_score", "선명도 (Laplacian 분산)", "quality", "quality_blur_weight"),
    FieldMapping("frame.quality", "brightness", "평균 밝기 (0-255)", "gate"),
    FieldMapping("frame.quality", "contrast", "대비 (표준편차)", "info"),
    # face.classify (1)
    FieldMapping("face.classify", "main_face_confidence", "주탑승자 분류 신뢰도", "info"),
    # frame.scoring (1)
    FieldMapping("frame.scoring", "frame_score", "종합 프레임 점수", "info"),
)

PIPELINE_DELTA_SPECS: tuple[DeltaSpec, ...] = (
    DeltaSpec("mouth_open_ratio"),
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
