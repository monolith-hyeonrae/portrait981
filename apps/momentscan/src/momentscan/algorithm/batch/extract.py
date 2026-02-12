"""FlowData → FrameRecord 변환.

on_frame() 콜백에서 호출되어,
각 vpx analyzer의 Observation 결과를 FrameRecord 수치로 추출한다.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

from momentscan.algorithm.batch.types import FrameRecord


def extract_frame_record(frame: Any, results: List[Any]) -> Optional[FrameRecord]:
    """프레임의 analyzer 결과를 FrameRecord로 변환한다.

    Args:
        frame: visualpath Frame 객체.
        results: terminal node의 FlowData 리스트.

    Returns:
        FrameRecord, 또는 결과가 비어있으면 None.
    """
    if not results:
        return None

    # FlowData에서 Observation들을 source별로 모은다
    obs_by_source: dict[str, Any] = {}
    for flow_data in results:
        for obs in getattr(flow_data, "results", []):
            source = getattr(obs, "source", None)
            if source:
                obs_by_source[source] = obs

    record = FrameRecord(
        frame_idx=getattr(frame, "frame_id", 0),
        timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
    )

    _extract_face_detect(record, obs_by_source.get("face.detect"))
    _extract_face_expression(record, obs_by_source.get("face.expression"))
    _extract_body_pose(record, obs_by_source.get("body.pose"))
    _extract_quality(record, obs_by_source.get("frame.quality"))
    _extract_face_classify(record, obs_by_source.get("face.classify"))
    _extract_frame_scoring(record, obs_by_source.get("frame.scoring"))

    return record


def _extract_face_detect(record: FrameRecord, obs: Any) -> None:
    """face.detect Observation에서 수치 feature 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    face_count = signals.get("face_count", 0)
    if face_count == 0:
        return

    record.face_detected = True
    record.face_confidence = float(signals.get("face_confidence", 0.0))
    record.face_area_ratio = float(signals.get("face_area_ratio", 0.0))
    record.face_center_distance = float(signals.get("face_center_distance", 0.0))
    record.head_yaw = float(signals.get("head_yaw", 0.0))
    record.head_pitch = float(signals.get("head_pitch", 0.0))
    record.head_roll = float(signals.get("head_roll", 0.0))


def _extract_face_expression(record: FrameRecord, obs: Any) -> None:
    """face.expression Observation에서 표정 수치 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.mouth_open_ratio = float(signals.get("mouth_open_ratio", 0.0))
    record.eye_open_ratio = float(signals.get("eye_open_ratio", 0.0))
    record.smile_intensity = float(signals.get("smile_intensity", 0.0))


def _extract_body_pose(record: FrameRecord, obs: Any) -> None:
    """body.pose Observation에서 포즈 수치 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.wrist_raise = float(signals.get("wrist_raise", 0.0))
    record.elbow_angle_change = float(signals.get("elbow_angle_change", 0.0))
    record.torso_rotation = float(signals.get("torso_rotation", 0.0))
    record.hand_near_face = float(signals.get("hand_near_face", 0.0))


def _extract_quality(record: FrameRecord, obs: Any) -> None:
    """frame.quality Observation에서 품질 수치 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.blur_score = float(signals.get("blur_score", 0.0))
    record.brightness = float(signals.get("brightness", 0.0))
    record.contrast = float(signals.get("contrast", 0.0))


def _extract_face_classify(record: FrameRecord, obs: Any) -> None:
    """face.classify Observation에서 주탑승자 정보 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.main_face_confidence = float(signals.get("main_confidence", 0.0))


def _extract_frame_scoring(record: FrameRecord, obs: Any) -> None:
    """frame.scoring Observation에서 프레임 점수 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.frame_score = float(signals.get("total_score", 0.0))
