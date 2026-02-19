"""FlowData → IdentityRecord 변환.

on_frame() 콜백에서 호출되어, raw 임베딩을 보존한 IdentityRecord를 생성.
batch/extract.py의 스칼라 추출과 달리 임베딩 벡터를 그대로 저장.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from momentscan.algorithm.identity.types import IdentityRecord


def extract_identity_record(
    frame: Any, results: List[Any]
) -> Optional[IdentityRecord]:
    """프레임의 analyzer 결과를 IdentityRecord로 변환한다.

    ArcFace (face.detect) 임베딩이 없으면 None을 반환한다.

    Args:
        frame: visualpath Frame 객체.
        results: terminal node의 FlowData 리스트.

    Returns:
        IdentityRecord, 또는 face.detect 임베딩이 없으면 None.
    """
    if not results:
        return None

    obs_by_source: dict[str, Any] = {}
    for flow_data in results:
        for obs in getattr(flow_data, "observations", []):
            source = getattr(obs, "source", None)
            if source:
                obs_by_source[source] = obs

    face_obs = obs_by_source.get("face.detect")
    if face_obs is None:
        return None

    face = _get_main_face(face_obs)
    if face is None:
        return None

    # ArcFace embedding is required
    embedding = getattr(face, "embedding", None)
    if embedding is None:
        return None

    e_id = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(e_id)
    if norm < 1e-8:
        return None
    e_id = e_id / norm

    record = IdentityRecord(
        frame_idx=getattr(frame, "frame_id", 0),
        timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
        e_id=e_id,
        face_confidence=float(getattr(face, "confidence", 0.0)),
        face_area_ratio=float(getattr(face, "area_ratio", 0.0)),
        head_yaw=float(getattr(face, "yaw", 0.0)),
        head_pitch=float(getattr(face, "pitch", 0.0)),
        head_roll=float(getattr(face, "roll", 0.0)),
        face_bbox=getattr(face, "bbox", None),
    )

    # face.expression
    _extract_expression(record, obs_by_source.get("face.expression"))

    # frame.quality
    _extract_quality(record, obs_by_source.get("frame.quality"))

    # face.embed → DINOv2 face embedding + crop box
    _extract_face_embed(record, obs_by_source.get("face.embed"))

    # body.embed → DINOv2 body embedding + crop box
    _extract_body_embed(record, obs_by_source.get("body.embed"))

    # face.au → AU intensities
    _extract_face_au(record, obs_by_source.get("face.au"))

    # head.pose → precise yaw/pitch/roll (overrides geometric estimate)
    _extract_head_pose(record, obs_by_source.get("head.pose"))

    # face.classify → person_id
    _extract_person_id(record, obs_by_source.get("face.classify"))

    return record


def _get_main_face(obs: Any) -> Any:
    """Observation.data에서 주 얼굴(area_ratio 최대)을 반환."""
    output = getattr(obs, "data", None)
    if output is None:
        return None
    faces = getattr(output, "faces", None)
    if not faces:
        return None
    return max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))


def _extract_expression(record: IdentityRecord, obs: Any) -> None:
    """face.expression signals 추출."""
    if obs is None:
        return

    face = _get_main_face(obs)
    if face is not None:
        face_signals = getattr(face, "signals", {}) or {}
        record.mouth_open_ratio = float(getattr(face, "expression", 0.0))
        neutral = float(face_signals.get("em_neutral", 1.0))
        record.eye_open_ratio = 1.0 - neutral
        record.smile_intensity = float(face_signals.get("em_happy", 0.0))
        return

    signals = getattr(obs, "signals", {}) or {}
    record.mouth_open_ratio = float(signals.get("max_expression", 0.0))
    record.smile_intensity = float(signals.get("expression_happy", 0.0))


def _extract_quality(record: IdentityRecord, obs: Any) -> None:
    """frame.quality signals 추출."""
    if obs is None:
        return
    signals = getattr(obs, "signals", {}) or {}
    record.blur_score = float(signals.get("blur_score", 0.0))
    record.brightness = float(signals.get("brightness", 0.0))


def _extract_face_embed(record: IdentityRecord, obs: Any) -> None:
    """face.embed: DINOv2 face embedding + crop box."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return

    e_face = getattr(data, "e_face", None)
    if e_face is not None:
        record.e_face = np.asarray(e_face, dtype=np.float32)

    record.face_crop_box = getattr(data, "face_crop_box", None)
    if record.image_size is None:
        record.image_size = getattr(data, "image_size", None)


def _extract_body_embed(record: IdentityRecord, obs: Any) -> None:
    """body.embed: DINOv2 body embedding + crop box."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return

    e_body = getattr(data, "e_body", None)
    if e_body is not None:
        record.e_body = np.asarray(e_body, dtype=np.float32)

    record.body_crop_box = getattr(data, "body_crop_box", None)
    if record.image_size is None:
        record.image_size = getattr(data, "image_size", None)


def _extract_face_au(record: IdentityRecord, obs: Any) -> None:
    """face.au → AU intensities dict."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    au_list = getattr(data, "au_intensities", None)
    if au_list and len(au_list) > 0:
        record.au_intensities = au_list[0]  # main face


def _extract_head_pose(record: IdentityRecord, obs: Any) -> None:
    """head.pose → precise yaw/pitch/roll (overrides geometric estimate)."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    estimates = getattr(data, "estimates", None)
    if estimates and len(estimates) > 0:
        est = estimates[0]  # main face
        record.head_yaw = float(getattr(est, "yaw", record.head_yaw))
        record.head_pitch = float(getattr(est, "pitch", record.head_pitch))
        record.head_roll = float(getattr(est, "roll", record.head_roll))


def _extract_person_id(record: IdentityRecord, obs: Any) -> None:
    """face.classify → person_id (main=0)."""
    if obs is None:
        return
    signals = getattr(obs, "signals", {}) or {}
    # main_confidence > 0 means this is the main person
    main_conf = float(signals.get("main_confidence", 0.0))
    if main_conf > 0:
        record.person_id = 0
