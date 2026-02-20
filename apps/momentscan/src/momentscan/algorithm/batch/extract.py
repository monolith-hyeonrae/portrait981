"""FlowData → FrameRecord 변환.

on_frame() 콜백에서 호출되어,
각 vpx analyzer의 Observation 결과를 FrameRecord 수치로 추출한다.

데이터 소스:
- face.detect: FaceDetectOutput.faces[i] (ArcFace embedding → face_identity)
- face.expression: ExpressionOutput.faces[i].signals (eye_open, smile)
- face.au: AU12 → smile_intensity 보정 (max 전략)
- frame.quality: Observation.signals (blur, brightness, contrast)
- face.classify: Observation.signals (main_confidence)
- shot.quality: ShotQualityOutput (head_blur, exposure, bg_sep, composition, aesthetic)
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from momentscan.algorithm.batch.types import FrameRecord

# Module-level state for ArcFace anchor (quality)
_anchor_embedding: Optional[np.ndarray] = None
_anchor_confidence: float = 0.0
_ANCHOR_DECAY: float = 0.998  # per-frame decay → half-life ~346 frames


def reset_extract_state() -> None:
    """비디오 간 상태 격리를 위한 모듈 레벨 상태 리셋."""
    global _anchor_embedding, _anchor_confidence
    _anchor_embedding = None
    _anchor_confidence = 0.0


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
        for obs in getattr(flow_data, "observations", []):
            source = getattr(obs, "source", None)
            if source:
                obs_by_source[source] = obs

    record = FrameRecord(
        frame_idx=getattr(frame, "frame_id", 0),
        timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
    )

    face_obs = obs_by_source.get("face.detect")
    _extract_face_detect(record, face_obs)
    _extract_face_expression(record, obs_by_source.get("face.expression"))
    _extract_face_au(record, obs_by_source.get("face.au"))
    _extract_quality(record, obs_by_source.get("frame.quality"))
    _extract_face_classify(record, obs_by_source.get("face.classify"))
    _extract_shot_quality(record, obs_by_source.get("shot.quality"))

    return record


def _get_main_face(obs: Any) -> Any:
    """Observation.data에서 주 얼굴(area_ratio 최대)을 반환한다."""
    output = getattr(obs, "data", None)
    if output is None:
        return None
    faces = getattr(output, "faces", None)
    if not faces:
        return None
    return max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))


def _extract_face_detect(record: FrameRecord, obs: Any) -> None:
    """face.detect: FaceDetectOutput.faces에서 주 얼굴 수치 추출."""
    global _anchor_embedding, _anchor_confidence

    if obs is None:
        return

    face = _get_main_face(obs)
    if face is None:
        return

    record.face_detected = True
    record.face_confidence = float(getattr(face, "confidence", 0.0))
    record.face_area_ratio = float(getattr(face, "area_ratio", 0.0))
    record.face_center_distance = float(getattr(face, "center_distance", 0.0))
    record.head_yaw = float(getattr(face, "yaw", 0.0))
    record.head_pitch = float(getattr(face, "pitch", 0.0))
    record.head_roll = float(getattr(face, "roll", 0.0))

    # ArcFace embedding → quality signal
    embedding = getattr(face, "embedding", None)
    if embedding is not None:
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
            # Decay anchor confidence so newer good faces can overtake
            _anchor_confidence *= _ANCHOR_DECAY
            face_conf = record.face_confidence
            if face_conf > _anchor_confidence:
                _anchor_embedding = emb
                _anchor_confidence = face_conf
            # Compute similarity to anchor
            if _anchor_embedding is not None:
                sim = float(np.dot(emb, _anchor_embedding))
                record.face_identity = max(0.0, sim)


def _extract_face_expression(record: FrameRecord, obs: Any) -> None:
    """face.expression: ExpressionOutput.faces에서 표정 수치 추출."""
    if obs is None:
        return

    face = _get_main_face(obs)
    if face is not None:
        face_signals = getattr(face, "signals", {}) or {}
        neutral = float(face_signals.get("em_neutral", 1.0))
        record.eye_open_ratio = 1.0 - neutral
        record.smile_intensity = float(face_signals.get("em_happy", 0.0))
        return

    # Fallback: frame-level signals
    signals = getattr(obs, "signals", {}) or {}
    record.smile_intensity = float(signals.get("expression_happy", 0.0))


def _extract_face_au(record: FrameRecord, obs: Any) -> None:
    """face.au: AU12(Lip Corner Puller)로 smile_intensity를 보정한다.

    max() 전략: AU12 > em_happy 이면 AU12 사용, 아니면 em_happy 유지.
    """
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    au_list = getattr(data, "au_intensities", None)
    if not au_list:
        return
    au = au_list[0]  # main face
    au12 = au.get("AU12", None)
    if au12 is not None:
        smile_from_au12 = min(1.0, float(au12) / 3.0)
        record.smile_intensity = max(record.smile_intensity, smile_from_au12)


def _extract_quality(record: FrameRecord, obs: Any) -> None:
    """frame.quality: Observation.signals에서 품질 수치 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.blur_score = float(signals.get("blur_score", 0.0))
    record.brightness = float(signals.get("brightness", 0.0))
    record.contrast = float(signals.get("contrast", 0.0))


def _extract_face_classify(record: FrameRecord, obs: Any) -> None:
    """face.classify: Observation.signals에서 주탑승자 정보 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.main_face_confidence = float(signals.get("main_confidence", 0.0))


def _extract_shot_quality(record: FrameRecord, obs: Any) -> None:
    """shot.quality: ShotQualityOutput에서 portrait crop 품질 수치 추출."""
    if obs is None:
        return

    data = getattr(obs, "data", None)
    if data is None:
        return

    record.head_blur = float(getattr(data, "head_blur", 0.0))
    record.head_exposure = float(getattr(data, "head_exposure", 0.0))
    record.head_aesthetic = float(getattr(data, "head_aesthetic", 0.0))
