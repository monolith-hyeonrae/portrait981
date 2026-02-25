"""FlowData → FrameRecord 변환.

on_frame() 콜백에서 호출되어,
각 vpx analyzer의 Observation 결과를 FrameRecord 수치로 추출한다.

데이터 소스:
- face.detect: FaceDetectOutput.faces[i] (ArcFace embedding → face_identity)
- face.expression: ExpressionOutput.faces[i].signals (eye_open, smile)
- face.au: AU12 → smile_intensity 보정 (max 전략)
- frame.quality: Observation.signals (blur, brightness, contrast)
- face.classify: Observation.signals (main_confidence)
- face.quality: FaceQualityOutput (head_blur, head_exposure)
- portrait.score: PortraitScoreOutput (head_aesthetic, CLIP axes)
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
    _extract_head_pose(record, obs_by_source.get("head.pose"))
    _extract_face_expression(record, obs_by_source.get("face.expression"))
    _extract_face_au(record, obs_by_source.get("face.au"))
    _extract_quality(record, obs_by_source.get("frame.quality"))
    _extract_face_classify(record, obs_by_source.get("face.classify"))
    _extract_face_quality(record, obs_by_source.get("face.quality"))
    _extract_portrait_score(record, obs_by_source.get("portrait.score"))
    _extract_face_gate(record, obs_by_source.get("face.gate"))
    _extract_face_baseline(record, obs_by_source.get("face.baseline"))

    _compute_composites(record)

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
        record.em_neutral = neutral
        record.smile_intensity = float(face_signals.get("em_happy", 0.0))
        return

    # Fallback: frame-level signals
    signals = getattr(obs, "signals", {}) or {}
    record.smile_intensity = float(signals.get("expression_happy", 0.0))


def _extract_face_au(record: FrameRecord, obs: Any) -> None:
    """face.au: 개별 AU 강도 추출 + AU12로 smile_intensity 보정.

    AU6/AU12/AU25/AU26을 FrameRecord에 저장하고,
    AU12는 기존 max() 전략으로 smile_intensity도 보정.
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

    # 개별 AU 강도 저장 (DISFA 0-5 스케일)
    record.au6_cheek_raiser = float(au.get("AU6", 0.0))
    record.au12_lip_corner = float(au.get("AU12", 0.0))
    record.au25_lips_part = float(au.get("AU25", 0.0))
    record.au26_jaw_drop = float(au.get("AU26", 0.0))

    # 기존 smile_intensity max 보정 유지
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


def _extract_face_quality(record: FrameRecord, obs: Any) -> None:
    """face.quality: FaceQualityOutput에서 얼굴 crop 품질 수치 추출."""
    if obs is None:
        return

    data = getattr(obs, "data", None)
    if data is None:
        return

    record.head_blur = float(getattr(data, "head_blur", 0.0))
    record.head_exposure = float(getattr(data, "head_exposure", 0.0))
    record.head_contrast = float(getattr(data, "head_contrast", 0.0))
    record.clipped_ratio = float(getattr(data, "clipped_ratio", 0.0))
    record.crushed_ratio = float(getattr(data, "crushed_ratio", 0.0))
    record.mask_method = str(getattr(data, "mask_method", ""))


def _extract_head_pose(record: FrameRecord, obs: Any) -> None:
    """head.pose: 6DRepNet 정밀 yaw/pitch/roll로 face.detect 기하학적 추정값 덮어쓰기.

    _extract_face_detect 이후 호출되어야 한다.
    """
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    estimates = getattr(data, "estimates", None)
    if not estimates:
        return
    est = estimates[0]  # main face
    yaw = getattr(est, "yaw", None)
    if yaw is not None:
        record.head_yaw = float(yaw)
    pitch = getattr(est, "pitch", None)
    if pitch is not None:
        record.head_pitch = float(pitch)
    roll = getattr(est, "roll", None)
    if roll is not None:
        record.head_roll = float(roll)


def _extract_face_gate(record: FrameRecord, obs: Any) -> None:
    """face.gate: per-face gate 판정 결과 추출 (main + passenger)."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    record.gate_passed = bool(getattr(data, "main_gate_passed", True))
    reasons = getattr(data, "main_fail_reasons", ())
    record.gate_fail_reasons = ",".join(reasons) if reasons else ""

    # Passenger gate
    results = getattr(data, "results", [])
    for r in results:
        if getattr(r, "role", "") == "passenger":
            record.passenger_detected = True
            record.passenger_gate_passed = bool(r.gate_passed)
            fail_r = getattr(r, "fail_reasons", ())
            record.passenger_gate_fail_reasons = ",".join(fail_r) if fail_r else ""
            record.passenger_face_area_ratio = float(getattr(r, "face_area_ratio", 0.0))
            record.passenger_head_blur = float(getattr(r, "head_blur", 0.0))
            record.passenger_head_exposure = float(getattr(r, "head_exposure", 0.0))
            break


def _extract_face_baseline(record: FrameRecord, obs: Any) -> None:
    """face.baseline: per-face identity baseline statistics."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    main = getattr(data, "main_profile", None)
    if main is not None and main.n >= 2:
        record.baseline_n = main.n
        record.baseline_area_mean = main.area_ratio_mean
        record.baseline_area_std = main.area_ratio_std


def _extract_portrait_score(record: FrameRecord, obs: Any) -> None:
    """portrait.score: PortraitScoreOutput에서 CLIP portrait 품질 수치 추출."""
    if obs is None:
        return

    data = getattr(obs, "data", None)
    if data is None:
        return

    record.head_aesthetic = float(getattr(data, "head_aesthetic", 0.0))

    # CLIP axis scores (metadata._clip_axes)
    metadata = getattr(obs, "metadata", None) or {}
    clip_axes = metadata.get("_clip_axes")
    if clip_axes:
        for ax in clip_axes:
            field_name = f"clip_{ax.name}"
            if hasattr(record, field_name):
                setattr(record, field_name, float(ax.score))


def _compute_composites(record: FrameRecord) -> None:
    """Cross-analyzer composite signals 계산.

    extract_frame_record() 마지막에 호출.
    모든 개별 필드가 채워진 후 조합.
    """
    # Duchenne smile: disney_smile 분위기 × AU 근육 증거
    # AU6(눈주름) + AU12(입꼬리) → 0~5 스케일 → /5.0으로 정규화 → clamp [0,1]
    au_duchenne = min(1.0, (record.au6_cheek_raiser + record.au12_lip_corner) / 5.0)
    record.duchenne_smile = record.clip_disney_smile * au_duchenne

    # Wild intensity: 함성 분위기 × 실제 입벌림
    au_mouth_open = min(1.0, max(record.au25_lips_part, record.au26_jaw_drop) / 3.0)
    record.wild_intensity = record.clip_wild_roar * au_mouth_open

    # Chill: 무표정 + 모든 CLIP 축 비활성
    axes_max = max(
        record.clip_disney_smile, record.clip_charisma,
        record.clip_wild_roar, record.clip_playful_cute,
    )
    neutral_high = max(0.0, record.em_neutral - 0.5) * 2.0  # 0.5+ → [0,1]
    axes_low = max(0.0, 1.0 - axes_max * 2.0)  # 0.5 이하일 때만 양수
    record.chill_score = neutral_high * axes_low if record.face_detected else 0.0
