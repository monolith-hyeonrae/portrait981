"""FlowData → FrameRecord 변환.

on_frame() 콜백에서 호출되어,
각 vpx analyzer의 Observation 결과를 FrameRecord 수치로 추출한다.

데이터 소스:
- face.detect: FaceDetectOutput.faces[i] (FaceObservation 데이터 객체)
- face.expression: ExpressionOutput.faces[i].signals (per-face emotion/AU)
- body.pose: PoseOutput.keypoints[i] (COCO 17 keypoints)
- frame.quality: Observation.signals (직접)
- face.classify: Observation.signals (직접)
- frame.scoring: Observation.signals (직접)
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np

from momentscan.algorithm.batch.types import FrameRecord

# COCO keypoint indices
_NOSE = 0
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10

# Module-level state for ArcFace anchor (quality)
_anchor_embedding: Optional[np.ndarray] = None
_anchor_confidence: float = 0.0
_ANCHOR_DECAY: float = 0.998  # per-frame decay → half-life ~346 frames

# Module-level state for DINOv2 dual-EMA (impact)
# fast EMA tracks recent changes, slow EMA provides stable baseline.
# delta = 1 - dot(fast, slow) captures sustained change, filters noise.
_EMBED_ALPHA_FAST: float = 0.3   # ~3 frame response
_EMBED_ALPHA_SLOW: float = 0.05  # ~20 frame response
_embed_fast_face: Optional[np.ndarray] = None
_embed_slow_face: Optional[np.ndarray] = None
_embed_fast_body: Optional[np.ndarray] = None
_embed_slow_body: Optional[np.ndarray] = None


def reset_extract_state() -> None:
    """비디오 간 상태 격리를 위한 모듈 레벨 상태 리셋."""
    global _anchor_embedding, _anchor_confidence
    global _embed_fast_face, _embed_slow_face, _embed_fast_body, _embed_slow_body
    _anchor_embedding = None
    _anchor_confidence = 0.0
    _embed_fast_face = None
    _embed_slow_face = None
    _embed_fast_body = None
    _embed_slow_body = None


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

    # 주 얼굴 중심 → body pose에서 인물 매칭용
    face_center = _get_main_face_center(face_obs)
    _extract_body_pose(record, obs_by_source.get("body.pose"), face_center)
    _extract_quality(record, obs_by_source.get("frame.quality"))
    _extract_face_classify(record, obs_by_source.get("face.classify"))
    _extract_frame_scoring(record, obs_by_source.get("frame.scoring"))
    _extract_face_embed(record, obs_by_source.get("face.embed"))
    _extract_body_embed(record, obs_by_source.get("body.embed"))

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


def _get_main_face_center(face_obs: Any) -> Optional[tuple[float, float]]:
    """주 얼굴 bbox 중심을 (x, y) 정규화 좌표로 반환."""
    face = _get_main_face(face_obs)
    if face is None:
        return None
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return None
    nx, ny, nw, nh = bbox
    return (nx + nw / 2.0, ny + nh / 2.0)


def _match_pose_to_face(
    kp_list: list,
    face_center: Optional[tuple[float, float]],
) -> Optional[dict]:
    """얼굴 중심에 가장 가까운 pose의 person dict를 반환.

    face_center가 없으면 첫 번째 person을 반환 (fallback).
    매칭 기준: nose 키포인트(index 0)와 face bbox 중심의 거리.
    """
    if not kp_list:
        return None
    if len(kp_list) == 1 or face_center is None:
        return kp_list[0]

    fx, fy = face_center
    best_person = None
    best_dist = float("inf")

    for person in kp_list:
        kpts = person.get("keypoints", [])
        img_size = person.get("image_size", (1, 1))
        img_w = float(img_size[0]) if img_size[0] > 0 else 1.0
        img_h = float(img_size[1]) if img_size[1] > 0 else 1.0

        # nose keypoint (index 0)
        if len(kpts) > 0:
            pt = kpts[0]
            if len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                # 정규화 좌표로 변환
                nx = pt[0] / img_w
                ny = pt[1] / img_h
                dist = math.hypot(nx - fx, ny - fy)
                if dist < best_dist:
                    best_dist = dist
                    best_person = person

    return best_person if best_person is not None else kp_list[0]


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
    """face.expression: ExpressionOutput.faces에서 표정 수치 추출.

    expression analyzer는 face.detect의 FaceObservation을 업데이트하여
    expression intensity와 per-face signals (em_*, au_*)를 추가한다.
    """
    if obs is None:
        return

    face = _get_main_face(obs)
    if face is not None:
        face_signals = getattr(face, "signals", {}) or {}
        # expression intensity as mouth_open proxy
        # (HSEmotion의 expression_intensity는 비중립 표정 강도)
        record.mouth_open_ratio = float(getattr(face, "expression", 0.0))
        # eye_open: 직접 제공되지 않음, 비중립도를 proxy로 사용
        neutral = float(face_signals.get("em_neutral", 1.0))
        record.eye_open_ratio = 1.0 - neutral
        # smile
        record.smile_intensity = float(face_signals.get("em_happy", 0.0))
        return

    # Fallback: frame-level signals
    signals = getattr(obs, "signals", {}) or {}
    record.mouth_open_ratio = float(signals.get("max_expression", 0.0))
    record.smile_intensity = float(signals.get("expression_happy", 0.0))


def _extract_body_pose(
    record: FrameRecord,
    obs: Any,
    face_center: Optional[tuple[float, float]] = None,
) -> None:
    """body.pose: PoseOutput.keypoints에서 포즈 수치를 계산.

    COCO keypoints [x, y, confidence] 배열에서
    wrist_raise, torso_rotation, hand_near_face를 직접 계산한다.
    주 얼굴 중심과 가장 가까운 pose를 매칭하여 사용.
    """
    if obs is None:
        return

    output = getattr(obs, "data", None)
    if output is None:
        return
    kp_list = getattr(output, "keypoints", [])
    if not kp_list:
        return

    person = _match_pose_to_face(kp_list, face_center)
    kpts = person.get("keypoints", [])
    if len(kpts) < 11:
        return

    image_size = person.get("image_size", (1, 1))
    img_h = float(image_size[1]) if image_size[1] > 0 else 1.0

    # Confidence threshold
    min_conf = 0.3

    def _kp(idx):
        """keypoint (x, y, conf) 반환. 없으면 None."""
        if idx >= len(kpts):
            return None
        pt = kpts[idx]
        if len(pt) < 3 or pt[2] < min_conf:
            return None
        return pt

    nose = _kp(_NOSE)
    l_shoulder = _kp(_LEFT_SHOULDER)
    r_shoulder = _kp(_RIGHT_SHOULDER)
    l_wrist = _kp(_LEFT_WRIST)
    r_wrist = _kp(_RIGHT_WRIST)

    # wrist_raise: 손목이 어깨보다 높은 정도 (y축 반전: 위가 작은 값)
    wrist_raise = 0.0
    if l_shoulder and l_wrist:
        raise_l = max(0.0, l_shoulder[1] - l_wrist[1]) / img_h
        wrist_raise = max(wrist_raise, raise_l)
    if r_shoulder and r_wrist:
        raise_r = max(0.0, r_shoulder[1] - r_wrist[1]) / img_h
        wrist_raise = max(wrist_raise, raise_r)
    record.wrist_raise = wrist_raise

    # torso_rotation: 어깨 라인의 기울기 (수평=0, 기울어질수록 큰 값)
    if l_shoulder and r_shoulder:
        dx = r_shoulder[0] - l_shoulder[0]
        dy = r_shoulder[1] - l_shoulder[1]
        # 수평 기준 절대 각도 (degrees)
        record.torso_rotation = abs(math.degrees(math.atan2(dy, dx))) if dx != 0 else 0.0

    # hand_near_face: 손목과 코 사이 거리의 역수 (가까울수록 1에 가까움)
    if nose is not None:
        dists = []
        for wrist in (l_wrist, r_wrist):
            if wrist is not None:
                d = math.hypot(wrist[0] - nose[0], wrist[1] - nose[1]) / img_h
                dists.append(d)
        if dists:
            # 0~1 범위로 클램프 (가까울수록 높은 값)
            record.hand_near_face = max(0.0, 1.0 - min(dists))

    # elbow_angle_change: 팔꿈치 각도 (raw value, delta는 BatchHighlightEngine에서)
    l_elbow = _kp(_LEFT_ELBOW)
    r_elbow = _kp(_RIGHT_ELBOW)
    angles = []
    if l_shoulder and l_elbow and l_wrist:
        angles.append(_elbow_angle(l_shoulder, l_elbow, l_wrist))
    if r_shoulder and r_elbow and r_wrist:
        angles.append(_elbow_angle(r_shoulder, r_elbow, r_wrist))
    if angles:
        record.elbow_angle_change = sum(angles) / len(angles)


def _elbow_angle(shoulder, elbow, wrist) -> float:
    """어깨-팔꿈치-손목 세 점의 팔꿈치 각도(degrees)를 계산한다."""
    ax, ay = shoulder[0] - elbow[0], shoulder[1] - elbow[1]
    bx, by = wrist[0] - elbow[0], wrist[1] - elbow[1]
    dot = ax * bx + ay * by
    mag_a = math.hypot(ax, ay)
    mag_b = math.hypot(bx, by)
    if mag_a < 1e-8 or mag_b < 1e-8:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_angle))


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


def _extract_frame_scoring(record: FrameRecord, obs: Any) -> None:
    """frame.scoring: Observation.signals에서 프레임 점수 추출."""
    if obs is None:
        return

    signals = getattr(obs, "signals", {}) or {}
    record.frame_score = float(signals.get("total_score", 0.0))


def _update_dual_ema(
    emb: np.ndarray,
    fast: Optional[np.ndarray],
    slow: Optional[np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Dual-EMA 업데이트 후 delta 반환.

    fast EMA는 최근 변화를 빠르게 추적, slow EMA는 안정 베이스라인.
    delta = 1 - dot(fast, slow): 지속적 변화만 포착, 순간 노이즈 필터.
    """
    af, asl = _EMBED_ALPHA_FAST, _EMBED_ALPHA_SLOW

    if fast is None:
        return 0.0, emb.copy(), emb.copy()

    new_fast = af * emb + (1.0 - af) * fast
    norm_f = np.linalg.norm(new_fast)
    if norm_f > 1e-8:
        new_fast = new_fast / norm_f

    new_slow = asl * emb + (1.0 - asl) * slow
    norm_s = np.linalg.norm(new_slow)
    if norm_s > 1e-8:
        new_slow = new_slow / norm_s

    delta = max(0.0, 1.0 - float(np.dot(new_fast, new_slow)))
    return delta, new_fast, new_slow


def _extract_face_embed(record: FrameRecord, obs: Any) -> None:
    """face.embed: DINOv2 face dual-EMA delta 계산."""
    global _embed_fast_face, _embed_slow_face

    if obs is None:
        return

    data = getattr(obs, "data", None)
    if data is None:
        return

    e_face = getattr(data, "e_face", None)
    if e_face is not None:
        e_face = np.asarray(e_face, dtype=np.float32)
        delta, _embed_fast_face, _embed_slow_face = _update_dual_ema(
            e_face, _embed_fast_face, _embed_slow_face
        )
        record.face_change = delta


def _extract_body_embed(record: FrameRecord, obs: Any) -> None:
    """body.embed: DINOv2 body dual-EMA delta 계산."""
    global _embed_fast_body, _embed_slow_body

    if obs is None:
        return

    data = getattr(obs, "data", None)
    if data is None:
        return

    e_body = getattr(data, "e_body", None)
    if e_body is not None:
        e_body = np.asarray(e_body, dtype=np.float32)
        delta, _embed_fast_body, _embed_slow_body = _update_dual_ema(
            e_body, _embed_fast_body, _embed_slow_body
        )
        record.body_change = delta
