"""Observer output binding — 다중 analyzer Observation을 단일 signal dict로 결합.

vpx/momentscan analyzer들이 각자의 이름 규칙으로 출력하는 signal을
visualbind 표준 49D signal dict로 변환.

이것이 visualbind의 핵심 역할: 여러 observer를 결합(bind)한다.

Usage:
    from visualbind.observer_bind import bind_observations
    signals = bind_observations(observations)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("visualbind.observer_bind")

# ── Signal key mappings (vpx 출력 → visualbind 표준) ──

_AU_KEY_MAP = {
    "au_au1": "au1_inner_brow", "au_au2": "au2_outer_brow",
    "au_au4": "au4_brow_lowerer", "au_au5": "au5_upper_lid",
    "au_au6": "au6_cheek_raiser", "au_au9": "au9_nose_wrinkler",
    "au_au12": "au12_lip_corner", "au_au15": "au15_lip_depressor",
    "au_au17": "au17_chin_raiser", "au_au20": "au20_lip_stretcher",
    "au_au25": "au25_lips_part", "au_au26": "au26_jaw_drop",
}

_EXPR_KEY_MAP = {
    "expression_happy": "em_happy", "expression_neutral": "em_neutral",
    "expression_surprise": "em_surprise", "expression_angry": "em_angry",
    "expression_contempt": "em_contempt", "expression_disgust": "em_disgust",
    "expression_fear": "em_fear", "expression_sad": "em_sad",
}

# BiSeNet 19-class segmentation groups
_SEG_CLASSES = {
    "face": [1], "eye": [4, 5], "mouth": [11, 12, 13],
    "mouth_in": [11], "hair": [17], "glasses": [6],
    "brow": [2, 3], "nose": [10],
}


def bind_observations(observations: list[Any]) -> dict[str, float]:
    """다중 analyzer Observation을 단일 signal dict로 결합.

    Args:
        observations: visualpath Path.analyze_all()의 결과 리스트.

    Returns:
        visualbind 표준 49D signal dict (raw values, 정규화 전).
    """
    signals: dict[str, float] = {}

    for obs in observations:
        src = getattr(obs, "source", "")
        obs_signals = getattr(obs, "signals", {}) or {}
        obs_data = getattr(obs, "data", None)

        if src == "face.detect":
            _bind_face_detect(signals, obs_data)
        elif src == "face.au":
            _bind_mapped_signals(signals, obs_signals, _AU_KEY_MAP)
        elif src == "face.expression":
            _bind_mapped_signals(signals, obs_signals, _EXPR_KEY_MAP)
        elif src == "head.pose":
            _bind_head_pose(signals, obs_signals)
        elif src == "face.parse":
            _bind_face_parse(signals, obs_data)
        elif src == "face.quality":
            _bind_face_quality(signals, obs_signals)
        elif src == "frame.quality":
            _bind_direct(signals, obs_signals, ["blur_score", "brightness", "contrast"])

    # Derived signals
    _compute_derived(signals)

    return signals


def _bind_face_detect(signals: dict, data: Any) -> None:
    """face.detect → detection signals from main face."""
    if data is None:
        return
    faces = getattr(data, "faces", [])
    if not faces:
        return
    face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
    signals["face_confidence"] = float(face.confidence)
    signals["face_area_ratio"] = float(face.area_ratio)
    signals["face_center_distance"] = float(face.center_distance)
    signals["head_yaw_dev"] = abs(float(getattr(face, "yaw", 0.0)))
    signals["head_pitch"] = float(getattr(face, "pitch", 0.0))
    signals["head_roll"] = float(getattr(face, "roll", 0.0))


def _bind_mapped_signals(signals: dict, obs_signals: dict, key_map: dict) -> None:
    """Key mapping을 적용하여 signals에 추가."""
    for src_key, dst_key in key_map.items():
        if src_key in obs_signals:
            signals[dst_key] = float(obs_signals[src_key])


def _bind_head_pose(signals: dict, obs_signals: dict) -> None:
    """head.pose → override geometric estimate with precise values."""
    if "head_yaw" in obs_signals:
        signals["head_yaw_dev"] = abs(float(obs_signals["head_yaw"]))
    if "head_pitch" in obs_signals:
        signals["head_pitch"] = float(obs_signals["head_pitch"])
    if "head_roll" in obs_signals:
        signals["head_roll"] = float(obs_signals["head_roll"])


def _bind_face_quality(signals: dict, obs_signals: dict) -> None:
    """face.quality → mask-based quality metrics."""
    for key in ("face_blur", "face_exposure", "face_contrast",
                "clipped_ratio", "crushed_ratio"):
        if key in obs_signals:
            signals[key] = float(obs_signals[key])


def _bind_face_parse(signals: dict, data: Any) -> None:
    """face.parse → segmentation ratios from BiSeNet class_map."""
    if data is None:
        return
    results = getattr(data, "results", [])
    if not results:
        return
    class_map = getattr(results[0], "class_map", None)
    if class_map is None:
        return

    total = class_map.size
    if total == 0:
        return

    face_px = int(np.isin(class_map, _SEG_CLASSES["face"]).sum())
    eye_px = int(np.isin(class_map, _SEG_CLASSES["eye"]).sum())
    mouth_px = int(np.isin(class_map, _SEG_CLASSES["mouth"]).sum())
    mouth_in_px = int(np.isin(class_map, _SEG_CLASSES["mouth_in"]).sum())
    hair_px = int(np.isin(class_map, _SEG_CLASSES["hair"]).sum())
    glasses_px = int(np.isin(class_map, _SEG_CLASSES["glasses"]).sum())
    brow_px = int(np.isin(class_map, _SEG_CLASSES["brow"]).sum())
    nose_px = int(np.isin(class_map, _SEG_CLASSES["nose"]).sum())

    signals["seg_face"] = float(face_px / total)
    signals["seg_eye"] = float(eye_px / total)
    signals["seg_mouth"] = float(mouth_px / total)
    signals["seg_hair"] = float(hair_px / total)

    face_area = max(face_px + eye_px + brow_px + nose_px + mouth_px, 1)
    signals["eye_visible_ratio"] = float(eye_px / face_area)
    signals["mouth_open_ratio"] = float(mouth_in_px / face_area)
    signals["glasses_ratio"] = float(glasses_px / face_area)


def _bind_direct(signals: dict, obs_signals: dict, keys: list[str]) -> None:
    """직접 매핑 (이름 변환 없음)."""
    for key in keys:
        if key in obs_signals:
            signals[key] = float(obs_signals[key])


def _compute_derived(signals: dict) -> None:
    """Derived/composite signals 계산."""
    # Backlight
    signals["backlight_score"] = max(0.0,
        signals.get("brightness", 0.0) - signals.get("face_exposure", 0.0))

    # CLIP placeholders
    for axis in ("warm_smile", "cool_gaze", "playful_face", "wild_energy"):
        signals.setdefault(axis, 0.0)

    # Composites
    au6 = signals.get("au6_cheek_raiser", 0.0)
    au12 = signals.get("au12_lip_corner", 0.0)
    signals["duchenne_smile"] = (au6 + au12) / 5.0 * signals.get("warm_smile", 0.0)
    signals["wild_intensity"] = max(
        signals.get("au25_lips_part", 0.0),
        signals.get("au26_jaw_drop", 0.0)) / 3.0 * signals.get("wild_energy", 0.0)
    clip_max = max(signals.get(a, 0.0) for a in
                   ("warm_smile", "cool_gaze", "playful_face", "wild_energy"))
    signals["chill_score"] = signals.get("em_neutral", 0.0) * (1.0 - clip_max)
