"""FlowData -> CollectionRecord unified extraction.

Replaces the dual extract_frame_record() + extract_identity_record()
with a single function that captures both signal fields and ArcFace embedding.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from momentscan.algorithm.collection.types import CollectionRecord

# Module-level state for ArcFace anchor (quality signal)
_anchor_embedding: Optional[np.ndarray] = None
_anchor_confidence: float = 0.0
_ANCHOR_DECAY: float = 0.998

# Module-level state for signal-profile catalog scoring
_catalog_profiles: list = []

# Module-level state for visualbind TreeStrategy (trained model)
_bind_strategy: object = None


def set_catalog_profiles(profiles: list) -> None:
    """Set signal-profile catalog for catalog scoring."""
    global _catalog_profiles
    _catalog_profiles = list(profiles)


def set_bind_strategy(strategy: object) -> None:
    """Set trained visualbind TreeStrategy for bind scoring."""
    global _bind_strategy
    _bind_strategy = strategy


def reset_extract_state() -> None:
    """Reset module-level state for per-video isolation."""
    global _anchor_embedding, _anchor_confidence, _catalog_profiles, _bind_strategy
    _anchor_embedding = None
    _anchor_confidence = 0.0
    _catalog_profiles = []
    _bind_strategy = None


def extract_collection_record(
    frame: Any, results: List[Any]
) -> Optional[CollectionRecord]:
    """Extract a CollectionRecord from a frame's analyzer results.

    Combines signal extraction (from batch/extract.py) with
    ArcFace embedding extraction (from identity/extract.py).

    Returns None if no analyzer results available.
    """
    if not results:
        return None

    obs_by_source: dict[str, Any] = {}
    for flow_data in results:
        for obs in getattr(flow_data, "observations", []):
            source = getattr(obs, "source", None)
            if source:
                obs_by_source[source] = obs

    record = CollectionRecord(
        frame_idx=getattr(frame, "frame_id", 0),
        timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
    )

    _extract_face_detect(record, obs_by_source.get("face.detect"))
    _extract_head_pose(record, obs_by_source.get("head.pose"))
    _extract_face_expression(record, obs_by_source.get("face.expression"))
    _extract_face_au(record, obs_by_source.get("face.au"))
    _extract_quality(record, obs_by_source.get("frame.quality"))
    _extract_face_quality(record, obs_by_source.get("face.quality"))
    _extract_portrait_score(record, obs_by_source.get("portrait.score"))
    _extract_face_gate(record, obs_by_source.get("face.gate"))
    _extract_person_id(record, obs_by_source.get("face.classify"))

    # Signal-profile catalog scoring (override mode)
    if _catalog_profiles:
        _apply_catalog_scoring(record)

    # Visualbind TreeStrategy scoring (trained XGBoost model)
    if _bind_strategy is not None:
        _apply_bind_scoring(record)

    return record


# ── Per-analyzer extractors ──


def _get_main_face(obs: Any) -> Any:
    """Get main face (largest area_ratio) from Observation.data."""
    output = getattr(obs, "data", None)
    if output is None:
        return None
    faces = getattr(output, "faces", None)
    if not faces:
        return None
    return max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))


def _extract_face_detect(record: CollectionRecord, obs: Any) -> None:
    """face.detect: face features + ArcFace embedding."""
    global _anchor_embedding, _anchor_confidence

    if obs is None:
        return

    face = _get_main_face(obs)
    if face is None:
        return

    record.face_detected = True
    record.face_confidence = float(getattr(face, "confidence", 0.0))
    record.face_area_ratio = float(getattr(face, "area_ratio", 0.0))
    record.head_yaw = float(getattr(face, "yaw", 0.0))
    record.head_pitch = float(getattr(face, "pitch", 0.0))
    record.head_roll = float(getattr(face, "roll", 0.0))
    record.face_bbox = getattr(face, "bbox", None)

    # ArcFace embedding
    embedding = getattr(face, "embedding", None)
    if embedding is not None:
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
            record.e_id = emb

            # Anchor tracking for face_identity quality signal
            _anchor_confidence *= _ANCHOR_DECAY
            if record.face_confidence > _anchor_confidence:
                _anchor_embedding = emb
                _anchor_confidence = record.face_confidence

            if _anchor_embedding is not None:
                sim = float(np.dot(emb, _anchor_embedding))
                record.face_identity = max(0.0, sim)


def _extract_head_pose(record: CollectionRecord, obs: Any) -> None:
    """head.pose: precise yaw/pitch/roll (overrides geometric estimate)."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    estimates = getattr(data, "estimates", None)
    if not estimates:
        return
    est = estimates[0]
    yaw = getattr(est, "yaw", None)
    if yaw is not None:
        record.head_yaw = float(yaw)
    pitch = getattr(est, "pitch", None)
    if pitch is not None:
        record.head_pitch = float(pitch)
    roll = getattr(est, "roll", None)
    if roll is not None:
        record.head_roll = float(roll)


def _extract_face_expression(record: CollectionRecord, obs: Any) -> None:
    """face.expression: smile, eye_open, mouth_open."""
    if obs is None:
        return

    face = _get_main_face(obs)
    if face is not None:
        face_signals = getattr(face, "signals", {}) or {}
        record.mouth_open_ratio = float(getattr(face, "expression", 0.0))
        neutral = float(face_signals.get("em_neutral", 1.0))
        record.em_neutral = neutral
        record.eye_open_ratio = 1.0 - neutral
        record.smile_intensity = float(face_signals.get("em_happy", 0.0))
        record.em_happy = float(face_signals.get("em_happy", 0.0))
        record.em_surprise = float(face_signals.get("em_surprise", 0.0))
        record.em_angry = float(face_signals.get("em_angry", 0.0))
        return

    signals = getattr(obs, "signals", {}) or {}
    record.smile_intensity = float(signals.get("expression_happy", 0.0))


def _extract_face_au(record: CollectionRecord, obs: Any) -> None:
    """face.au: AU intensities + smile correction (max strategy)."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    au_list = getattr(data, "au_intensities", None)
    if not au_list:
        return
    au = au_list[0]

    record.au_intensities = au
    record.au1_inner_brow = float(au.get("AU1", 0.0))
    record.au2_outer_brow = float(au.get("AU2", 0.0))
    record.au4_brow_lowerer = float(au.get("AU4", 0.0))
    record.au5_upper_lid = float(au.get("AU5", 0.0))
    record.au6_cheek_raiser = float(au.get("AU6", 0.0))
    record.au9_nose_wrinkler = float(au.get("AU9", 0.0))
    record.au12_lip_corner = float(au.get("AU12", 0.0))
    record.au15_lip_depressor = float(au.get("AU15", 0.0))
    record.au25_lips_part = float(au.get("AU25", 0.0))
    record.au26_jaw_drop = float(au.get("AU26", 0.0))

    au12 = au.get("AU12", None)
    if au12 is not None:
        smile_from_au12 = min(1.0, float(au12) / 3.0)
        record.smile_intensity = max(record.smile_intensity, smile_from_au12)


def _extract_quality(record: CollectionRecord, obs: Any) -> None:
    """frame.quality: blur, brightness."""
    if obs is None:
        return
    signals = getattr(obs, "signals", {}) or {}
    record.blur_score = float(signals.get("blur_score", 0.0))


def _extract_face_quality(record: CollectionRecord, obs: Any) -> None:
    """face.quality: face region metrics + crop coords."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    record.face_blur = float(getattr(data, "face_blur", 0.0))
    record.face_crop_box = getattr(data, "face_crop_box", None)
    if record.image_size is None:
        record.image_size = getattr(data, "image_size", None)


def _extract_portrait_score(record: CollectionRecord, obs: Any) -> None:
    """portrait.score: CLIP axes → dynamic dict."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return

    metadata = getattr(obs, "metadata", None) or {}
    clip_axes = metadata.get("_clip_axes")
    if clip_axes:
        for ax in clip_axes:
            record.clip_axes[ax.name] = float(ax.score)


def _extract_face_gate(record: CollectionRecord, obs: Any) -> None:
    """face.gate: gate result + passenger suitability."""
    if obs is None:
        return
    data = getattr(obs, "data", None)
    if data is None:
        return
    record.gate_passed = bool(getattr(data, "main_gate_passed", True))

    results = getattr(data, "results", [])
    for r in results:
        if getattr(r, "role", "") == "passenger":
            record.passenger_suitability = float(getattr(r, "suitability", 0.0))
            break


def _extract_person_id(record: CollectionRecord, obs: Any) -> None:
    """face.classify: person_id."""
    if obs is None:
        return
    signals = getattr(obs, "signals", {}) or {}
    main_conf = float(signals.get("main_confidence", 0.0))
    if main_conf > 0:
        record.person_id = 0


def _apply_catalog_scoring(record: CollectionRecord) -> None:
    """Signal-profile catalog scoring."""
    from momentscan.algorithm.batch.catalog_scoring import compute_catalog_scores
    compute_catalog_scores(record, _catalog_profiles)


def _apply_bind_scoring(record: CollectionRecord) -> None:
    """Visualbind TreeStrategy scoring."""
    from visualbind.signals import SIGNAL_FIELDS, normalize_signal
    import numpy as np

    vec = np.zeros(len(SIGNAL_FIELDS), dtype=np.float64)
    for i, f in enumerate(SIGNAL_FIELDS):
        if f == "head_yaw_dev":
            raw = abs(float(getattr(record, "head_yaw", 0.0)))
        elif hasattr(record, 'composites') and f in record.composites:
            raw = float(record.composites.get(f, 0.0))
        elif hasattr(record, 'clip_axes') and f in record.clip_axes:
            raw = float(record.clip_axes.get(f, 0.0))
        elif hasattr(record, f):
            raw = float(getattr(record, f, 0.0))
        else:
            raw = 0.0
        vec[i] = normalize_signal(raw, f)

    scores = _bind_strategy.predict(vec)
    if scores:
        record.bind_scores = scores
        best_name = max(scores, key=scores.get)
        record.bind_best = scores[best_name]
        record.bind_primary = best_name
