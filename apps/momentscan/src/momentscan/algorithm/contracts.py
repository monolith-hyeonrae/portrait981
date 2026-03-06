"""Data contracts — field ownership and consumer dependencies.

Documents which analyzer produces each FrameRecord/CollectionRecord field group,
and which consumers (engines, reports) depend on them. This module enables
runtime validation and serves as living documentation.

Architecture principle:
  vpx = "what do you see?" (objective features, domain-agnostic)
  momentscan = "what does it mean for 981park?" (interpretation, domain-specific)

Usage:
    from momentscan.algorithm.contracts import validate_record_fields
    validate_record_fields(frame_record)  # raises if required fields missing
"""

from __future__ import annotations

from typing import FrozenSet, Protocol, runtime_checkable


# ── Field ownership: analyzer → fields it writes ──

FIELD_SOURCES = {
    "face.detect": frozenset({
        "face_detected", "face_confidence", "face_area_ratio",
        "face_center_distance", "head_yaw", "head_pitch", "head_roll",
        "face_identity",
    }),
    "face.expression": frozenset({
        "eye_open_ratio", "smile_intensity",
        "em_happy", "em_neutral", "em_surprise", "em_angry",
    }),
    "frame.quality": frozenset({
        "blur_score", "brightness", "contrast",
    }),
    "face.classify": frozenset({
        "main_face_confidence",
    }),
    "face.quality": frozenset({
        "face_blur", "face_exposure", "face_contrast",
        "clipped_ratio", "crushed_ratio", "mask_method", "parsing_coverage",
        "seg_face", "seg_eye", "seg_mouth", "seg_hair", "eye_pixel_ratio",
    }),
    "portrait.score": frozenset({
        "head_aesthetic", "clip_axes",
        "catalog_best", "catalog_primary", "catalog_scores",
    }),
    "face.au": frozenset({
        "au1_inner_brow", "au2_outer_brow", "au4_brow_lowerer",
        "au5_upper_lid", "au6_cheek_raiser", "au9_nose_wrinkler",
        "au12_lip_corner", "au15_lip_depressor",
        "au25_lips_part", "au26_jaw_drop",
    }),
    "face.gate": frozenset({
        "gate_passed", "gate_fail_reasons",
        "passenger_detected", "passenger_suitability",
        "passenger_confidence", "passenger_parsing_coverage",
        "passenger_face_area_ratio", "passenger_face_blur",
        "passenger_face_exposure",
    }),
    "face.baseline": frozenset({
        "baseline_n", "baseline_area_mean", "baseline_area_std",
    }),
}


# ── Consumer dependencies: consumer → field groups it reads ──

CONSUMER_DEPS = {
    "BatchHighlightEngine": {
        "required": frozenset({
            "frame_idx", "timestamp_ms", "face_detected", "gate_passed",
        }),
        "quality": frozenset({
            "face_blur", "face_area_ratio", "face_identity",
            "blur_score", "brightness", "contrast",
        }),
        "impact": frozenset({
            "smile_intensity", "head_yaw", "head_aesthetic",
            "clip_axes", "catalog_best",
        }),
    },
    "CollectionEngine": {
        "required": frozenset({
            "frame_idx", "timestamp_ms", "face_detected",
            "head_yaw", "head_pitch",
        }),
        "identity": frozenset({
            "e_id",
        }),
        "scoring": frozenset({
            "gate_passed", "face_blur", "face_identity",
            "catalog_best", "catalog_primary", "catalog_scores",
        }),
    },
}


# ── Protocol interfaces for typed consumers ──

@runtime_checkable
class HasFrameIndex(Protocol):
    """Minimal record interface — frame identification."""
    frame_idx: int
    timestamp_ms: float


@runtime_checkable
class HasFaceDetection(Protocol):
    """Record with face detection fields."""
    face_detected: bool
    face_confidence: float
    face_area_ratio: float


@runtime_checkable
class HasGateResult(Protocol):
    """Record with gate result for quality filtering."""
    gate_passed: bool


@runtime_checkable
class HasHeadPose(Protocol):
    """Record with head pose for pose classification."""
    head_yaw: float
    head_pitch: float
    head_roll: float


# ── Validation helpers ──

def get_fields_for_analyzer(analyzer_name: str) -> FrozenSet[str]:
    """Return fields written by a given analyzer."""
    return FIELD_SOURCES.get(analyzer_name, frozenset())


def get_all_owned_fields() -> FrozenSet[str]:
    """Return all fields with a declared owner."""
    result = set()
    for fields in FIELD_SOURCES.values():
        result |= fields
    return frozenset(result)
