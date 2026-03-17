"""Tests for visualbind.signals."""

from __future__ import annotations

import numpy as np
import pytest

from visualbind.signals import (
    SIGNAL_FIELDS,
    SIGNAL_FIELDS_EXTENDED,
    SIGNAL_FIELDS_LEGACY,
    SIGNAL_RANGES,
    _AU_FIELDS,
    _DETECTION_FIELDS,
    _DEFAULT_CLIP_AXIS_NAMES,
    _EMOTION_FIELDS,
    _FACE_QUALITY_FIELDS,
    _FRAME_QUALITY_FIELDS,
    _SEGMENTATION_FIELDS,
    _COMPOSITE_FIELDS,
    _NDIM,
    _POSE_FIELDS,
    extract_signal_vector_from_dict,
    get_signal_fields,
    normalize_signal,
)


class TestSignalFields:
    def test_extended_dimensions(self):
        """49D: AU12 + Emotion8 + Pose3 + Det3 + FaceQ5 + FrameQ3 + Seg4 + Comp3 + CLIP4."""
        assert len(SIGNAL_FIELDS_EXTENDED) == 45
        assert SIGNAL_FIELDS is SIGNAL_FIELDS_EXTENDED

    def test_legacy_dimensions(self):
        """21D legacy: AU10 + Emotion4 + Pose3 + CLIP4."""
        assert len(SIGNAL_FIELDS_LEGACY) == 21

    def test_field_groups(self):
        assert len(_AU_FIELDS) == 12
        assert len(_EMOTION_FIELDS) == 8
        assert len(_POSE_FIELDS) == 3
        assert len(_DETECTION_FIELDS) == 3
        assert len(_FACE_QUALITY_FIELDS) == 5
        assert len(_FRAME_QUALITY_FIELDS) == 3
        assert len(_SEGMENTATION_FIELDS) == 4
        assert len(_COMPOSITE_FIELDS) == 3
        assert len(_DEFAULT_CLIP_AXIS_NAMES) == 4

    def test_signal_fields_composition(self):
        expected = (
            _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
            + _DETECTION_FIELDS + _FACE_QUALITY_FIELDS
            + _FRAME_QUALITY_FIELDS + _SEGMENTATION_FIELDS
            + _COMPOSITE_FIELDS + _DEFAULT_CLIP_AXIS_NAMES
        )
        assert SIGNAL_FIELDS_EXTENDED == expected

    def test_no_duplicate_fields(self):
        assert len(SIGNAL_FIELDS) == len(set(SIGNAL_FIELDS))
        assert len(SIGNAL_FIELDS_LEGACY) == len(set(SIGNAL_FIELDS_LEGACY))

    def test_all_fixed_fields_have_ranges(self):
        fixed = (
            _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
            + _DETECTION_FIELDS + _FACE_QUALITY_FIELDS
            + _FRAME_QUALITY_FIELDS + _SEGMENTATION_FIELDS
            + _COMPOSITE_FIELDS
        )
        for f in fixed:
            assert f in SIGNAL_RANGES, f"Missing range for {f}"


class TestGetSignalFields:
    def test_default_extended(self):
        fields = get_signal_fields()
        assert len(fields) == 45

    def test_default_legacy(self):
        fields = get_signal_fields(extended=False)
        assert len(fields) == 21

    def test_custom_clip_axes(self):
        fields = get_signal_fields(clip_axis_names=["axis_a", "axis_b"])
        # 45 base (without CLIP) + 2 custom
        assert fields[-2:] == ("axis_a", "axis_b")


class TestNormalizeSignal:
    def test_au_midpoint(self):
        assert normalize_signal(2.5, "au1_inner_brow") == pytest.approx(0.5)

    def test_au_clamp_high(self):
        assert normalize_signal(10.0, "au6_cheek_raiser") == 1.0

    def test_au_clamp_low(self):
        assert normalize_signal(-1.0, "au12_lip_corner") == 0.0

    def test_emotion(self):
        assert normalize_signal(0.5, "em_happy") == pytest.approx(0.5)

    def test_pose_symmetric(self):
        assert normalize_signal(0.0, "head_pitch") == pytest.approx(0.5)

    def test_confidence(self):
        assert normalize_signal(0.8, "face_confidence") == pytest.approx(0.8)

    def test_face_blur(self):
        assert normalize_signal(250.0, "face_blur") == pytest.approx(0.5)

    def test_brightness(self):
        assert normalize_signal(127.5, "brightness") == pytest.approx(0.5)

    def test_segmentation(self):
        assert normalize_signal(0.5, "seg_face") == pytest.approx(0.5)

    def test_unknown_field_uses_clip_range(self):
        assert normalize_signal(0.5, "custom_axis") == pytest.approx(0.5)


class TestExtractSignalVectorFromDict:
    def test_empty_dict(self):
        vec = extract_signal_vector_from_dict({})
        assert vec.shape == (_NDIM,)
        # Some fields normalize 0 to non-zero (e.g. head_pitch range -30~30 → 0 maps to 0.5)
        assert vec.shape[0] == 45

    def test_partial_dict(self):
        vec = extract_signal_vector_from_dict({"em_happy": 0.5})
        idx = list(SIGNAL_FIELDS).index("em_happy")
        assert vec[idx] == pytest.approx(0.5)

    def test_custom_fields(self):
        fields = ("em_happy", "em_angry")
        vec = extract_signal_vector_from_dict(
            {"em_happy": 0.8, "em_angry": 0.2}, signal_fields=fields,
        )
        assert vec.shape == (2,)
        assert vec[0] == pytest.approx(0.8)
        assert vec[1] == pytest.approx(0.2)
