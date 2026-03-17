"""Tests for visualbind.signals."""

from __future__ import annotations

import numpy as np
import pytest

from visualbind.signals import (
    SIGNAL_FIELDS,
    SIGNAL_RANGES,
    _AU_FIELDS,
    _CONFIDENCE_FIELD,
    _DEFAULT_CLIP_AXIS_NAMES,
    _EMOTION_FIELDS,
    _FACE_SIZE_FIELD,
    _NDIM,
    _POSE_FIELDS,
    extract_signal_vector_from_dict,
    get_signal_fields,
    normalize_signal,
)


class TestSignalFields:
    def test_total_dimensions(self):
        """23D base: AU 10 + Emotion 4 + Pose 3 + Confidence 1 + FaceSize 1 + CLIP 4."""
        assert len(SIGNAL_FIELDS) == 23
        assert _NDIM == 23

    def test_field_groups(self):
        assert len(_AU_FIELDS) == 10
        assert len(_EMOTION_FIELDS) == 4
        assert len(_POSE_FIELDS) == 3
        assert len(_CONFIDENCE_FIELD) == 1
        assert len(_FACE_SIZE_FIELD) == 1
        assert len(_DEFAULT_CLIP_AXIS_NAMES) == 4

    def test_signal_fields_composition(self):
        expected = (
            _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS
            + _CONFIDENCE_FIELD + _FACE_SIZE_FIELD
            + _DEFAULT_CLIP_AXIS_NAMES
        )
        assert SIGNAL_FIELDS == expected

    def test_no_duplicate_fields(self):
        assert len(SIGNAL_FIELDS) == len(set(SIGNAL_FIELDS))

    def test_all_fixed_fields_have_ranges(self):
        for f in _AU_FIELDS + _EMOTION_FIELDS + _POSE_FIELDS + _CONFIDENCE_FIELD + _FACE_SIZE_FIELD:
            assert f in SIGNAL_RANGES, f"Missing range for {f}"


class TestGetSignalFields:
    def test_default(self):
        fields = get_signal_fields()
        assert fields == SIGNAL_FIELDS

    def test_custom_clip_axes(self):
        fields = get_signal_fields(["axis_a", "axis_b"])
        assert fields[-2:] == ("axis_a", "axis_b")
        # AU + Emotion + Pose + Confidence + FaceSize + 2 custom
        assert len(fields) == 10 + 4 + 3 + 1 + 1 + 2


class TestNormalizeSignal:
    def test_au_midpoint(self):
        assert normalize_signal(2.5, "au1_inner_brow") == pytest.approx(0.5)

    def test_au_clamp_high(self):
        assert normalize_signal(10.0, "au1_inner_brow") == pytest.approx(1.0)

    def test_au_clamp_low(self):
        assert normalize_signal(-1.0, "au1_inner_brow") == pytest.approx(0.0)

    def test_emotion(self):
        assert normalize_signal(0.5, "em_happy") == pytest.approx(0.5)

    def test_pose_symmetric(self):
        # head_pitch: (-30, 30) -> 0 maps to 0.5
        assert normalize_signal(0.0, "head_pitch") == pytest.approx(0.5)

    def test_confidence(self):
        assert normalize_signal(0.8, "detect_confidence") == pytest.approx(0.8)

    def test_face_size(self):
        assert normalize_signal(0.3, "face_size_ratio") == pytest.approx(0.3)

    def test_unknown_field_uses_clip_range(self):
        # Unknown field defaults to (0, 1) range
        assert normalize_signal(0.7, "custom_axis") == pytest.approx(0.7)


class TestExtractSignalVectorFromDict:
    def test_empty_dict(self):
        vec = extract_signal_vector_from_dict({})
        assert vec.shape == (_NDIM,)
        # head_pitch with raw=0.0 and range (-30, 30) -> 0.5
        pitch_idx = SIGNAL_FIELDS.index("head_pitch")
        assert vec[pitch_idx] == pytest.approx(0.5)

    def test_partial_dict(self):
        signals = {"au1_inner_brow": 5.0, "em_happy": 1.0}
        vec = extract_signal_vector_from_dict(signals)
        assert vec[0] == pytest.approx(1.0)  # au1 = 5/5 = 1.0
        assert vec[SIGNAL_FIELDS.index("em_happy")] == pytest.approx(1.0)

    def test_custom_fields(self):
        fields = ("em_happy", "em_neutral")
        signals = {"em_happy": 0.5, "em_neutral": 0.8}
        vec = extract_signal_vector_from_dict(signals, signal_fields=fields)
        assert vec.shape == (2,)
        assert vec[0] == pytest.approx(0.5)
        assert vec[1] == pytest.approx(0.8)
