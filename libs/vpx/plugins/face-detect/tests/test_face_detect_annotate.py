"""Tests for FaceDetectionAnalyzer.annotate()."""

from vpx.sdk import Observation
from vpx.sdk.marks import BBoxMark
from vpx.face_detect.analyzer import FaceDetectionAnalyzer
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput


def _make_face(face_id, confidence=0.95, yaw=5.0, pitch=3.0, inside=True):
    return FaceObservation(
        face_id=face_id,
        confidence=confidence,
        bbox=(0.3, 0.2, 0.2, 0.3),
        inside_frame=inside,
        yaw=yaw,
        pitch=pitch,
        roll=0.0,
        area_ratio=0.06,
        signals={},
    )


def _make_obs(faces):
    return Observation(
        source="face.detect",
        frame_id=1,
        t_ns=0,
        signals={"face_count": len(faces)},
        data=FaceDetectOutput(faces=faces, detected_faces=[], image_size=(640, 480)),
    )


class TestFaceDetectAnnotate:
    def test_none_obs_returns_empty(self):
        analyzer = FaceDetectionAnalyzer()
        assert analyzer.annotate(None) == []

    def test_none_data_returns_empty(self):
        analyzer = FaceDetectionAnalyzer()
        obs = Observation(source="face.detect", frame_id=1, t_ns=0, signals={}, data=None)
        assert analyzer.annotate(obs) == []

    def test_returns_bbox_marks(self):
        analyzer = FaceDetectionAnalyzer()
        faces = [_make_face(1), _make_face(2)]
        obs = _make_obs(faces)
        marks = analyzer.annotate(obs)

        assert len(marks) == 2
        assert all(isinstance(m, BBoxMark) for m in marks)
        assert marks[0].label == "ID:1"
        assert marks[1].label == "ID:2"

    def test_good_face_green(self):
        analyzer = FaceDetectionAnalyzer()
        obs = _make_obs([_make_face(1, confidence=0.95, yaw=5, pitch=3, inside=True)])
        marks = analyzer.annotate(obs)

        assert marks[0].color == (0, 255, 0)

    def test_bad_face_red(self):
        analyzer = FaceDetectionAnalyzer()
        obs = _make_obs([_make_face(1, confidence=0.3, yaw=40, pitch=30, inside=False)])
        marks = analyzer.annotate(obs)

        assert marks[0].color == (0, 0, 255)

    def test_empty_faces_returns_empty(self):
        analyzer = FaceDetectionAnalyzer()
        obs = _make_obs([])
        marks = analyzer.annotate(obs)

        assert marks == []
