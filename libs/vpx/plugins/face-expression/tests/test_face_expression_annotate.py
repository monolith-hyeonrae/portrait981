"""Tests for ExpressionAnalyzer.annotate()."""

from vpx.sdk import Observation
from vpx.sdk.marks import BarMark
from vpx.face_expression.analyzer import ExpressionAnalyzer
from vpx.face_detect.types import FaceObservation
from vpx.face_expression.output import ExpressionOutput


def _make_face(face_id, signals=None):
    return FaceObservation(
        face_id=face_id,
        confidence=0.95,
        bbox=(0.3, 0.2, 0.2, 0.3),
        inside_frame=True,
        yaw=5.0,
        pitch=3.0,
        roll=0.0,
        area_ratio=0.06,
        expression=0.7,
        signals=signals or {"em_happy": 0.8, "em_angry": 0.1, "em_neutral": 0.1},
    )


class TestExpressionAnnotate:
    def test_none_obs_returns_empty(self):
        analyzer = ExpressionAnalyzer()
        assert analyzer.annotate(None) == []

    def test_none_data_returns_empty(self):
        analyzer = ExpressionAnalyzer()
        obs = Observation(source="face.expression", frame_id=1, t_ns=0, signals={}, data=None)
        assert analyzer.annotate(obs) == []

    def test_returns_bar_marks(self):
        analyzer = ExpressionAnalyzer()
        faces = [_make_face(1)]
        obs = Observation(
            source="face.expression",
            frame_id=1,
            t_ns=0,
            signals={},
            data=ExpressionOutput(faces=faces),
        )
        marks = analyzer.annotate(obs)

        # 3 bars per face (happy, angry, neutral)
        assert len(marks) == 3
        assert all(isinstance(m, BarMark) for m in marks)
        # Happy bar value
        assert marks[0].value == 0.8
        assert marks[0].color == (0, 255, 255)
        # Angry bar value
        assert marks[1].value == 0.1
        assert marks[1].color == (0, 0, 255)

    def test_multiple_faces(self):
        analyzer = ExpressionAnalyzer()
        faces = [_make_face(1), _make_face(2)]
        obs = Observation(
            source="face.expression",
            frame_id=1,
            t_ns=0,
            signals={},
            data=ExpressionOutput(faces=faces),
        )
        marks = analyzer.annotate(obs)

        # 3 bars per face × 2 faces = 6
        assert len(marks) == 6

    def test_empty_faces_returns_empty(self):
        analyzer = ExpressionAnalyzer()
        obs = Observation(
            source="face.expression",
            frame_id=1,
            t_ns=0,
            signals={},
            data=ExpressionOutput(faces=[]),
        )
        marks = analyzer.annotate(obs)
        assert marks == []
