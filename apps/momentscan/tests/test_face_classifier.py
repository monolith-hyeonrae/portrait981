"""Tests for FaceClassifierAnalyzer."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from momentscan.algorithm.analyzers.face_classifier import (
    FaceClassifierAnalyzer,
    ClassifiedFace,
    FaceClassifierOutput,
)
from vpx.sdk import Observation
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput


def make_face(face_id: int, bbox: tuple, confidence: float = 0.9) -> FaceObservation:
    """Create a FaceObservation for testing."""
    x, y, w, h = bbox
    return FaceObservation(
        face_id=face_id,
        confidence=confidence,
        bbox=bbox,
        inside_frame=True,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        area_ratio=w * h,
        center_distance=abs(x + w / 2 - 0.5),
        signals={},
    )


def make_frame(frame_id: int = 1):
    """Create a mock Frame."""
    frame = MagicMock()
    frame.frame_id = frame_id
    frame.t_src_ns = frame_id * 100_000_000  # 100ms per frame
    frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame


def make_face_detect_obs(faces: list, frame_id: int = 1) -> Observation:
    """Create a face detection Observation."""
    return Observation(
        source="face.detect",
        frame_id=frame_id,
        t_ns=frame_id * 100_000_000,
        signals={},
        data=FaceDetectOutput(faces=faces, detected_faces=[], image_size=(640, 480)),
    )


class TestFaceClassifierAnalyzer:
    """Tests for FaceClassifierAnalyzer."""

    @pytest.fixture
    def classifier(self):
        return FaceClassifierAnalyzer(
            min_track_frames=3,  # Lower for testing
            min_area_ratio=0.005,
            min_confidence=0.5,
        )

    def test_single_face_becomes_main(self, classifier):
        """Single face should be classified as main."""
        classifier.initialize()

        # Process multiple frames to build track history
        for i in range(5):
            frame = make_frame(i)
            face = make_face(1, (0.4, 0.3, 0.2, 0.3))  # Center, large
            deps = {"face.detect": make_face_detect_obs([face], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        assert data.main_face is not None
        assert data.main_face.role == "main"
        assert len(data.passenger_faces) == 0

    def test_two_faces_main_and_passenger(self, classifier):
        """Two faces: closer to camera origin (1.0, 1.0) becomes main."""
        classifier.initialize()

        for i in range(5):
            frame = make_frame(i)
            # Face closer to camera origin (right-bottom) -> main
            face1 = make_face(1, (0.60, 0.4, 0.25, 0.35))
            # Face farther from camera origin (left-top) -> passenger
            face2 = make_face(2, (0.15, 0.1, 0.15, 0.25))
            deps = {"face.detect": make_face_detect_obs([face1, face2], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        assert data.main_face is not None
        assert data.main_face.face.face_id == 1
        assert data.main_face.role == "main"
        assert len(data.passenger_faces) == 1
        assert data.passenger_faces[0].face.face_id == 2
        assert data.passenger_faces[0].role == "passenger"

    def test_max_one_passenger(self, classifier):
        """Even with 3+ faces, at most 1 becomes passenger."""
        classifier.initialize()

        for i in range(5):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))  # Largest -> main
            face2 = make_face(2, (0.65, 0.3, 0.15, 0.25))  # Second -> passenger
            face3 = make_face(3, (0.1, 0.4, 0.1, 0.15))    # Third -> transient
            deps = {"face.detect": make_face_detect_obs([face1, face2, face3], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data

        # Exactly 1 main
        assert data.main_face is not None
        assert data.main_face.role == "main"

        # At most 1 passenger
        assert len(data.passenger_faces) <= 1
        if data.passenger_faces:
            assert data.passenger_faces[0].role == "passenger"

        # Third face becomes transient
        assert data.transient_count >= 1

    def test_noise_faces_excluded(self, classifier):
        """Very small or low confidence faces are noise."""
        classifier.initialize()

        for i in range(5):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))
            # Very small face -> noise
            noise_face = make_face(99, (0.9, 0.9, 0.02, 0.02), confidence=0.3)
            deps = {"face.detect": make_face_detect_obs([face1, noise_face], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        assert data.main_face is not None
        assert data.noise_count >= 1
        assert len(data.passenger_faces) == 0  # Noise doesn't become passenger

    def test_transient_faces(self, classifier):
        """Faces seen for few frames are transient."""
        classifier.initialize()

        # First, establish a main face
        for i in range(5):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))
            deps = {"face.detect": make_face_detect_obs([face1], i)}
            classifier.process(frame, deps)

        # Now add a new face for just 1 frame
        frame = make_frame(5)
        face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))
        new_face = make_face(99, (0.6, 0.3, 0.15, 0.2))
        deps = {"face.detect": make_face_detect_obs([face1, new_face], 5)}
        obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        # The new face should be transient (not enough history)
        # It might be classified as passenger initially but transient due to short track
        all_roles = [cf.role for cf in data.faces]
        assert "transient" in all_roles or data.transient_count > 0

    def test_all_faces_in_output(self, classifier):
        """All faces should appear in data.faces for visualization."""
        classifier.initialize()

        for i in range(5):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))  # main
            face2 = make_face(2, (0.65, 0.3, 0.15, 0.25))  # passenger
            face3 = make_face(3, (0.1, 0.4, 0.08, 0.1))    # transient (small)
            deps = {"face.detect": make_face_detect_obs([face1, face2, face3], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data

        # All 3 faces should be in the faces list
        face_ids_in_output = {cf.face.face_id for cf in data.faces}
        assert 1 in face_ids_in_output
        assert 2 in face_ids_in_output
        assert 3 in face_ids_in_output

    def test_signals_output(self, classifier):
        """Check signals are correctly set."""
        classifier.initialize()

        for i in range(5):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))
            face2 = make_face(2, (0.65, 0.3, 0.15, 0.25))
            deps = {"face.detect": make_face_detect_obs([face1, face2], i)}
            obs = classifier.process(frame, deps)

        assert obs.signals["main_detected"] == 1
        assert obs.signals["passenger_count"] == 1
        assert obs.signals["total_faces"] == 2

    def test_no_faces(self, classifier):
        """Handle no faces gracefully."""
        classifier.initialize()

        frame = make_frame(1)
        deps = {"face.detect": make_face_detect_obs([], 1)}
        obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        assert data.main_face is None
        assert len(data.passenger_faces) == 0
        assert obs.signals["main_detected"] == 0

    def test_position_stability_main_passenger(self, classifier):
        """Faces with stable positions become main/passenger."""
        classifier.initialize()

        # Simulate stable positions (same location each frame)
        for i in range(10):
            frame = make_frame(i)
            # Main: stable center position
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))
            # Passenger: stable side position
            face2 = make_face(2, (0.65, 0.3, 0.15, 0.25))
            deps = {"face.detect": make_face_detect_obs([face1, face2], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        # Both should be classified as main/passenger due to stable positions
        assert data.main_face is not None
        assert len(data.passenger_faces) == 1

    def test_moving_face_becomes_transient(self, classifier):
        """Face that moves significantly becomes transient."""
        classifier.initialize()

        # First establish stable faces
        for i in range(5):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))  # Stable
            deps = {"face.detect": make_face_detect_obs([face1], i)}
            classifier.process(frame, deps)

        # Now add a moving face (position changes each frame)
        for i in range(5, 15):
            frame = make_frame(i)
            face1 = make_face(1, (0.35, 0.2, 0.25, 0.35))  # Stable main
            # Moving face: x position changes significantly
            x_pos = 0.1 + (i - 5) * 0.05  # Moves from 0.1 to 0.6
            moving_face = make_face(99, (x_pos, 0.3, 0.12, 0.2))
            deps = {"face.detect": make_face_detect_obs([face1, moving_face], i)}
            obs = classifier.process(frame, deps)

        data: FaceClassifierOutput = obs.data
        # Main should be the stable face
        assert data.main_face is not None
        assert data.main_face.face.face_id == 1

        # Moving face should be transient (high position drift)
        moving_face_role = None
        for cf in data.faces:
            if cf.face.face_id == 99:
                moving_face_role = cf.role
                break
        # Due to high drift, should be transient
        assert moving_face_role == "transient"


class TestFaceClassifierAnnotate:
    """Tests for FaceClassifierAnalyzer.annotate()."""

    def test_none_obs_returns_empty(self):
        classifier = FaceClassifierAnalyzer()
        assert classifier.annotate(None) == []

    def test_none_data_returns_empty(self):
        classifier = FaceClassifierAnalyzer()
        obs = Observation(source="face.classify", frame_id=1, t_ns=0, signals={}, data=None)
        assert classifier.annotate(obs) == []

    def test_returns_bbox_marks(self):
        from vpx.sdk.marks import BBoxMark

        classifier = FaceClassifierAnalyzer()
        face = make_face(1, (0.3, 0.2, 0.2, 0.3))
        cf = ClassifiedFace(face=face, role="main", confidence=0.9, track_length=50, avg_area=0.06)
        output = FaceClassifierOutput(faces=[cf], main_face=cf)
        obs = Observation(source="face.classify", frame_id=1, t_ns=0, signals={}, data=output)

        marks = classifier.annotate(obs)
        assert len(marks) == 1
        assert isinstance(marks[0], BBoxMark)
        assert marks[0].label == "MAIN (50f)"
        assert marks[0].color == (0, 255, 0)
        assert marks[0].thickness == 3

    def test_role_colors(self):
        from vpx.sdk.marks import BBoxMark

        classifier = FaceClassifierAnalyzer()
        faces_data = []
        for role in ("main", "passenger", "transient", "noise"):
            face = make_face(1, (0.3, 0.2, 0.2, 0.3))
            cf = ClassifiedFace(face=face, role=role, confidence=0.9, track_length=10, avg_area=0.06)
            faces_data.append(cf)

        output = FaceClassifierOutput(faces=faces_data)
        obs = Observation(source="face.classify", frame_id=1, t_ns=0, signals={}, data=output)
        marks = classifier.annotate(obs)

        assert marks[0].color == (0, 255, 0)       # main: green
        assert marks[1].color == (0, 165, 255)      # passenger: orange
        assert marks[2].color == (0, 255, 255)      # transient: yellow
        assert marks[3].color == (128, 128, 128)    # noise: gray
