"""Tests for FaceDetectionAnalyzer batch processing.

Tests verify:
- process_batch() produces correct output for each frame
- Batch detection calls detect_batch() on the backend
- Stateful tracking works correctly in batch mode
- Capabilities include BATCHING flag
"""

import numpy as np

from vpx.face_detect.analyzer import FaceDetectionAnalyzer
from vpx.face_detect.backends.base import DetectedFace
from vpx.sdk import Capability


class MockBatchBackend:
    """Mock backend that tracks detect_batch() calls."""

    def __init__(self, faces_per_frame=None):
        self._faces_per_frame = faces_per_frame or []
        self._call_idx = 0
        self.detect_calls = 0
        self.detect_batch_calls = 0
        self.detect_batch_sizes = []

    def initialize(self, device: str) -> None:
        pass

    def detect(self, image):
        self.detect_calls += 1
        if self._call_idx < len(self._faces_per_frame):
            result = self._faces_per_frame[self._call_idx]
            self._call_idx += 1
            return result
        return []

    def detect_batch(self, images):
        self.detect_batch_calls += 1
        self.detect_batch_sizes.append(len(images))
        results = []
        for _ in images:
            if self._call_idx < len(self._faces_per_frame):
                results.append(self._faces_per_frame[self._call_idx])
                self._call_idx += 1
            else:
                results.append([])
        return results

    def cleanup(self) -> None:
        pass


class MockFrame:
    def __init__(self, frame_id=0, t_src_ns=0, w=640, h=480):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((h, w, 3), dtype=np.uint8)


def _make_faces(count=1, x_start=200, spacing=150):
    """Create a list of DetectedFace objects."""
    return [
        DetectedFace(
            bbox=(x_start + i * spacing, 150, 100, 120),
            confidence=0.90 - i * 0.05,
        )
        for i in range(count)
    ]


class TestFaceDetectBatch:
    """Tests for FaceDetectionAnalyzer.process_batch()."""

    def test_capabilities_include_batching(self):
        """Capabilities should include BATCHING flag."""
        analyzer = FaceDetectionAnalyzer()
        caps = analyzer.capabilities
        assert Capability.BATCHING in caps.flags
        assert caps.max_batch_size == 8

    def test_batch_empty_frames(self):
        """process_batch() with all empty detections."""
        backend = MockBatchBackend(faces_per_frame=[[], [], []])
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        frames = [MockFrame(frame_id=i) for i in range(3)]
        results = analyzer.process_batch(frames, [None] * 3)

        assert len(results) == 3
        for obs in results:
            assert obs is not None
            assert obs.signals["face_count"] == 0
        assert backend.detect_batch_calls == 1
        assert backend.detect_batch_sizes == [3]

    def test_batch_with_faces(self):
        """process_batch() returns correct face counts per frame."""
        faces_per_frame = [
            _make_faces(2),
            [],
            _make_faces(1),
        ]
        backend = MockBatchBackend(faces_per_frame=faces_per_frame)
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        frames = [MockFrame(frame_id=i) for i in range(3)]
        results = analyzer.process_batch(frames, [None] * 3)

        assert len(results) == 3
        assert results[0].signals["face_count"] == 2
        assert results[1].signals["face_count"] == 0
        assert results[2].signals["face_count"] == 1

    def test_batch_uses_detect_batch(self):
        """process_batch() calls detect_batch() on backend, not detect()."""
        backend = MockBatchBackend(faces_per_frame=[[], [], []])
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        frames = [MockFrame(frame_id=i) for i in range(3)]
        analyzer.process_batch(frames, [None] * 3)

        assert backend.detect_batch_calls == 1
        assert backend.detect_calls == 0

    def test_batch_preserves_frame_ids(self):
        """Each observation in batch result has correct frame_id."""
        backend = MockBatchBackend(faces_per_frame=[
            _make_faces(1), _make_faces(1), _make_faces(1), _make_faces(1)
        ])
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        frames = [MockFrame(frame_id=i, t_src_ns=i * 100) for i in range(4)]
        results = analyzer.process_batch(frames, [None] * 4)

        for i, obs in enumerate(results):
            assert obs.frame_id == i
            assert obs.t_ns == i * 100
            assert obs.source == "face.detect"

    def test_batch_has_metrics(self):
        """Each observation in batch has _metrics metadata."""
        backend = MockBatchBackend(faces_per_frame=[
            _make_faces(2), []
        ])
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        frames = [MockFrame(frame_id=i) for i in range(2)]
        results = analyzer.process_batch(frames, [None] * 2)

        assert "_metrics" in results[0].metadata
        assert results[0].metadata["_metrics"]["detection_count"] == 2
        assert "_metrics" in results[1].metadata
        assert results[1].metadata["_metrics"]["detection_count"] == 0

    def test_batch_tracking_sequential(self):
        """Stateful face tracking in batch mode processes frames in order."""
        # Two frames with the same face at similar position
        face_frame1 = [DetectedFace(bbox=(200, 150, 100, 120), confidence=0.9)]
        face_frame2 = [DetectedFace(bbox=(205, 152, 100, 120), confidence=0.85)]

        backend = MockBatchBackend(faces_per_frame=[face_frame1, face_frame2])
        analyzer = FaceDetectionAnalyzer(
            face_backend=backend, track_faces=True, iou_threshold=0.3
        )
        analyzer.initialize()

        frames = [MockFrame(frame_id=i) for i in range(2)]
        results = analyzer.process_batch(frames, [None] * 2)

        # Both frames should have detected faces (tracking should work)
        assert results[0].signals["face_count"] == 1
        assert results[1].signals["face_count"] == 1
        # Face IDs should be consistent (same face tracked)
        face0 = results[0].data.faces[0]
        face1 = results[1].data.faces[0]
        assert face0.face_id == face1.face_id

    def test_batch_single_frame(self):
        """process_batch() with a single frame works correctly."""
        backend = MockBatchBackend(faces_per_frame=[_make_faces(1)])
        analyzer = FaceDetectionAnalyzer(face_backend=backend, track_faces=False)
        analyzer.initialize()

        frames = [MockFrame(frame_id=0)]
        results = analyzer.process_batch(frames, [None])

        assert len(results) == 1
        assert results[0].signals["face_count"] == 1


class TestInsightFaceBatchSession:
    """Tests for InsightFaceSCRFD batch session setup."""

    def test_batch_session_init_default_none(self):
        """Batch session attributes are None before initialization."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        assert backend._batch_session is None
        assert backend._batch_input_name is None
        assert backend._batch_output_names is None

    def test_detect_batch_sequential_fallback_when_no_session(self):
        """detect_batch() falls back to sequential detect() when no batch session."""
        from unittest.mock import MagicMock

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()
        backend._batch_session = None  # No batch session

        # Mock detect() to return empty
        backend.detect = MagicMock(return_value=[])

        images = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
        results = backend.detect_batch(images)

        assert len(results) == 3
        assert backend.detect.call_count == 3  # Sequential fallback

    def test_detect_batch_single_image_uses_detect(self):
        """detect_batch() with single image delegates to detect()."""
        from unittest.mock import MagicMock

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()
        backend.detect = MagicMock(return_value=[])

        results = backend.detect_batch([np.zeros((480, 640, 3), dtype=np.uint8)])

        assert len(results) == 1
        assert backend.detect.call_count == 1

    def test_detect_batch_empty(self):
        """detect_batch() with empty list returns empty."""
        from unittest.mock import MagicMock

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()

        assert backend.detect_batch([]) == []

    def test_cleanup_releases_batch_session(self):
        """cleanup() sets batch session to None."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._batch_session = "mock_session"
        backend.cleanup()
        assert backend._batch_session is None


class TestDecodeSingleOutputLayout:
    """Tests for _decode_single handling 2D vs 3D output layouts."""

    def _make_scrfd_mock(self):
        """Create a mock SCRFD model with minimal attributes."""
        from unittest.mock import MagicMock

        scrfd = MagicMock()
        scrfd.input_size = (640, 640)  # (width, height)
        scrfd.fmc = 3
        scrfd.det_thresh = 0.5
        scrfd._feat_stride_fpn = [8, 16, 32]
        scrfd._num_anchors = 2
        scrfd.use_kps = False
        scrfd.center_cache = {}
        return scrfd

    def test_decode_single_2d_flattened(self):
        """_decode_single handles 2D [N*anchors, features] output."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        scrfd = self._make_scrfd_mock()
        batch_size = 2

        # Build fake 2D flattened outputs (scores per stride)
        net_outs = []
        for stride in [8, 16, 32]:
            h, w = 640 // stride, 640 // stride
            anchors = h * w * 2  # num_anchors=2
            # Flattened: [N*anchors, 1]
            scores = np.zeros((batch_size * anchors, 1), dtype=np.float32)
            net_outs.append(scores)
        # bbox outputs (same shape but 4 features)
        for stride in [8, 16, 32]:
            h, w = 640 // stride, 640 // stride
            anchors = h * w * 2
            bboxes = np.zeros((batch_size * anchors, 4), dtype=np.float32)
            net_outs.append(bboxes)

        # Should not raise
        scores_list, bboxes_list, kpss_list = InsightFaceSCRFD._decode_single(
            scrfd, net_outs, img_idx=0
        )
        assert len(scores_list) == 3  # 3 FPN levels
        assert len(bboxes_list) == 3
        assert len(kpss_list) == 0  # use_kps=False

    def test_decode_single_3d_batched(self):
        """_decode_single handles 3D [N, anchors, features] output."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        scrfd = self._make_scrfd_mock()
        batch_size = 4

        # Build fake 3D batched outputs
        net_outs = []
        for stride in [8, 16, 32]:
            h, w = 640 // stride, 640 // stride
            anchors = h * w * 2
            # Batched: [N, anchors, 1]
            scores = np.zeros((batch_size, anchors, 1), dtype=np.float32)
            net_outs.append(scores)
        for stride in [8, 16, 32]:
            h, w = 640 // stride, 640 // stride
            anchors = h * w * 2
            bboxes = np.zeros((batch_size, anchors, 4), dtype=np.float32)
            net_outs.append(bboxes)

        # Should not raise — 3D layout is handled
        scores_list, bboxes_list, kpss_list = InsightFaceSCRFD._decode_single(
            scrfd, net_outs, img_idx=2
        )
        assert len(scores_list) == 3
        assert len(bboxes_list) == 3

    def test_decode_single_3d_with_detection(self):
        """_decode_single with 3D layout correctly finds high-score anchors."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        scrfd = self._make_scrfd_mock()
        scrfd._feat_stride_fpn = [8]  # Single stride for simplicity
        scrfd.fmc = 1
        batch_size = 2

        h, w = 640 // 8, 640 // 8  # 80x80
        anchors = h * w * 2  # 12800

        # 3D scores: only img_idx=1 has a detection
        scores = np.zeros((batch_size, anchors, 1), dtype=np.float32)
        scores[1, 100, 0] = 0.95  # High score in second image

        bboxes = np.zeros((batch_size, anchors, 4), dtype=np.float32)
        bboxes[1, 100, :] = [10, 10, 50, 50]

        net_outs = [scores, bboxes]

        # img_idx=0 should have no detections
        s0, b0, _ = InsightFaceSCRFD._decode_single(scrfd, net_outs, img_idx=0)
        assert all(len(s) == 0 for s in s0)

        # img_idx=1 should find the detection
        s1, b1, _ = InsightFaceSCRFD._decode_single(scrfd, net_outs, img_idx=1)
        assert any(len(s) > 0 for s in s1)


class TestSetupBatchSessionTransposeFix:
    """Tests for _setup_batch_session fixing Transpose→Reshape pattern."""

    def _build_scrfd_like_model(self):
        """Build a minimal ONNX model that mimics SCRFD's detection head.

        Conv [1,3,4,4] → [1,2,4,4] → Transpose perm=[2,3,0,1] → [4,4,1,2]
        → Reshape [-1,1] → [32,1]

        With batch=2 and WITHOUT fix:
            [2,2,4,4] → Transpose [4,4,2,2] → Reshape [-1,1] → [64,1]
            Data interleaved by spatial position (not per-image).
        WITH fix (perm=[0,2,3,1]):
            [2,2,4,4] → Transpose [2,4,4,2] → Reshape [-1,1] → [64,1]
            Data grouped per-image: first 32 = img0, next 32 = img1.
        """
        import onnx
        from onnx import TensorProto, numpy_helper, helper

        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 1])

        conv_w = numpy_helper.from_array(
            np.ones((2, 3, 1, 1), dtype=np.float32), name="conv_w",
        )
        conv_node = helper.make_node("Conv", ["input", "conv_w"], ["conv_out"])

        transpose_node = helper.make_node(
            "Transpose", ["conv_out"], ["transposed"],
            perm=[2, 3, 0, 1],
        )

        reshape_shape = numpy_helper.from_array(
            np.array([-1, 1], dtype=np.int64), name="reshape_shape",
        )
        reshape_node = helper.make_node(
            "Reshape", ["transposed", "reshape_shape"], ["output"],
        )

        graph = helper.make_graph(
            [conv_node, transpose_node, reshape_node], "test_scrfd",
            [X], [Y], [conv_w, reshape_shape],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    def test_transpose_perm_fixed_for_batch(self):
        """Transpose perm=[2,3,0,1] feeding Reshape is changed to [0,2,3,1]."""
        import onnx
        import tempfile, os
        from unittest.mock import MagicMock
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        model = self._build_scrfd_like_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = f.name

        try:
            backend = InsightFaceSCRFD()
            backend._initialized = True
            scrfd = MagicMock()
            scrfd.model_file = model_path
            scrfd.batched = False
            scrfd.session.get_providers.return_value = ["CPUExecutionProvider"]
            scrfd.output_names = ["output"]
            app = MagicMock()
            app.det_model = scrfd
            backend._app = app

            backend._setup_batch_session()
            assert backend._batch_session is not None

            # Create two different inputs
            np.random.seed(123)
            blob_a = np.random.randn(1, 3, 4, 4).astype(np.float32)
            blob_b = np.random.randn(1, 3, 4, 4).astype(np.float32) + 5.0

            # Run individually
            out_a = backend._batch_session.run(None, {backend._batch_input_name: blob_a})[0]
            out_b = backend._batch_session.run(None, {backend._batch_input_name: blob_b})[0]

            # Run as batch
            blob_ab = np.concatenate([blob_a, blob_b], axis=0)
            out_batch = backend._batch_session.run(None, {backend._batch_input_name: blob_ab})[0]

            anchors = 4 * 4 * 2  # H * W * num_filters = 32
            assert out_batch.shape == (64, 1)

            # With the fix, first 32 elements = img_a, next 32 = img_b
            np.testing.assert_allclose(out_batch[:anchors], out_a, atol=1e-5)
            np.testing.assert_allclose(out_batch[anchors:], out_b, atol=1e-5)
        finally:
            os.unlink(model_path)

    def test_transpose_not_feeding_reshape_not_modified(self):
        """Transpose nodes that don't feed into Reshape are left unchanged."""
        import onnx
        from onnx import TensorProto, numpy_helper, helper
        import tempfile, os
        from unittest.mock import MagicMock
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        # Transpose [2,3,0,1] NOT followed by Reshape — should not be modified
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3, 4])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4, 1, 2])

        transpose_node = helper.make_node(
            "Transpose", ["input"], ["output"], perm=[2, 3, 0, 1],
        )
        graph = helper.make_graph(
            [transpose_node], "test_no_fix", [X], [Y],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = f.name

        try:
            backend = InsightFaceSCRFD()
            backend._initialized = True
            scrfd = MagicMock()
            scrfd.model_file = model_path
            scrfd.batched = False
            scrfd.session.get_providers.return_value = ["CPUExecutionProvider"]
            scrfd.output_names = ["output"]
            app = MagicMock()
            app.det_model = scrfd
            backend._app = app

            backend._setup_batch_session()
            assert backend._batch_session is not None

            # With batch=1, output should still be [3,4,1,2] (original perm)
            inp = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
            result = backend._batch_session.run(None, {backend._batch_input_name: inp})[0]
            expected = inp.transpose(2, 3, 0, 1)
            np.testing.assert_array_equal(result, expected)
        finally:
            os.unlink(model_path)


class TestDetectBatchSanityFallback:
    """Tests for detect_batch all-empty sanity check."""

    def test_detect_batch_falls_back_when_batch_empty_but_sequential_works(self):
        """detect_batch falls back to sequential when batch returns all empty."""
        from unittest.mock import MagicMock, patch

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()
        backend._batch_session = MagicMock()  # Has batch session

        fake_face = DetectedFace(bbox=(100, 100, 50, 50), confidence=0.9)

        # Patch _detect_batch_onnx to return all-empty
        with patch.object(backend, '_detect_batch_onnx', return_value=[[], [], []]):
            # Patch detect() to return a face (sequential works)
            with patch.object(backend, 'detect', return_value=[fake_face]) as mock_detect:
                results = backend.detect_batch([
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    np.zeros((480, 640, 3), dtype=np.uint8),
                ])

        # Should have fallen back to sequential
        assert mock_detect.call_count == 4  # 1 sanity check + 3 sequential
        assert len(results) == 3
        assert len(results[0]) == 1  # Each has 1 face from sequential


class TestBatchCalibration:
    """Tests for first-batch calibration sanity check."""

    def test_calibration_disables_batch_on_bbox_mismatch(self):
        """Calibration disables batch when bbox IoU < 0.5."""
        from unittest.mock import MagicMock, patch

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()
        backend._batch_session = MagicMock()
        backend._batch_calibrated = False

        # Batch returns face at wrong location
        batch_face = DetectedFace(bbox=(400, 400, 50, 50), confidence=0.9)
        seq_face = DetectedFace(bbox=(100, 100, 50, 50), confidence=0.9)

        with patch.object(backend, '_detect_batch_onnx', return_value=[[batch_face], []]):
            with patch.object(backend, 'detect', return_value=[seq_face]) as mock_detect:
                results = backend.detect_batch([
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    np.zeros((480, 640, 3), dtype=np.uint8),
                ])

        # Should have disabled batch and fallen back to sequential
        assert backend._batch_session is None
        assert mock_detect.call_count == 3  # 1 calibration + 2 sequential

    def test_calibration_passes_on_matching_bbox(self):
        """Calibration passes when batch and sequential bboxes match."""
        from unittest.mock import MagicMock, patch

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()
        backend._batch_session = MagicMock()
        backend._batch_calibrated = False

        # Batch and sequential return faces at same location
        face = DetectedFace(bbox=(100, 100, 50, 50), confidence=0.9)

        with patch.object(backend, '_detect_batch_onnx', return_value=[[face], [face]]):
            with patch.object(backend, 'detect', return_value=[face]):
                results = backend.detect_batch([
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    np.zeros((480, 640, 3), dtype=np.uint8),
                ])

        # Batch session should still be active
        assert backend._batch_session is not None
        assert backend._batch_calibrated is True
        assert len(results) == 2

    def test_calibration_runs_only_once(self):
        """Calibration check only runs on the first batch call."""
        from unittest.mock import MagicMock, patch, call

        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        backend._initialized = True
        backend._app = MagicMock()
        backend._batch_session = MagicMock()
        backend._batch_calibrated = True  # Already calibrated

        face = DetectedFace(bbox=(100, 100, 50, 50), confidence=0.9)

        with patch.object(backend, '_detect_batch_onnx', return_value=[[face], [face]]):
            with patch.object(backend, 'detect') as mock_detect:
                results = backend.detect_batch([
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    np.zeros((480, 640, 3), dtype=np.uint8),
                ])

        # detect() should NOT be called (no calibration)
        mock_detect.assert_not_called()
        assert len(results) == 2


class TestBboxIoU:
    """Tests for _bbox_iou static method."""

    def test_identical_boxes(self):
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD
        iou = InsightFaceSCRFD._bbox_iou((100, 100, 50, 50), (100, 100, 50, 50))
        assert abs(iou - 1.0) < 1e-6

    def test_no_overlap(self):
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD
        iou = InsightFaceSCRFD._bbox_iou((0, 0, 50, 50), (200, 200, 50, 50))
        assert iou == 0.0

    def test_partial_overlap(self):
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD
        # 50% horizontal overlap, full vertical overlap
        iou = InsightFaceSCRFD._bbox_iou((0, 0, 100, 100), (50, 0, 100, 100))
        expected = 5000.0 / 15000.0  # intersection=50*100, union=10000+10000-5000
        assert abs(iou - expected) < 1e-6


class TestPoseEstimation:
    """Tests for _estimate_pose_from_kps static method."""

    def test_frontal_face_near_zero(self):
        """Frontal face landmarks should produce near-zero yaw/pitch."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        # Symmetric frontal face landmarks
        kps = np.array([
            [170, 200],   # right eye
            [230, 200],   # left eye
            [200, 240],   # nose (centered)
            [175, 270],   # right mouth
            [225, 270],   # left mouth
        ], dtype=np.float32)

        yaw, pitch, roll = InsightFaceSCRFD._estimate_pose_from_kps(kps)
        assert abs(yaw) < 5.0, f"Expected near-zero yaw, got {yaw}"
        assert abs(pitch) < 10.0, f"Expected near-zero pitch, got {pitch}"
        assert abs(roll) < 2.0, f"Expected near-zero roll, got {roll}"

    def test_turned_right_positive_yaw(self):
        """Face turned right should produce positive yaw."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        # Nose shifted right of center
        kps = np.array([
            [170, 200],   # right eye
            [230, 200],   # left eye
            [220, 240],   # nose shifted RIGHT
            [180, 270],   # right mouth
            [230, 270],   # left mouth
        ], dtype=np.float32)

        yaw, pitch, roll = InsightFaceSCRFD._estimate_pose_from_kps(kps)
        assert yaw > 10.0, f"Expected positive yaw for right turn, got {yaw}"

    def test_turned_left_negative_yaw(self):
        """Face turned left should produce negative yaw."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        # Nose shifted left of center
        kps = np.array([
            [170, 200],   # right eye
            [230, 200],   # left eye
            [180, 240],   # nose shifted LEFT
            [170, 270],   # right mouth
            [220, 270],   # left mouth
        ], dtype=np.float32)

        yaw, pitch, roll = InsightFaceSCRFD._estimate_pose_from_kps(kps)
        assert yaw < -10.0, f"Expected negative yaw for left turn, got {yaw}"

    def test_tilted_head_nonzero_roll(self):
        """Tilted head should produce non-zero roll."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        # Eyes tilted (left eye higher)
        kps = np.array([
            [170, 210],   # right eye (lower)
            [230, 190],   # left eye (higher)
            [200, 240],   # nose
            [175, 270],   # right mouth
            [225, 270],   # left mouth
        ], dtype=np.float32)

        yaw, pitch, roll = InsightFaceSCRFD._estimate_pose_from_kps(kps)
        assert abs(roll) > 5.0, f"Expected non-zero roll for tilted head, got {roll}"

    def test_insufficient_landmarks_returns_zero(self):
        """Less than 5 landmarks should return (0, 0, 0)."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        kps = np.array([[100, 100], [200, 100]], dtype=np.float32)
        yaw, pitch, roll = InsightFaceSCRFD._estimate_pose_from_kps(kps)
        assert yaw == 0.0
        assert pitch == 0.0
        assert roll == 0.0

    def test_det_to_faces_uses_pose_estimation(self):
        """_det_to_faces should populate yaw/pitch/roll from landmarks."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()

        # Simulated detection with 1 face, nose shifted right
        det = np.array([[100, 100, 300, 350, 0.95]], dtype=np.float32)
        kpss = np.array([[
            [150, 170],   # right eye
            [250, 170],   # left eye
            [220, 230],   # nose shifted right
            [160, 280],   # right mouth
            [240, 280],   # left mouth
        ]], dtype=np.float32)

        faces = backend._det_to_faces(det, kpss)
        assert len(faces) == 1
        # Should have non-zero yaw (nose shifted right)
        assert faces[0].yaw != 0.0, "Expected non-zero yaw from landmarks"
        assert faces[0].yaw > 0.0, f"Expected positive yaw, got {faces[0].yaw}"

    def test_det_to_faces_no_landmarks_zero_pose(self):
        """_det_to_faces without landmarks should have zero pose."""
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD

        backend = InsightFaceSCRFD()
        det = np.array([[100, 100, 300, 350, 0.95]], dtype=np.float32)

        faces = backend._det_to_faces(det, None)
        assert len(faces) == 1
        assert faces[0].yaw == 0.0
        assert faces[0].pitch == 0.0
        assert faces[0].roll == 0.0
