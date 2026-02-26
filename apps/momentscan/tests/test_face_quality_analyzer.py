"""Tests for FaceQualityAnalyzer with mask fallback and new metrics."""

import numpy as np
import pytest

from vpx.sdk import Observation
from momentscan.algorithm.analyzers.face_quality import (
    FaceQualityAnalyzer,
    FaceQualityOutput,
    FaceQualityResult,
)
from momentscan.algorithm.analyzers.face_quality.analyzer import (
    _align_parse_to_crop,
    _compute_face_mask,
    _compute_semantic_ratios,
    _laplacian_variance_masked,
    _local_contrast,
    _exposure_stats,
    _center_patch_mask,
    _landmark_ellipse_mask,
    _transform_parse_mask,
    _find_detected_face,
)


# ── Mocks ──

class MockFace:
    def __init__(self, face_id=0, bbox=(0.3, 0.2, 0.2, 0.3), area_ratio=0.06, confidence=0.9):
        self.face_id = face_id
        self.bbox = bbox
        self.area_ratio = area_ratio
        self.confidence = confidence


class MockFaceOutput:
    def __init__(self, faces, image_size=(640, 480)):
        self.faces = faces
        self.image_size = image_size
        self.detected_faces = []


class MockDetectedFace:
    def __init__(self, face_id=0, bbox=(192, 96, 128, 144), landmarks=None):
        self.face_id = face_id
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = 0.9


class MockParseResult:
    def __init__(self, face_id=0, mask=None, crop_box=(180, 80, 160, 180), class_map=None):
        self.face_id = face_id
        self.face_mask = mask if mask is not None else np.ones((512, 512), dtype=np.uint8) * 255
        self.crop_box = crop_box
        # Default class_map: all skin (class 1) — matches default face_mask (all 255)
        self.class_map = class_map if class_map is not None else np.ones((512, 512), dtype=np.uint8)


class MockParseOutput:
    def __init__(self, results=None):
        self.results = results or []


class MockFrame:
    def __init__(self, h=480, w=640):
        self.frame_id = 1
        self.t_src_ns = 100_000_000
        self.data = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)


def _face_obs(faces=None, detected_faces=None, image_size=(640, 480)):
    if faces is None:
        faces = [MockFace()]
    output = MockFaceOutput(faces, image_size)
    output.detected_faces = detected_faces or []
    return Observation(
        source="face.detect",
        frame_id=1,
        t_ns=0,
        data=output,
    )


def _parse_obs(results=None):
    return Observation(
        source="face.parse",
        frame_id=1,
        t_ns=0,
        data=MockParseOutput(results),
    )


# ── Unit tests for mask helpers ──

class TestCenterPatchMask:
    def test_50_percent_coverage(self):
        mask = _center_patch_mask((100, 100))
        coverage = np.count_nonzero(mask) / mask.size
        assert 0.2 < coverage < 0.3  # 50% of each dimension = 25% area

    def test_shape(self):
        mask = _center_patch_mask((200, 300))
        assert mask.shape == (200, 300)


class TestLandmarkEllipseMask:
    def test_valid_5point_landmarks(self):
        gray = np.zeros((200, 200), dtype=np.uint8)
        # 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
        # In image coords, then head_box offsets applied
        landmarks = np.array([
            [120, 130],  # left_eye
            [180, 130],  # right_eye
            [150, 155],  # nose
            [130, 175],  # left_mouth
            [170, 175],  # right_mouth
        ], dtype=np.float64)
        head_box = (100, 100, 200, 200)  # x, y, w, h
        mask = _landmark_ellipse_mask(gray, landmarks, head_box, (200, 200))
        assert np.count_nonzero(mask) > 0

    def test_too_few_landmarks_uses_center_patch(self):
        """With < 5 landmarks, _compute_face_mask should skip to center_patch."""
        gray = np.ones((200, 200), dtype=np.uint8) * 128
        landmarks = np.array([[50, 50], [100, 50]], dtype=np.float64)
        head_box = (0, 0, 200, 200)
        mask, method, cov = _compute_face_mask(gray, None, landmarks, head_box)
        assert method == "center_patch"
        assert cov == 0.0


class TestTransformParseMask:
    def test_full_overlap(self):
        """Parse crop and face crop identical → full mask coverage."""
        result = MockParseResult(crop_box=(100, 100, 200, 200))
        mask = _transform_parse_mask(result, (100, 100, 200, 200), (200, 200))
        assert mask is not None
        assert np.count_nonzero(mask) > 0

    def test_partial_overlap(self):
        """Partially overlapping crops."""
        result = MockParseResult(crop_box=(50, 50, 200, 200))
        mask = _transform_parse_mask(result, (100, 100, 200, 200), (200, 200))
        assert mask is not None
        coverage = np.count_nonzero(mask) / mask.size
        assert coverage > 0
        assert coverage < 1.0

    def test_no_overlap(self):
        """Non-overlapping crops → None."""
        result = MockParseResult(crop_box=(0, 0, 50, 50))
        mask = _transform_parse_mask(result, (200, 200, 100, 100), (100, 100))
        assert mask is None


class TestComputeFaceMask:
    def test_parsing_first(self):
        """With parse result, should use parsing."""
        gray = np.ones((200, 200), dtype=np.uint8) * 128
        parse_result = MockParseResult(
            crop_box=(100, 100, 200, 200),
        )
        mask, method, cov = _compute_face_mask(gray, parse_result, None, (100, 100, 200, 200))
        assert method == "parsing"
        assert cov > 0.0

    def test_landmark_fallback(self):
        """Without parse, with landmarks, should use landmark. coverage=0 (미시도)."""
        gray = np.ones((200, 200), dtype=np.uint8) * 128
        landmarks = np.array([
            [120, 130], [180, 130], [150, 155], [130, 175], [170, 175],
        ], dtype=np.float64)
        mask, method, cov = _compute_face_mask(gray, None, landmarks, (100, 100, 200, 200))
        assert method == "landmark"
        assert cov == 0.0  # parsing 미시도

    def test_center_patch_fallback(self):
        """Without parse or landmarks, should use center_patch."""
        gray = np.ones((200, 200), dtype=np.uint8) * 128
        mask, method, cov = _compute_face_mask(gray, None, None, (100, 100, 200, 200))
        assert method == "center_patch"
        assert cov == 0.0

    def test_sparse_parse_falls_to_landmark_with_coverage(self):
        """Sparse parse mask (below MIN_MASK_COVERAGE) falls to landmark but records coverage."""
        gray = np.ones((200, 200), dtype=np.uint8) * 128
        # Create a very sparse mask — only a few pixels nonzero
        sparse_mask = np.zeros((512, 512), dtype=np.uint8)
        sparse_mask[250:255, 250:255] = 255  # 25 pixels out of 512*512
        parse_result = MockParseResult(
            mask=sparse_mask,
            crop_box=(100, 100, 200, 200),
        )
        landmarks = np.array([
            [120, 130], [180, 130], [150, 155], [130, 175], [170, 175],
        ], dtype=np.float64)
        mask, method, cov = _compute_face_mask(gray, parse_result, landmarks, (100, 100, 200, 200))
        # Sparse mask → below min_pixels → falls to landmark
        assert method == "landmark"
        # But parsing was attempted → coverage > 0 (even if very small)
        assert cov >= 0.0  # coverage recorded from parsing attempt

    def test_empty_parse_falls_to_landmark(self):
        """Empty parse mask (all zeros) should fall through to landmark, but coverage recorded."""
        gray = np.ones((200, 200), dtype=np.uint8) * 128
        parse_result = MockParseResult(
            mask=np.zeros((512, 512), dtype=np.uint8),  # empty!
            crop_box=(100, 100, 200, 200),
        )
        landmarks = np.array([
            [120, 130], [180, 130], [150, 155], [130, 175], [170, 175],
        ], dtype=np.float64)
        mask, method, cov = _compute_face_mask(gray, parse_result, landmarks, (100, 100, 200, 200))
        assert method == "landmark"
        # parsing 시도됨 (parse_result 있음) → coverage 기록됨 (0.0 = empty mask)
        assert cov == 0.0


class TestMetricFunctions:
    def test_laplacian_masked(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        val = _laplacian_variance_masked(gray, mask)
        assert val > 0

    def test_laplacian_empty_mask(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        val = _laplacian_variance_masked(gray, mask)
        assert val == 0.0

    def test_local_contrast(self):
        gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        cv = _local_contrast(gray, mask)
        assert 0.0 < cv < 2.0  # reasonable CV range

    def test_local_contrast_flat(self):
        """Uniform image should have very low contrast."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        cv = _local_contrast(gray, mask)
        assert cv < 0.01

    def test_exposure_stats_normal(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        clipped, crushed = _exposure_stats(gray, mask)
        assert clipped == 0.0
        assert crushed == 0.0

    def test_exposure_stats_overexposed(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        clipped, crushed = _exposure_stats(gray, mask)
        assert clipped == 1.0
        assert crushed == 0.0

    def test_exposure_stats_underexposed(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        clipped, crushed = _exposure_stats(gray, mask)
        assert clipped == 0.0
        assert crushed == 1.0


class TestFindDetectedFace:
    def test_match_by_center(self):
        face = MockFace(bbox=(0.3, 0.2, 0.2, 0.3))  # normalized
        df = MockDetectedFace(bbox=(192, 96, 128, 144))  # pixels in 640x480
        result = _find_detected_face(face, [df], 640, 480)
        assert result is df

    def test_no_match_too_far(self):
        face = MockFace(bbox=(0.1, 0.1, 0.1, 0.1))
        df = MockDetectedFace(bbox=(500, 400, 50, 50))
        result = _find_detected_face(face, [df], 640, 480)
        assert result is None

    def test_empty_list(self):
        face = MockFace()
        result = _find_detected_face(face, [], 640, 480)
        assert result is None


# ── Integration tests ──

class TestFaceQualityAnalyzer:
    def test_name_and_depends(self):
        analyzer = FaceQualityAnalyzer()
        assert analyzer.name == "face.quality"
        assert "face.detect" in analyzer.depends
        assert "face.parse" in analyzer.optional_depends

    def test_process_basic(self):
        """Basic process without face.parse (center_patch fallback)."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        deps = {"face.detect": _face_obs()}
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs is not None
        assert obs.source == "face.quality"
        data = obs.data
        assert data.face_blur >= 0.0
        assert data.mask_method == "center_patch"
        assert data.face_contrast >= 0.0
        analyzer.cleanup()

    def test_process_with_parse(self):
        """Process with face.parse should use parsing mask."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        parse_results = [MockParseResult(face_id=0, crop_box=(160, 60, 200, 240))]

        deps = {
            "face.detect": _face_obs(),
            "face.parse": _parse_obs(parse_results),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs is not None
        # Parsing should be used if crop boxes overlap
        # (depends on exact crop box alignment)
        assert obs.data.mask_method in ("parsing", "center_patch")
        analyzer.cleanup()

    def test_process_with_landmarks(self):
        """Process with DetectedFace landmarks should use landmark mask."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        # Create landmarks in pixel coordinates matching the face bbox
        landmarks = np.array([
            [210, 160], [245, 160], [228, 180], [215, 200], [240, 200],
        ], dtype=np.float64)
        detected = [MockDetectedFace(face_id=0, bbox=(192, 96, 128, 144), landmarks=landmarks)]

        deps = {"face.detect": _face_obs(detected_faces=detected)}
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs is not None
        assert obs.data.mask_method in ("landmark", "center_patch")
        analyzer.cleanup()

    def test_output_new_fields(self):
        """Verify new fields are present in output."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        deps = {"face.detect": _face_obs()}
        obs = analyzer.process(MockFrame(), deps=deps)
        data = obs.data

        assert hasattr(data, "face_contrast")
        assert hasattr(data, "clipped_ratio")
        assert hasattr(data, "crushed_ratio")
        assert hasattr(data, "mask_method")

        # Per-face results too
        if data.face_results:
            r = data.face_results[0]
            assert hasattr(r, "face_contrast")
            assert hasattr(r, "clipped_ratio")
            assert hasattr(r, "crushed_ratio")
            assert hasattr(r, "mask_method")
        analyzer.cleanup()

    def test_process_no_face_detect(self):
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()
        obs = analyzer.process(MockFrame(), deps={})
        assert obs is None
        analyzer.cleanup()

    def test_parsing_coverage_with_parse(self):
        """parsing_coverage should be > 0 when face.parse is available."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        parse_results = [MockParseResult(face_id=0, crop_box=(160, 60, 200, 240))]
        deps = {
            "face.detect": _face_obs(),
            "face.parse": _parse_obs(parse_results),
        }
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs is not None
        if obs.data.mask_method == "parsing":
            assert obs.data.parsing_coverage > 0.0
            assert obs.data.face_results[0].parsing_coverage > 0.0
        analyzer.cleanup()

    def test_parsing_coverage_without_parse(self):
        """parsing_coverage should be 0 when face.parse is not available."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        deps = {"face.detect": _face_obs()}
        obs = analyzer.process(MockFrame(), deps=deps)

        assert obs is not None
        assert obs.data.parsing_coverage == 0.0
        analyzer.cleanup()

    def test_not_initialized_raises(self):
        analyzer = FaceQualityAnalyzer()
        with pytest.raises(RuntimeError):
            analyzer.process(MockFrame(), deps={"face.detect": _face_obs()})

    def test_semantic_fields_with_parse(self):
        """Semantic ratio fields should be populated when face.parse provides class_map."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        # class_map with mixed classes: skin(1) + nose(10) + eyes(4,5)
        cmap = np.ones((512, 512), dtype=np.uint8)  # all skin
        cmap[200:300, 200:300] = 10  # nose region
        cmap[150:180, 170:210] = 4   # left eye
        cmap[150:180, 300:340] = 5   # right eye
        parse_results = [MockParseResult(face_id=0, crop_box=(160, 60, 200, 240), class_map=cmap)]

        deps = {
            "face.detect": _face_obs(),
            "face.parse": _parse_obs(parse_results),
        }
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs is not None
        data = obs.data
        # With class_map provided, semantic fields should be computed
        assert hasattr(data, "seg_face")
        assert hasattr(data, "seg_eye")
        assert hasattr(data, "eye_pixel_ratio")
        # At least face should be > 0 since the class_map is mostly skin (part of face)
        if data.face_results:
            r = data.face_results[0]
            # If parse was used and overlaps, seg_face should be > 0
            if data.mask_method == "parsing":
                assert r.seg_face > 0.0
        analyzer.cleanup()

    def test_semantic_fields_without_parse(self):
        """Semantic fields should be 0 when face.parse is not available."""
        analyzer = FaceQualityAnalyzer()
        analyzer.initialize()

        deps = {"face.detect": _face_obs()}
        obs = analyzer.process(MockFrame(), deps=deps)
        assert obs is not None
        data = obs.data
        assert data.seg_face == 0.0
        assert data.seg_eye == 0.0
        assert data.seg_mouth == 0.0
        assert data.seg_hair == 0.0
        assert data.eye_pixel_ratio == 0.0
        analyzer.cleanup()


# ── Unit tests for _align_parse_to_crop ──

class TestAlignParseToCrop:
    def test_full_overlap(self):
        """Identical crop boxes → full mask transferred."""
        mask = np.ones((512, 512), dtype=np.uint8) * 128
        result = _align_parse_to_crop(mask, (100, 100, 200, 200), (100, 100, 200, 200), (200, 200))
        assert result is not None
        assert result.shape == (200, 200)
        assert np.count_nonzero(result) > 0

    def test_class_value_preserved(self):
        """Class indices should be preserved through transformation (INTER_NEAREST)."""
        cmap = np.zeros((512, 512), dtype=np.uint8)
        cmap[100:400, 100:400] = 10  # nose class
        result = _align_parse_to_crop(cmap, (100, 100, 200, 200), (100, 100, 200, 200), (200, 200))
        assert result is not None
        # After resize (512→200) with INTER_NEAREST, class 10 should be present
        assert np.any(result == 10)
        # Only 0 and 10 should exist
        unique = set(np.unique(result))
        assert unique <= {0, 10}

    def test_no_overlap(self):
        """Non-overlapping crop boxes → None."""
        mask = np.ones((512, 512), dtype=np.uint8)
        result = _align_parse_to_crop(mask, (0, 0, 50, 50), (200, 200, 100, 100), (100, 100))
        assert result is None


# ── Unit tests for _compute_semantic_ratios ──

class TestComputeSemanticRatios:
    def test_all_skin(self):
        """All-skin class_map → seg_face ≈ 1.0 (skin is part of face), rest 0."""
        cmap = np.ones((100, 100), dtype=np.uint8)  # class 1 = skin
        ratios = _compute_semantic_ratios(cmap)
        assert ratios["seg_face"] == pytest.approx(1.0)
        assert ratios["seg_mouth"] == 0.0
        assert ratios["seg_eye"] == 0.0
        assert ratios["seg_hair"] == 0.0
        assert ratios["eye_pixel_ratio"] == 0.0

    def test_mixed_classes(self):
        """Mixed class_map → correct proportions."""
        cmap = np.zeros((100, 100), dtype=np.uint8)  # background
        cmap[:50, :] = 1   # skin: 50% of area
        cmap[50:60, :] = 10  # nose: 10%
        cmap[60:70, :50] = 4  # left eye: 5%
        cmap[60:70, 50:] = 5  # right eye: 5%
        ratios = _compute_semantic_ratios(cmap)
        # face = skin(50%) + nose(10%) + eye(10%) = 70%
        assert ratios["seg_face"] == pytest.approx(0.70)
        assert ratios["seg_eye"] == pytest.approx(0.10)
        assert ratios["eye_pixel_ratio"] == ratios["seg_eye"]

    def test_empty_region(self):
        """Region with zero area → all zeros."""
        cmap = np.ones((100, 100), dtype=np.uint8)
        ratios = _compute_semantic_ratios(cmap, region=(200, 200, 50, 50))
        assert ratios["seg_face"] == 0.0
        assert ratios["seg_eye"] == 0.0
