"""Tests for facemoment.pipeline.frame_processor."""

from unittest.mock import Mock, MagicMock, call, patch
import pytest

from facemoment.pipeline.frame_processor import process_frame, FrameResult
from helpers import create_mock_frame


def _make_extractor(name, depends=None, optional_depends=None, obs=None):
    """Create a mock analyzer with given name and depends."""
    ext = Mock()
    ext.name = name
    ext.depends = depends or []
    ext.optional_depends = optional_depends or []
    if obs is not None:
        ext.process.return_value = obs
    else:
        mock_obs = Mock()
        mock_obs.timing = None
        mock_obs.source = name
        ext.process.return_value = mock_obs
    return ext


def _make_obs(source, timing=None, **signals):
    """Create a mock observation."""
    obs = Mock()
    obs.source = source
    obs.timing = timing
    obs.signals = signals
    obs.metadata = {}
    obs.data = None
    return obs


class TestDepsAccumulation:
    """Test dependency accumulation between extractors."""

    def test_deps_accumulation(self):
        """ext2(depends=[ext1]) should receive ext1 result as deps."""
        frame = create_mock_frame()

        obs1 = _make_obs("ext1")
        obs2 = _make_obs("ext2")

        ext1 = _make_extractor("ext1", obs=obs1)
        ext2 = _make_extractor("ext2", depends=["ext1"], obs=obs2)

        result = process_frame(frame, [ext1, ext2])

        assert "ext1" in result.observations
        assert "ext2" in result.observations
        # ext2 should have been called with deps containing ext1's observation
        ext2.process.assert_called_once()
        call_args = ext2.process.call_args
        assert call_args[0][1] == {"ext1": obs1}

    def test_no_deps(self):
        """Analyzer with no depends should get None deps."""
        frame = create_mock_frame()
        ext = _make_extractor("solo")

        result = process_frame(frame, [ext])

        assert "solo" in result.observations
        ext.process.assert_called_once_with(frame, None)

    def test_missing_dep_not_in_dict(self):
        """If a dep is declared but not produced, it's omitted from deps dict."""
        frame = create_mock_frame()

        ext1 = _make_extractor("ext1")
        ext1.process.return_value = None  # produces nothing

        obs2 = _make_obs("ext2")
        ext2 = _make_extractor("ext2", depends=["ext1"], obs=obs2)

        result = process_frame(frame, [ext1, ext2])

        # ext2 still runs but with empty deps dict
        ext2.process.assert_called_once()
        call_args = ext2.process.call_args
        assert call_args[0][1] == {}


class TestFaceDetectDeps:
    """Test that face.detect obs satisfies face.detect dependency."""

    def test_face_detect_deps(self):
        """depends=['face.detect'] + 'face.detect' obs → passed as dep."""
        frame = create_mock_frame()

        face_obs = _make_obs("face.detect")
        ext_face = _make_extractor("face.detect", obs=face_obs)

        expr_obs = _make_obs("face.expression")
        ext_expr = _make_extractor("face.expression", depends=["face.detect"], obs=expr_obs)

        result = process_frame(frame, [ext_face, ext_expr])

        assert "face.detect" in result.observations
        assert "face.expression" in result.observations
        # expression should have received face.detect obs as dep
        ext_expr.process.assert_called_once()
        call_args = ext_expr.process.call_args
        assert call_args[0][1] == {"face.detect": face_obs}


class TestWorkerProcessing:
    """Test subprocess worker integration."""

    def test_worker_processing(self):
        """Mock worker's observation should appear in results."""
        frame = create_mock_frame()

        worker_obs = _make_obs("body.pose")
        worker = Mock()
        worker_result = Mock()
        worker_result.observation = worker_obs
        worker.process.return_value = worker_result

        result = process_frame(frame, [], workers={"body.pose": worker})

        assert "body.pose" in result.observations
        assert result.observations["body.pose"] is worker_obs

    def test_worker_error_isolation(self):
        """Worker error should not crash processing."""
        frame = create_mock_frame()

        ext = _make_extractor("face.detect")
        worker = Mock()
        worker.process.side_effect = RuntimeError("worker died")

        result = process_frame(frame, [ext], workers={"body.pose": worker})

        assert "face.detect" in result.observations
        assert "body.pose" not in result.observations


class TestMonitorHooks:
    """Test monitor callback ordering."""

    def test_monitor_hooks_order(self):
        """Monitor hooks should be called in correct order."""
        frame = create_mock_frame()
        monitor = Mock()

        ext = _make_extractor("face.detect")
        result = process_frame(frame, [ext], monitor=monitor)

        # Expected order: begin_frame → begin_analyzer → end_analyzer → end_frame
        call_names = [c[0] for c in monitor.method_calls]
        assert call_names[0] == "begin_frame"
        assert "begin_analyzer" in call_names
        assert "end_analyzer" in call_names
        assert call_names[-1] == "end_frame"

    def test_monitor_hooks_with_fusion(self):
        """Monitor hooks should include fusion timing."""
        frame = create_mock_frame()
        monitor = Mock()

        ext = _make_extractor("face.detect")
        fusion = Mock()
        fusion.update.return_value = Mock(should_trigger=False)
        fusion.is_gate_open = False
        fusion.in_cooldown = False

        merge_fn = Mock(return_value=_make_obs("merged"))

        result = process_frame(
            frame, [ext],
            fusion=fusion,
            merge_fn=merge_fn,
            monitor=monitor,
        )

        call_names = [c[0] for c in monitor.method_calls]
        assert "record_merge" in call_names
        assert "begin_fusion" in call_names
        assert "end_fusion" in call_names

    def test_no_monitor(self):
        """monitor=None should not cause errors."""
        frame = create_mock_frame()
        ext = _make_extractor("face.detect")

        result = process_frame(frame, [ext], monitor=None)

        assert "face.detect" in result.observations


class TestFusion:
    """Test fusion integration."""

    def test_fusion_update(self):
        """fusion.update(merged_obs, classifier_obs=) should be called."""
        frame = create_mock_frame()

        ext = _make_extractor("face.detect")
        fusion = Mock()
        fusion_result = Mock(should_trigger=True)
        fusion.update.return_value = fusion_result
        fusion.is_gate_open = True
        fusion.in_cooldown = False

        merge_fn = Mock()
        merged_obs = _make_obs("merged")
        merge_fn.return_value = merged_obs

        result = process_frame(
            frame, [ext],
            fusion=fusion,
            merge_fn=merge_fn,
        )

        fusion.update.assert_called_once_with(merged_obs, classifier_obs=None)
        assert result.fusion_result is fusion_result
        assert result.is_gate_open is True
        assert result.in_cooldown is False

    def test_fusion_with_classifier(self):
        """Fusion should receive classifier_obs when present."""
        frame = create_mock_frame()

        cls_obs = _make_obs("face.classify")
        classifier = _make_extractor("face.classify", obs=cls_obs)

        ext = _make_extractor("face.detect")
        fusion = Mock()
        fusion.update.return_value = Mock(should_trigger=False)
        fusion.is_gate_open = False
        fusion.in_cooldown = True

        merge_fn = Mock(return_value=_make_obs("merged"))

        result = process_frame(
            frame, [ext, classifier],
            classifier=classifier,
            fusion=fusion,
            merge_fn=merge_fn,
        )

        fusion.update.assert_called_once()
        _, kwargs = fusion.update.call_args
        assert kwargs["classifier_obs"] is cls_obs
        assert result.classifier_obs is cls_obs
        assert result.in_cooldown is True

    def test_no_fusion(self):
        """Without fusion, fusion fields should be defaults."""
        frame = create_mock_frame()
        ext = _make_extractor("face.detect")

        result = process_frame(frame, [ext])

        assert result.fusion_result is None
        assert result.is_gate_open is False
        assert result.in_cooldown is False


class TestExtractorErrorIsolation:
    """Test that analyzer errors don't break other analyzers."""

    def test_extractor_error_isolation(self):
        """ext1 error should not prevent ext2 from running."""
        frame = create_mock_frame()

        ext1 = _make_extractor("ext1")
        ext1.process.side_effect = RuntimeError("ext1 crashed")

        obs2 = _make_obs("ext2")
        ext2 = _make_extractor("ext2", obs=obs2)

        result = process_frame(frame, [ext1, ext2])

        assert "ext1" not in result.observations
        assert "ext2" in result.observations

    def test_extractor_error_with_monitor(self):
        """Monitor should record failure for errored analyzer."""
        frame = create_mock_frame()
        monitor = Mock()

        ext = _make_extractor("bad")
        ext.process.side_effect = RuntimeError("crash")

        result = process_frame(frame, [ext], monitor=monitor)

        # Monitor should have end_analyzer called with None obs
        end_calls = [
            c for c in monitor.method_calls
            if c[0] == "end_analyzer"
        ]
        assert len(end_calls) == 1
        assert end_calls[0][1] == ("bad", None)


class TestFrameResultFields:
    """Test that FrameResult is correctly populated."""

    def test_frame_result_fields(self):
        """All FrameResult fields should be correctly filled."""
        frame = create_mock_frame()

        face_obs = _make_obs("face.detect", timing={"detect_ms": 5.0})
        ext = _make_extractor("face.detect", obs=face_obs)

        cls_obs = _make_obs("face.classify")
        classifier = _make_extractor("face.classify", obs=cls_obs)

        fusion = Mock()
        fusion_result = Mock(should_trigger=False)
        fusion.update.return_value = fusion_result
        fusion.is_gate_open = True
        fusion.in_cooldown = False

        result = process_frame(
            frame, [ext, classifier],
            classifier=classifier,
            fusion=fusion,
        )

        assert isinstance(result, FrameResult)
        assert result.observations == {"face.detect": face_obs, "face.classify": cls_obs}
        assert result.classifier_obs is cls_obs
        assert result.fusion_result is fusion_result
        assert result.is_gate_open is True
        assert result.in_cooldown is False
        assert result.timing_info == {"detect_ms": 5.0}

    def test_frame_result_no_timing(self):
        """timing_info should be None when face obs has no timing."""
        frame = create_mock_frame()
        ext = _make_extractor("body.pose")  # not face, so no timing_info

        result = process_frame(frame, [ext])

        assert result.timing_info is None

    def test_typedef_fallback(self):
        """Analyzer that doesn't accept deps should fallback to process(frame)."""
        frame = create_mock_frame()

        obs = _make_obs("legacy")
        ext = _make_extractor("legacy", obs=obs)
        ext.depends = []
        # First call with (frame, None) raises TypeError, second with (frame,) succeeds
        ext.process.side_effect = [TypeError("unexpected keyword"), obs]

        # This should not raise — fallback to process(frame)
        result = process_frame(frame, [ext])

        assert ext.process.call_count == 2
        assert "legacy" in result.observations
