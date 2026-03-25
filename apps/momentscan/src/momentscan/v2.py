"""Momentscan v2 — 간결한 비디오 분석 앱.

visualbase(I/O) → visualpath(DAG) → vpx+plugins(측정) → visualbind(결합+판단)

Usage:
    from momentscan.v2 import MomentscanV2

    app = MomentscanV2(bind_model="models/bind_v4.pkl", pose_model="models/pose_v2.pkl")
    app.initialize()
    results = app.analyze_video("video.mp4", fps=2)
    selected = app.select_frames(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from visualbase.core.frame import Frame
from visualpath.core.path import Path as AnalysisPath
from visualbind.observer_bind import bind_observations
from visualbind.signals import SIGNAL_FIELDS, normalize_signal

logger = logging.getLogger("momentscan.v2")


@dataclass
class FrameResult:
    """프레임별 분석 결과."""
    frame_idx: int = 0
    timestamp_ms: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False)
    signals: dict = field(default_factory=dict)
    gate_passed: bool = False
    gate_reasons: list = field(default_factory=list)
    expression: str = ""
    expression_conf: float = 0.0
    pose: str = ""
    pose_conf: float = 0.0
    face_detected: bool = False
    face_count: int = 0


def _load_modules():
    """Load analyzers for the analysis DAG."""
    modules = []

    def _try(name, mod_path, cls_name):
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            modules.append(getattr(mod, cls_name)())
            logger.info("Loaded: %s", name)
        except Exception as e:
            logger.warning("Failed: %s (%s)", name, e)

    # 의존성 순서: face.detect가 먼저, 나머지는 depends 선언에 따라 visualpath가 해결
    _try("face.detect", "vpx.face_detect", "FaceDetectionAnalyzer")
    _try("face.au", "vpx.face_au", "FaceAUAnalyzer")
    _try("face.expression", "vpx.face_expression", "ExpressionAnalyzer")
    _try("head.pose", "vpx.head_pose", "HeadPoseAnalyzer")
    _try("face.parse", "vpx.face_parse", "FaceParseAnalyzer")
    _try("face.quality", "momentscan.face_quality", "FaceQualityAnalyzer")
    _try("frame.quality", "momentscan.frame_quality", "QualityAnalyzer")

    return modules


class MomentscanV2:
    """간결한 momentscan — visualpath DAG + visualbind 결합/판단."""

    def __init__(self, bind_model=None, pose_model=None):
        self._bind_path = Path(bind_model) if bind_model else None
        self._pose_path = Path(pose_model) if pose_model else None
        self._path = None
        self._gate = None
        self._bind = None
        self._pose = None

    def initialize(self):
        """DAG + 모델 로딩."""
        # visualpath DAG
        self._path = AnalysisPath(name="momentscan", modules=_load_modules())
        self._path.initialize()

        # visualbind strategies
        from visualbind.strategies.heuristic import HeuristicStrategy
        self._gate = HeuristicStrategy()

        if self._bind_path and self._bind_path.exists():
            from visualbind.strategies.tree import TreeStrategy
            self._bind = TreeStrategy.load(self._bind_path)
            logger.info("Bind: %s (%d classes)", self._bind_path.name, len(self._bind.classes))

        if self._pose_path and self._pose_path.exists():
            from visualbind.strategies.tree import TreeStrategy
            self._pose = TreeStrategy.load(self._pose_path)
            logger.info("Pose: %s (%d classes)", self._pose_path.name, len(self._pose.classes))

    def analyze_image(self, image: np.ndarray, frame_id: int = 0) -> FrameResult:
        """단일 이미지 분석."""
        h, w = image.shape[:2]
        frame = Frame(data=image, frame_id=frame_id, t_src_ns=0, width=w, height=h)

        # Layer 1: visualpath DAG → Observations
        observations = self._path.analyze_all(frame)

        # Layer 1→2: visualbind observer binding → 49D signals
        signals = bind_observations(observations)

        # Face detection check
        face_detected = signals.get("face_confidence", 0.0) > 0
        face_count = sum(1 for obs in observations
                        if obs.source == "face.detect" and obs.signals.get("face_count", 0) > 0)

        result = FrameResult(
            frame_idx=frame_id, image=image, signals=signals,
            face_detected=face_detected, face_count=face_count,
        )

        if not face_detected:
            return result

        # Layer 2: visualbind — gate + predict
        result.gate_reasons = self._gate.check_gate_from_signals(signals)
        result.gate_passed = len(result.gate_reasons) == 0

        vec = self._to_vector(signals)
        if self._bind:
            scores = self._bind.predict(vec)
            if scores:
                result.expression = max(scores, key=scores.get)
                result.expression_conf = scores[result.expression]

        if self._pose:
            scores = self._pose.predict(vec)
            if scores:
                result.pose = max(scores, key=scores.get)
                result.pose_conf = scores[result.pose]

        return result

    def analyze_video(self, video_path, fps=2, max_frames=500):
        """비디오 분석."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps))
        results, idx, extracted = [], 0, 0

        while True:
            ret, img = cap.read()
            if not ret:
                break
            if idx % interval == 0 and extracted < max_frames:
                r = self.analyze_image(img, extracted)
                r.timestamp_ms = idx / video_fps * 1000
                results.append(r)
                extracted += 1
                if extracted % 50 == 0:
                    logger.info("Processing %d frames", extracted)
            idx += 1

        cap.release()
        logger.info("Analyzed %d frames from %s", len(results), Path(video_path).name)
        return results

    def select_frames(self, results, top_k=10):
        """다양성 기반 프레임 선택 (expression × pose 버킷별 best)."""
        shoot = [r for r in results
                 if r.gate_passed and r.face_detected and r.expression and r.expression != "cut"]
        buckets = {}
        for r in shoot:
            key = f"{r.expression}|{r.pose}"
            if key not in buckets or r.expression_conf > buckets[key].expression_conf:
                buckets[key] = r
        return sorted(buckets.values(), key=lambda r: -r.expression_conf)[:top_k]

    def _to_vector(self, signals):
        return np.array([normalize_signal(signals.get(f, 0.0), f) for f in SIGNAL_FIELDS])
