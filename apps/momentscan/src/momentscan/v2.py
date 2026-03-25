"""Momentscan v2 — 간결한 비디오 분석 앱.

plugins(측정) + visualbind(판단) 조합. 본질만.

Usage:
    from momentscan.v2 import MomentscanV2

    app = MomentscanV2(bind_model="models/bind_v4.pkl", pose_model="models/pose_v2.pkl")
    app.initialize()

    # 단일 이미지
    result = app.analyze_image(image)

    # 비디오
    results = app.analyze_video("video.mp4", fps=2)
    selected = app.select_frames(results, top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("momentscan.v2")


@dataclass
class FrameResult:
    """프레임별 분석 결과 — 단일 레코드."""
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


class MomentscanV2:
    """간결한 momentscan — plugins(측정) + visualbind(판단) 조합."""

    def __init__(
        self,
        bind_model: str | Path | None = None,
        pose_model: str | Path | None = None,
    ):
        from momentscan.signals import SignalExtractor
        from visualbind.strategies.heuristic import HeuristicStrategy

        self._extractor = SignalExtractor()
        self._gate = HeuristicStrategy()
        self._bind = None
        self._pose = None
        self._bind_path = Path(bind_model) if bind_model else None
        self._pose_path = Path(pose_model) if pose_model else None

    def initialize(self):
        """Analyzer + model 로딩."""
        self._extractor.initialize()

        if self._bind_path and self._bind_path.exists():
            from visualbind.strategies.tree import TreeStrategy
            self._bind = TreeStrategy.load(self._bind_path)
            logger.info("Expression model: %s (%d classes)",
                        self._bind_path, len(self._bind.classes))

        if self._pose_path and self._pose_path.exists():
            from visualbind.strategies.tree import TreeStrategy
            self._pose = TreeStrategy.load(self._pose_path)
            logger.info("Pose model: %s (%d classes)",
                        self._pose_path, len(self._pose.classes))

    def analyze_image(self, image: np.ndarray, frame_id: int = 0) -> FrameResult:
        """단일 이미지 분석."""
        sig = self._extractor.extract(image, frame_id)

        result = FrameResult(
            frame_idx=frame_id,
            image=image,
            signals=sig.signals,
            face_detected=sig.face_detected,
            face_count=sig.face_count,
        )

        if not sig.face_detected:
            return result

        # Gate
        result.gate_reasons = self._gate.check_gate_from_signals(sig.signals)
        result.gate_passed = len(result.gate_reasons) == 0

        # Expression
        if self._bind:
            vec = self._to_vector(sig.signals)
            scores = self._bind.predict(vec)
            if scores:
                result.expression = max(scores, key=scores.get)
                result.expression_conf = scores[result.expression]

        # Pose
        if self._pose:
            vec = self._to_vector(sig.signals)
            scores = self._pose.predict(vec)
            if scores:
                result.pose = max(scores, key=scores.get)
                result.pose_conf = scores[result.pose]

        return result

    def analyze_video(
        self, video_path: str | Path, fps: int = 2, max_frames: int = 500,
    ) -> list[FrameResult]:
        """비디오 분석 — 프레임별 결과."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps))
        results = []
        idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0 and extracted < max_frames:
                result = self.analyze_image(frame, extracted)
                result.timestamp_ms = idx / video_fps * 1000
                results.append(result)
                extracted += 1
                if (extracted) % 50 == 0:
                    logger.info("Processing %d/%s frames", extracted, max_frames)
            idx += 1

        cap.release()
        logger.info("Analyzed %d frames from %s", len(results), video_path.name)
        return results

    def select_frames(self, results: list[FrameResult], top_k: int = 10) -> list[FrameResult]:
        """SHOOT 프레임 중 다양성 기반 선택.

        expression × pose 버킷별 best confidence.
        """
        shoot = [r for r in results
                 if r.gate_passed and r.face_detected and r.expression and r.expression != "cut"]
        if not shoot:
            return []

        # 버킷별 best
        buckets: dict[str, FrameResult] = {}
        for r in shoot:
            key = f"{r.expression}|{r.pose}"
            if key not in buckets or r.expression_conf > buckets[key].expression_conf:
                buckets[key] = r

        selected = sorted(buckets.values(), key=lambda r: -r.expression_conf)
        return selected[:top_k]

    def _to_vector(self, signals: dict) -> np.ndarray:
        from visualbind.signals import SIGNAL_FIELDS, normalize_signal
        return np.array([normalize_signal(signals.get(f, 0.0), f) for f in SIGNAL_FIELDS])
