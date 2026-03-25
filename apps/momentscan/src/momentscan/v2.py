"""Momentscan v2 — 간결한 비디오 분석 앱.

visualbase(I/O) → visualpath(DAG) → vpx+plugins(측정) → visualbind(결합+판단)

Usage:
    from momentscan.v2 import MomentscanV2

    app = MomentscanV2(
        expression_model="models/bind_v4.pkl",
        pose_model="models/pose_v2.pkl",
    )
    app.initialize()

    # 비디오 분석
    for result in app.stream("video.mp4", fps=2):
        if result.is_shoot:
            print(result.expression, result.pose)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from visualbase import VisualBase, FileSource
from visualpath.core.path import Path as VisualPath
from visualbind import VisualBind, HeuristicStrategy, TreeStrategy, bind_observations
from visualbind.judge import JudgmentResult

logger = logging.getLogger("momentscan.v2")

# momentscan이 사용하는 analyzer 목록
MOMENTSCAN_MODULES = [
    "face.detect", "face.au", "face.expression", "head.pose",
    "face.parse", "face.quality", "frame.quality",
]


@dataclass
class FrameResult:
    """프레임별 분석 결과."""
    frame_idx: int = 0
    timestamp_ms: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False)
    signals: dict = field(default_factory=dict)
    judgment: JudgmentResult = field(default_factory=JudgmentResult)
    face_detected: bool = False
    face_count: int = 0

    # JudgmentResult 위임
    @property
    def gate_passed(self): return self.judgment.gate_passed
    @property
    def expression(self): return self.judgment.expression
    @property
    def expression_conf(self): return self.judgment.expression_conf
    @property
    def pose(self): return self.judgment.pose
    @property
    def pose_conf(self): return self.judgment.pose_conf
    @property
    def is_shoot(self): return self.face_detected and self.judgment.is_shoot


class MomentscanV2:
    """간결한 momentscan — visualpath DAG + visualbind 결합/판단."""

    def __init__(self, expression_model=None, pose_model=None):
        self._expression_model = expression_model
        self._pose_model = pose_model
        self.pipeline = None
        self.judge = None

    def initialize(self):
        """DAG + 모델 로딩."""
        self.pipeline = VisualPath.from_plugins(names=MOMENTSCAN_MODULES, name="momentscan")
        self.pipeline.initialize()

        self.judge = VisualBind(
            gate=HeuristicStrategy(),
            expression=TreeStrategy.load(self._expression_model) if self._expression_model else None,
            pose=TreeStrategy.load(self._pose_model) if self._pose_model else None,
        )
        logger.info("MomentscanV2 ready")

    def analyze(self, frame) -> FrameResult:
        """단일 프레임 분석."""
        observations = self.pipeline.analyze_all(frame)
        signals = bind_observations(observations)

        face_detected = signals.get("face_confidence", 0.0) > 0
        judgment = self.judge(signals) if face_detected else JudgmentResult()

        return FrameResult(
            frame_idx=getattr(frame, "frame_id", 0),
            timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
            image=getattr(frame, "data", None),
            signals=signals,
            judgment=judgment,
            face_detected=face_detected,
            face_count=sum(1 for obs in observations
                          if obs.source == "face.detect" and obs.signals.get("face_count", 0) > 0),
        )

    def stream(self, video_path, fps=2):
        """비디오 프레임별 분석 제너레이터."""
        with VisualBase() as vb:
            vb.connect(FileSource(str(video_path)))
            for frame in vb.get_stream(fps=fps):
                yield self.analyze(frame)

    def analyze_video(self, video_path, fps=2):
        """비디오 전체 분석 → 결과 리스트."""
        return list(self.stream(video_path, fps=fps))

    def select_frames(self, results, top_k=10):
        """다양성 기반 프레임 선택 (expression × pose 버킷별 best)."""
        buckets = {}
        for r in results:
            if not r.is_shoot:
                continue
            key = f"{r.expression}|{r.pose}"
            if key not in buckets or r.expression_conf > buckets[key].expression_conf:
                buckets[key] = r
        return sorted(buckets.values(), key=lambda r: -r.expression_conf)[:top_k]
