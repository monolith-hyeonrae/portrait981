"""Momentscan v2 — visualpath App + visualbind 판단.

visualbase(I/O) → visualpath(App/FlowGraph) → vpx+plugins(측정) → visualbind(결합+판단)

Usage:
    from momentscan.v2 import MomentscanV2

    app = MomentscanV2(
        expression_model="models/bind_v4.pkl",
        pose_model="models/pose_v2.pkl",
    )
    results = app.run("video.mp4", fps=2)
    selected = app.select_frames(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import visualpath as vp

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


class MomentscanV2(vp.App):
    """간결한 momentscan — vp.App 상속 + visualbind 판단.

    vp.App이 FlowGraph 구성 + 비디오 실행을 담당.
    MomentscanV2는 on_frame에서 visualbind 판단만 추가.
    """

    modules = MOMENTSCAN_MODULES
    fps = 2
    backend = "simple"

    def __init__(self, expression_model=None, pose_model=None, **kwargs):
        self._expression_model = expression_model
        self._pose_model = pose_model
        self.judge = None
        self._results: list[FrameResult] = []

    def setup(self):
        """vp.App hook: visualbind 모델 로딩."""
        self.judge = VisualBind(
            gate=HeuristicStrategy(),
            expression=TreeStrategy.load(self._expression_model) if self._expression_model else None,
            pose=TreeStrategy.load(self._pose_model) if self._pose_model else None,
        )
        self._results = []
        logger.info("MomentscanV2 ready")

    def on_frame(self, frame, terminal_results):
        """vp.App hook: 프레임마다 visualbind 판단."""
        observations = []
        for flow_data in terminal_results:
            observations.extend(getattr(flow_data, "observations", []))

        signals = bind_observations(observations)
        face_detected = signals.get("face_confidence", 0.0) > 0
        judgment = self.judge(signals) if face_detected else JudgmentResult()

        self._results.append(FrameResult(
            frame_idx=getattr(frame, "frame_id", 0),
            timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
            image=getattr(frame, "data", None),
            signals=signals,
            judgment=judgment,
            face_detected=face_detected,
            face_count=sum(1 for obs in observations
                          if getattr(obs, "source", "") == "face.detect"
                          and obs.signals.get("face_count", 0) > 0),
        ))
        return True

    def after_run(self, result):
        """vp.App hook: 결과 반환."""
        logger.info("Analyzed %d frames, %d SHOOT",
                    len(self._results),
                    sum(1 for r in self._results if r.is_shoot))
        return self._results

    def teardown(self):
        """vp.App hook: 정리."""
        pass

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
