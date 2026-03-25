"""VisualBind — 다중 전략 조합 판단.

gate + expression + pose를 하나의 호출로.

Usage:
    from visualbind import VisualBind, HeuristicStrategy, TreeStrategy

    judge = VisualBind(
        gate=HeuristicStrategy(),
        expression=TreeStrategy.load("models/bind_v4.pkl"),
        pose=TreeStrategy.load("models/pose_v2.pkl"),
    )
    result = judge(signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JudgmentResult:
    """판단 결과."""
    gate_passed: bool = True
    gate_reasons: list[str] = field(default_factory=list)
    expression: str = ""
    expression_conf: float = 0.0
    expression_scores: dict[str, float] = field(default_factory=dict)
    pose: str = ""
    pose_conf: float = 0.0
    pose_scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_shoot(self) -> bool:
        return self.gate_passed and self.expression != "" and self.expression != "cut"


class VisualBind:
    """다중 전략 조합 — gate + expression + pose를 하나의 호출로.

    Args:
        gate: 품질 gate 전략 (HeuristicStrategy)
        expression: 표정 분류 전략 (TreeStrategy)
        pose: 포즈 분류 전략 (TreeStrategy)
    """

    def __init__(self, gate=None, expression=None, pose=None):
        self.gate = gate
        self.expression_strategy = expression
        self.pose_strategy = pose

    def __call__(self, signals: dict[str, float]) -> JudgmentResult:
        """signals dict → 통합 판단 결과."""
        return self.judge(signals)

    def judge(self, signals: dict[str, float]) -> JudgmentResult:
        """signals dict → 통합 판단 결과.

        gate가 fail이면 expression/pose 건너뜀 (최적화).
        """
        result = JudgmentResult()

        # Gate
        if self.gate is not None:
            result.gate_reasons = self.gate.check_gate_from_signals(signals)
            result.gate_passed = len(result.gate_reasons) == 0

        if not result.gate_passed:
            result.expression = "cut"
            result.expression_conf = 0.99
            return result

        # Expression
        if self.expression_strategy is not None:
            scores = self.expression_strategy.predict(signals)
            if scores:
                result.expression_scores = scores
                result.expression = max(scores, key=scores.get)
                result.expression_conf = scores[result.expression]

        # Pose
        if self.pose_strategy is not None:
            scores = self.pose_strategy.predict(signals)
            if scores:
                result.pose_scores = scores
                result.pose = max(scores, key=scores.get)
                result.pose_conf = scores[result.pose]

        return result
