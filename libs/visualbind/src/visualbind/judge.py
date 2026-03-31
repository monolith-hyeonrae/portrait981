"""VisualBind — 4단 전략 조합 판단.

gate(heuristic) → quality(binary) → expression(5-class) → pose(3-class).

Usage:
    from visualbind import VisualBind, HeuristicStrategy, TreeStrategy

    judge = VisualBind(
        gate=HeuristicStrategy(),
        quality=TreeStrategy.load("models/quality_v1.pkl"),
        expression=TreeStrategy.load("models/bind_v12.pkl"),
        pose=TreeStrategy.load("models/pose_v10.pkl"),
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
    quality: str = ""          # "shoot" or "cut"
    quality_conf: float = 0.0
    quality_scores: dict[str, float] = field(default_factory=dict)
    expression: str = ""
    expression_conf: float = 0.0
    expression_scores: dict[str, float] = field(default_factory=dict)
    pose: str = ""
    pose_conf: float = 0.0
    pose_scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_shoot(self) -> bool:
        """gate 통과 + quality=shoot (또는 quality 모델 없으면 expression 기반 fallback)."""
        if not self.gate_passed:
            return False
        if self.quality:
            return self.quality == "shoot"
        # fallback: quality 모델 없으면 기존 방식
        return self.expression != "" and self.expression != "cut"


class VisualBind:
    """4단 전략 조합 — gate → quality → expression → pose.

    Args:
        gate: 물리적 품질 gate (HeuristicStrategy)
        quality: shoot/cut 이진 분류 (TreeStrategy)
        expression: 표정 분류, shoot-only (TreeStrategy)
        pose: 포즈 분류, shoot-only (TreeStrategy)
    """

    def __init__(self, gate=None, quality=None, expression=None, pose=None):
        self.gate = gate
        self.quality_strategy = quality
        self.expression_strategy = expression
        self.pose_strategy = pose

    def __call__(self, signals: dict[str, float]) -> JudgmentResult:
        """signals dict → 통합 판단 결과."""
        return self.judge(signals)

    def judge(self, signals: dict[str, float]) -> JudgmentResult:
        """signals dict → 통합 판단 결과.

        4단 파이프라인:
        1. gate (heuristic) — 물리적 품질 실패 → 즉시 cut
        2. quality (binary) — shoot/cut 이진 분류 → cut이면 expression/pose 건너뜀
        3. expression (5-class) — shoot 프레임만 표정 분류
        4. pose (3-class) — shoot 프레임만 포즈 분류
        """
        result = JudgmentResult()

        # 1. Gate (heuristic)
        if self.gate is not None:
            result.gate_reasons = self.gate.check_gate_from_signals(signals)
            result.gate_passed = len(result.gate_reasons) == 0

        if not result.gate_passed:
            result.quality = "cut"
            result.quality_conf = 0.99
            result.expression = "cut"
            result.expression_conf = 0.99
            return result

        # 2. Quality (binary: shoot/cut)
        if self.quality_strategy is not None:
            scores = self.quality_strategy.predict(signals)
            if scores:
                result.quality_scores = scores
                result.quality = max(scores, key=scores.get)
                result.quality_conf = scores[result.quality]

            if result.quality == "cut":
                result.expression = "cut"
                result.expression_conf = result.quality_conf
                return result

        # 3. Expression (5-class, shoot-only)
        if self.expression_strategy is not None:
            scores = self.expression_strategy.predict(signals)
            if scores:
                result.expression_scores = scores
                result.expression = max(scores, key=scores.get)
                result.expression_conf = scores[result.expression]

        # 4. Pose (3-class, shoot-only)
        if self.pose_strategy is not None:
            scores = self.pose_strategy.predict(signals)
            if scores:
                result.pose_scores = scores
                result.pose = max(scores, key=scores.get)
                result.pose_conf = scores[result.pose]

        return result
