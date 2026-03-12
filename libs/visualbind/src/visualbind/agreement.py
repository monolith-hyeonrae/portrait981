"""AgreementEngine — cross-module consensus computation.

When multiple independent observers point in the same direction,
their agreement is more trustworthy than any single label.

    engine = AgreementEngine([
        CrossCheck("face.expression", "em_happy", "face.au", "AU12", relation="positive"),
        CrossCheck("face.expression", "em_happy", "face.au", "AU6", relation="positive"),
        CrossCheck("head.pose", "yaw", "face.quality", "face_blur", relation="positive"),
    ])

    result = engine.compute(hint_frame)
    # result.score: 0.0 (disagreement) ~ 1.0 (full consensus)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from visualbind.types import AgreementResult, CheckResult, CrossCheck, HintFrame


class AgreementEngine:
    """Computes cross-module consensus from domain-knowledge rules.

    Each CrossCheck defines an expected relationship between two signals
    from different observers. The engine evaluates all checks against a
    HintFrame and produces an aggregate agreement score.

    Agreement for a single check is computed as:
        correlation = value_a * value_b  (for "positive" relation)
        correlation = value_a * (1 - value_b)  (for "negative" relation)

    The overall score is the weighted average of individual agreements.

    Args:
        checks: List of CrossCheck rules.
        confidence_weighted: Weight each check by observer confidence.
    """

    def __init__(
        self,
        checks: Sequence[CrossCheck],
        confidence_weighted: bool = True,
    ):
        self._checks = list(checks)
        self._confidence_weighted = confidence_weighted

    @property
    def checks(self) -> List[CrossCheck]:
        """Registered cross-checks."""
        return list(self._checks)

    def compute(self, frame: HintFrame) -> AgreementResult:
        """Evaluate all cross-checks against a HintFrame.

        Args:
            frame: HintFrame with collected hint vectors.

        Returns:
            AgreementResult with per-check details and aggregate score.
        """
        details: List[CheckResult] = []
        n_available = 0

        for check in self._checks:
            hint_a = frame.hints.get(check.source_a)
            hint_b = frame.hints.get(check.source_b)

            # Check if both sources are present
            if hint_a is None or hint_b is None:
                details.append(CheckResult(
                    check=check,
                    available=False,
                ))
                continue

            value_a = hint_a.get(check.signal_a, 0.0)
            value_b = hint_b.get(check.signal_b, 0.0)

            # Compute agreement based on relation
            if check.relation == "positive":
                # Both high or both low → agreement
                agreement = _positive_agreement(value_a, value_b)
            else:  # negative
                # A high when B low (or vice versa) → agreement
                agreement = _negative_agreement(value_a, value_b)

            details.append(CheckResult(
                check=check,
                value_a=value_a,
                value_b=value_b,
                agreement=agreement,
                available=True,
            ))
            n_available += 1

        # Weighted average of agreements
        score = _weighted_average(details, self._confidence_weighted, frame)

        return AgreementResult(
            score=score,
            details=details,
            n_available=n_available,
            n_total=len(self._checks),
        )


def _positive_agreement(a: float, b: float) -> float:
    """Agreement score for positively correlated signals.

    Uses geometric-mean-like formula:
        agreement = 2 * a * b / (a + b)  when both > 0
        agreement = 1 - |a - b|           general form

    Both high → ~1.0, both low → ~1.0, one high one low → ~0.0

    Returns value in [0, 1].
    """
    # Simple product captures co-activation
    # Both must be active for high agreement
    return a * b


def _negative_agreement(a: float, b: float) -> float:
    """Agreement score for negatively correlated signals.

    A high + B low → agreement
    A low + B high → agreement
    Both high or both low → disagreement

    Returns value in [0, 1].
    """
    return a * (1.0 - b)


def _weighted_average(
    details: List[CheckResult],
    confidence_weighted: bool,
    frame: HintFrame,
) -> float:
    """Compute weighted average agreement score.

    Weight = check.weight * min(confidence_a, confidence_b)
    """
    total_weight = 0.0
    total_score = 0.0

    for result in details:
        if not result.available:
            continue

        weight = result.check.weight

        if confidence_weighted:
            hint_a = frame.hints.get(result.check.source_a)
            hint_b = frame.hints.get(result.check.source_b)
            conf_a = hint_a.confidence if hint_a else 1.0
            conf_b = hint_b.confidence if hint_b else 1.0
            weight *= min(conf_a, conf_b)

        total_weight += weight
        total_score += weight * result.agreement

    if total_weight <= 0:
        return 0.0

    return total_score / total_weight
