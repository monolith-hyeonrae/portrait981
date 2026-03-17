"""Day 0 analysis: N_eff, correlation matrix, distribution diagnostics.

These tools support the go/no-go decision before investing in model training.
Key question: "Do the 23 signal dimensions carry enough independent information?"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix.

    Args:
        vectors: ``(N, D)`` signal matrix where N is samples, D is dimensions.

    Returns:
        ``(D, D)`` correlation matrix.  Returns identity if N < 2.
    """
    n, d = vectors.shape
    if n < 2:
        return np.eye(d, dtype=np.float64)
    return np.corrcoef(vectors, rowvar=False)


def compute_neff(corr_matrix: np.ndarray) -> float:
    """Compute effective dimensionality (N_eff) from a correlation matrix.

    Uses the eigenvalue-based formula::

        N_eff = (sum(lambda_i))^2 / sum(lambda_i^2)

    This equals D when all dimensions are uncorrelated (identity matrix),
    and approaches 1 when all dimensions are perfectly correlated.

    Args:
        corr_matrix: ``(D, D)`` correlation matrix.

    Returns:
        Effective number of independent dimensions.
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total < 1e-12:
        return 0.0
    return float(total ** 2 / (eigenvalues ** 2).sum())


def analyze_distributions(
    vectors: np.ndarray,
    field_names: tuple[str, ...],
) -> dict:
    """Compute per-field distributional statistics.

    Args:
        vectors: ``(N, D)`` signal matrix (already normalized to [0, 1]).
        field_names: names for each dimension.

    Returns:
        Dict with keys:

        - ``fields``: dict of field_name -> {mean, std, min, max, q25, median, q75, zero_frac}
        - ``n_samples``: number of samples
        - ``n_dims``: number of dimensions
    """
    n, d = vectors.shape
    result: dict = {
        "n_samples": int(n),
        "n_dims": int(d),
        "fields": {},
    }

    if n == 0:
        return result

    for i, name in enumerate(field_names[:d]):
        col = vectors[:, i]
        result["fields"][name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "q25": float(np.percentile(col, 25)),
            "median": float(np.median(col)),
            "q75": float(np.percentile(col, 75)),
            "zero_frac": float(np.mean(col == 0.0)),
        }

    return result


def generate_report(
    vectors: np.ndarray,
    field_names: tuple[str, ...],
    corr_threshold: float = 0.7,
) -> str:
    """Generate a text report for Day 0 go/no-go analysis.

    Args:
        vectors: ``(N, D)`` normalized signal matrix.
        field_names: dimension names.
        corr_threshold: threshold for flagging high correlations.

    Returns:
        Multi-line text report.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("VisualBind Day 0 Analysis Report")
    lines.append("=" * 60)

    n, d = vectors.shape
    lines.append(f"\nSamples: {n}, Dimensions: {d}")

    if n < 2:
        lines.append("\nInsufficient samples for analysis (need >= 2).")
        return "\n".join(lines)

    # N_eff
    corr = compute_correlation_matrix(vectors)
    neff = compute_neff(corr)
    lines.append(f"\nEffective dimensionality (N_eff): {neff:.1f} / {d}")
    ratio = neff / d if d > 0 else 0.0
    lines.append(f"N_eff ratio: {ratio:.2%}")

    if ratio >= 0.7:
        lines.append("-> GOOD: signals carry substantial independent information")
    elif ratio >= 0.4:
        lines.append("-> MODERATE: some redundancy, but usable")
    else:
        lines.append("-> WARNING: high redundancy, consider reducing dimensions")

    # High correlations
    lines.append(f"\nHighly correlated pairs (|r| > {corr_threshold}):")
    found_high = False
    for i in range(d):
        for j in range(i + 1, d):
            r = corr[i, j]
            if abs(r) > corr_threshold:
                ni = field_names[i] if i < len(field_names) else f"dim_{i}"
                nj = field_names[j] if j < len(field_names) else f"dim_{j}"
                lines.append(f"  {ni} <-> {nj}: r={r:.3f}")
                found_high = True
    if not found_high:
        lines.append("  (none)")

    # Distribution warnings
    dist = analyze_distributions(vectors, field_names)
    lines.append("\nDistribution warnings:")
    found_warn = False
    for name, stats in dist["fields"].items():
        issues = []
        if stats["zero_frac"] > 0.9:
            issues.append(f"mostly zeros ({stats['zero_frac']:.0%})")
        if stats["std"] < 0.01:
            issues.append(f"near-constant (std={stats['std']:.4f})")
        if issues:
            lines.append(f"  {name}: {', '.join(issues)}")
            found_warn = True
    if not found_warn:
        lines.append("  (none)")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
