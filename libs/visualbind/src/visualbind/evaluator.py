"""Strategy comparison: AUC, accuracy, confusion matrix.

Compare multiple :class:`~visualbind.strategies.BindingStrategy` implementations
on the same test set.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .strategies import BindingStrategy

logger = logging.getLogger(__name__)


def compare_strategies(
    strategies: dict[str, BindingStrategy],
    test_vectors: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Evaluate and compare multiple binding strategies.

    Args:
        strategies: name -> fitted BindingStrategy instance.
        test_vectors: ``(N, D)`` normalized signal vectors.
        test_labels: ``(N,)`` string labels (category names).

    Returns:
        Dict of strategy_name -> {accuracy, per_class_accuracy, confusion_matrix, classes}.
        AUC is included when scikit-learn is available and there are >= 2 classes.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    unique_classes = sorted(set(test_labels))
    results: dict[str, dict[str, Any]] = {}

    for strat_name, strategy in strategies.items():
        # Predict best category for each test vector
        pred_labels = []
        for vec in test_vectors:
            scores = strategy.predict(vec)
            if scores:
                pred_labels.append(max(scores, key=scores.get))  # type: ignore[arg-type]
            else:
                pred_labels.append("")

        pred_arr = np.array(pred_labels)
        acc = float(accuracy_score(test_labels, pred_arr))
        cm = confusion_matrix(test_labels, pred_arr, labels=unique_classes)

        # Per-class accuracy
        per_class = {}
        for i, cls in enumerate(unique_classes):
            mask = test_labels == cls
            if mask.sum() > 0:
                per_class[cls] = float((pred_arr[mask] == cls).mean())
            else:
                per_class[cls] = 0.0

        entry: dict[str, Any] = {
            "accuracy": acc,
            "per_class_accuracy": per_class,
            "confusion_matrix": cm.tolist(),
            "classes": unique_classes,
        }

        # Try to compute AUC (requires probability outputs + >= 2 classes)
        if len(unique_classes) >= 2:
            try:
                from sklearn.metrics import roc_auc_score

                # Build probability matrix
                proba_matrix = []
                for vec in test_vectors:
                    scores = strategy.predict(vec)
                    row = [scores.get(cls, 0.0) for cls in unique_classes]
                    proba_matrix.append(row)
                proba_arr = np.array(proba_matrix)

                # One-hot encode labels
                label_to_idx = {c: i for i, c in enumerate(unique_classes)}
                y_onehot = np.zeros((len(test_labels), len(unique_classes)))
                for i, lbl in enumerate(test_labels):
                    if lbl in label_to_idx:
                        y_onehot[i, label_to_idx[lbl]] = 1.0

                if len(unique_classes) == 2:
                    auc = float(roc_auc_score(y_onehot[:, 1], proba_arr[:, 1]))
                else:
                    auc = float(roc_auc_score(y_onehot, proba_arr, multi_class="ovr", average="macro"))
                entry["auc"] = auc
            except Exception as e:
                logger.debug("AUC computation failed for %s: %s", strat_name, e)

        results[strat_name] = entry
        logger.info(
            "Strategy '%s': accuracy=%.3f%s",
            strat_name, acc,
            f", AUC={entry['auc']:.3f}" if "auc" in entry else "",
        )

    return results
