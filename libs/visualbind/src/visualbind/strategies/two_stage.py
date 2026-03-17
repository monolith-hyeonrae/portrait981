"""Two-stage binding strategy: gate (reject/accept) → category classification.

Stage 1: Binary gate — "Is this frame portrait-worthy?"
Stage 2: Category classifier — "Which bucket does it belong to?"

Mirrors a photographer's judgment: first decide whether to press the shutter,
then decide the style/category.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class TwoStageStrategy:
    """Two-stage binding: gate + classify.

    Stage 1 (gate): reject vs accept (binary XGBoost)
    Stage 2 (classify): category classification (multi-class XGBoost, accept only)

    Implements :class:`~visualbind.strategies.BindingStrategy`.
    """

    def __init__(self) -> None:
        self._gate = None  # binary classifier
        self._classifier = None  # multi-class classifier
        self._categories: list[str] = []
        self._gate_threshold: float = 0.5

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    def fit(self, vectors: dict[str, np.ndarray], **kwargs: object) -> None:
        """Train both stages from labeled category vectors.

        Args:
            vectors: category_name -> (N, D) signal vectors.
                Must include "reject" as a category for the gate stage.
        """
        if "reject" not in vectors:
            logger.warning("No 'reject' category — falling back to single-stage")
            self._fit_single_stage(vectors)
            return

        reject_vecs = vectors["reject"]
        accept_cats = {k: v for k, v in vectors.items() if k != "reject"}

        if not accept_cats:
            logger.warning("No accept categories — gate only")
            return

        # Stage 1: gate (reject=0, accept=1)
        X_reject = reject_vecs
        X_accept = np.vstack(list(accept_cats.values()))
        X_gate = np.vstack([X_reject, X_accept])
        y_gate = np.array([0] * len(X_reject) + [1] * len(X_accept))

        self._gate = self._make_xgb(n_classes=2)
        self._gate.fit(X_gate, y_gate)
        logger.info("Stage 1 (gate): %d reject + %d accept = %d samples",
                     len(X_reject), len(X_accept), len(X_gate))

        # Stage 2: category classifier (accept only)
        self._categories = sorted(accept_cats.keys())
        if len(self._categories) < 2:
            # Single category — no classifier needed
            self._classifier = None
            logger.info("Stage 2 (classify): single category '%s', no classifier needed",
                        self._categories[0])
            return

        from sklearn.preprocessing import LabelEncoder
        X_cat_parts, y_cat_parts = [], []
        for name in self._categories:
            vecs = accept_cats[name]
            X_cat_parts.append(vecs)
            y_cat_parts.extend([name] * len(vecs))

        X_cat = np.vstack(X_cat_parts)
        le = LabelEncoder()
        y_cat = le.fit_transform(y_cat_parts)
        self._categories = list(le.classes_)

        self._classifier = self._make_xgb(n_classes=len(self._categories))
        self._classifier.fit(X_cat, y_cat)
        logger.info("Stage 2 (classify): %d samples, %d categories %s",
                     len(X_cat), len(self._categories), self._categories)

    def predict(self, frame_vec: np.ndarray) -> dict[str, float]:
        """Predict gate + category scores.

        Returns:
            Dict with:
            - "reject": probability of rejection (1 - accept_prob)
            - category_name: probability for each accept category
            Category probabilities are scaled by accept probability.
        """
        if self._gate is None:
            return {}

        x = frame_vec.reshape(1, -1)

        # Stage 1: gate
        gate_proba = self._gate.predict_proba(x)[0]
        reject_prob = float(gate_proba[0])
        accept_prob = float(gate_proba[1])

        result = {"reject": reject_prob}

        # Stage 2: category (only if accepted)
        if self._classifier is not None and accept_prob > 0:
            cat_proba = self._classifier.predict_proba(x)[0]
            for name, p in zip(self._categories, cat_proba):
                result[name] = float(p) * accept_prob
        elif len(self._categories) == 1:
            result[self._categories[0]] = accept_prob

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save model to directory."""
        import json
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "strategy": "two_stage",
            "categories": self._categories,
            "gate_threshold": self._gate_threshold,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        with open(path / "gate.pkl", "wb") as f:
            pickle.dump(self._gate, f)

        if self._classifier is not None:
            with open(path / "classifier.pkl", "wb") as f:
                pickle.dump(self._classifier, f)

        logger.info("TwoStageStrategy saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TwoStageStrategy":
        """Load model from directory."""
        import json
        import pickle

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))

        strategy = cls()
        strategy._categories = meta["categories"]
        strategy._gate_threshold = meta.get("gate_threshold", 0.5)

        with open(path / "gate.pkl", "rb") as f:
            strategy._gate = pickle.load(f)  # noqa: S301

        classifier_path = path / "classifier.pkl"
        if classifier_path.exists():
            with open(classifier_path, "rb") as f:
                strategy._classifier = pickle.load(f)  # noqa: S301

        logger.info("TwoStageStrategy loaded from %s (%d categories)",
                     path, len(strategy._categories))
        return strategy

    def _fit_single_stage(self, vectors: dict[str, np.ndarray]) -> None:
        """Fallback: single-stage classifier without gate."""
        from sklearn.preprocessing import LabelEncoder

        self._categories = sorted(vectors.keys())
        X_parts, y_parts = [], []
        for name in self._categories:
            X_parts.append(vectors[name])
            y_parts.extend([name] * len(vectors[name]))

        X = np.vstack(X_parts)
        le = LabelEncoder()
        y = le.fit_transform(y_parts)
        self._categories = list(le.classes_)

        self._classifier = self._make_xgb(n_classes=len(self._categories))
        self._classifier.fit(X, y)

    @staticmethod
    def _make_xgb(n_classes: int):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss" if n_classes > 2 else "logloss",
                verbosity=0,
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)
