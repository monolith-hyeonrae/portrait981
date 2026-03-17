"""Tree-based binding strategy: XGBoost with logistic regression fallback.

Requires labeled training data (signal vectors + category labels).
Falls back to logistic regression if ``xgboost`` is not installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_xgboost():
    """Try to import XGBClassifier, return None if unavailable."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


class TreeStrategy:
    """XGBoost / logistic regression binding strategy.

    Implements :class:`~visualbind.strategies.BindingStrategy`.

    Falls back to ``sklearn.linear_model.LogisticRegression`` if ``xgboost``
    is not installed.
    """

    def __init__(self, use_xgboost: Optional[bool] = None) -> None:
        """
        Args:
            use_xgboost: Force XGBoost (True) or logistic regression (False).
                ``None`` auto-detects XGBoost availability.
        """
        self._model = None
        self._classes: list[str] = []
        self._use_xgboost = use_xgboost

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def fit(self, vectors: dict[str, np.ndarray], **kwargs: object) -> None:
        """Train classifier from labeled category vectors.

        Args:
            vectors: category_name -> ``(N, D)`` normalized signal vectors.
        """
        from sklearn.preprocessing import LabelEncoder

        # Build X, y arrays
        X_parts, y_parts = [], []
        for name, vecs in sorted(vectors.items()):
            if len(vecs) == 0:
                continue
            X_parts.append(vecs)
            y_parts.extend([name] * len(vecs))

        if not X_parts:
            logger.warning("TreeStrategy.fit: no data provided")
            return

        X = np.vstack(X_parts)
        le = LabelEncoder()
        y = le.fit_transform(y_parts)
        self._classes = list(le.classes_)

        # Choose backend
        XGBClassifier = _try_import_xgboost()
        use_xgb = self._use_xgboost if self._use_xgboost is not None else (XGBClassifier is not None)

        if use_xgb and XGBClassifier is not None:
            self._model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
            self._model.fit(X, y)
            logger.info("TreeStrategy fitted with XGBoost (%d classes, %d samples)", len(self._classes), len(X))
        else:
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
            )
            self._model.fit(X, y)
            backend_name = "LogisticRegression" + (" (XGBoost unavailable)" if self._use_xgboost is None else "")
            logger.info("TreeStrategy fitted with %s (%d classes, %d samples)", backend_name, len(self._classes), len(X))

    def predict(self, frame_vec: np.ndarray) -> dict[str, float]:
        """Predict category probabilities for a single frame.

        Args:
            frame_vec: ``(D,)`` normalized signal vector.

        Returns:
            Dict of category_name -> probability.
        """
        if self._model is None or not self._classes:
            return {}

        proba = self._model.predict_proba(frame_vec.reshape(1, -1))[0]
        return {name: float(p) for name, p in zip(self._classes, proba)}
