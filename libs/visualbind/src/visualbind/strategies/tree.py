"""Tree-based binding strategy: XGBoost with logistic regression fallback.

Requires labeled training data (signal vectors + category labels).
Falls back to logistic regression if ``xgboost`` is not installed.

Persistence: :meth:`TreeStrategy.save` / :meth:`TreeStrategy.load` serialize
the trained model + class list via :mod:`joblib` (or :mod:`pickle` fallback).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

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

    # ── Persistence ──

    def save(self, path: Union[str, Path]) -> None:
        """Save trained model to disk.

        Creates a directory containing ``model.joblib`` (or ``model.pkl``)
        and ``meta.json`` (class list + backend info).

        Args:
            path: Directory to save model artifacts into.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("TreeStrategy has not been fitted — nothing to save")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine backend name for metadata
        backend_name = type(self._model).__name__

        # Save model via joblib (preferred) or pickle fallback
        try:
            import joblib
            model_path = save_dir / "model.joblib"
            joblib.dump(self._model, model_path)
        except ImportError:
            import pickle
            model_path = save_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self._model, f)

        # Save metadata
        meta = {
            "classes": self._classes,
            "backend": backend_name,
            "model_file": model_path.name,
        }
        with open(save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("TreeStrategy saved to %s (%s, %d classes)", save_dir, backend_name, len(self._classes))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TreeStrategy":
        """Load a trained model from disk.

        Args:
            path: Directory containing ``meta.json`` and model file.

        Returns:
            TreeStrategy instance with loaded model.

        Raises:
            FileNotFoundError: If meta.json or model file is missing.
        """
        load_path = Path(path)

        # Support both directory (meta.json + model) and direct file (.pkl/.joblib)
        if load_path.is_file() and load_path.suffix in (".pkl", ".joblib"):
            model_path = load_path
            meta_path = load_path.with_suffix(".json")
            if not meta_path.exists():
                raise FileNotFoundError(f"No meta JSON found: {meta_path}")
            with open(meta_path) as f:
                meta = json.load(f)
        elif load_path.is_dir():
            meta_path = load_path / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"No meta.json in {load_path}")
            with open(meta_path) as f:
                meta = json.load(f)
            model_file = meta["model_file"]
            model_path = load_path / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
        else:
            raise FileNotFoundError(f"Model path not found: {load_path}")

        # Load model via joblib or pickle
        if model_file.endswith(".joblib"):
            try:
                import joblib
                model = joblib.load(model_path)
            except ImportError:
                raise ImportError("joblib is required to load .joblib model files")
        else:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        instance = cls()
        instance._model = model
        instance._classes = meta["classes"]

        logger.info("TreeStrategy loaded from %s (%d classes)", load_dir, len(instance._classes))
        return instance
