"""Real-time quality tracker for debug visualization.

Mirrors the batch extract.py logic but runs per-frame during debug mode,
producing face_identity (ArcFace) and portrait.score metrics for immediate
visual feedback.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class EmbedTracker:
    """Tracks ArcFace anchor and portrait.score metrics for debug display.

    Usage::

        tracker = EmbedTracker()
        stats = tracker.update(face_obs, portrait_score_obs)
        # stats = {"face_identity": 0.92, "face_blur": 320.0, ...}
    """

    _ANCHOR_DECAY: float = 0.998  # per-frame decay → half-life ~346 frames

    def __init__(self):
        self._anchor_emb: Optional[np.ndarray] = None
        self._anchor_conf: float = 0.0

    def reset(self) -> None:
        self._anchor_emb = None
        self._anchor_conf = 0.0

    def update(
        self,
        face_obs: Any = None,
        portrait_score_obs: Any = None,
    ) -> dict[str, float]:
        """Compute quality stats for this frame.

        Args:
            face_obs: face.detect Observation (ArcFace embedding → face_identity).
            portrait_score_obs: portrait.score Observation (PortraitScoreOutput fields).

        Returns:
            Dict with face_identity, face_blur, head_aesthetic, anchor_conf.
        """
        stats: dict[str, float] = {
            "face_identity": 0.0,
            "face_blur": 0.0,
            "head_aesthetic": 0.0,
            "anchor_conf": self._anchor_conf,
            "anchor_updated": 0.0,
        }

        if face_obs is not None:
            self._update_arcface(face_obs, stats)

        if portrait_score_obs is not None:
            self._update_portrait_score(portrait_score_obs, stats)

        stats["anchor_conf"] = self._anchor_conf
        return stats

    def _update_arcface(self, face_obs: Any, stats: dict) -> None:
        """Extract ArcFace embedding from main face and compute anchor similarity."""
        data = getattr(face_obs, "data", None)
        if data is None:
            return
        faces = getattr(data, "faces", None)
        if not faces:
            return

        face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            return

        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return
        emb = emb / norm

        conf = float(getattr(face, "confidence", 0.0))

        self._anchor_conf *= self._ANCHOR_DECAY
        if conf > self._anchor_conf:
            self._anchor_emb = emb
            self._anchor_conf = conf
            stats["anchor_updated"] = 1.0

        if self._anchor_emb is not None:
            sim = float(np.dot(emb, self._anchor_emb))
            stats["face_identity"] = max(0.0, sim)

    def _update_portrait_score(self, portrait_score_obs: Any, stats: dict) -> None:
        """Pass through portrait.score metrics from PortraitScoreOutput."""
        data = getattr(portrait_score_obs, "data", None)
        if data is None:
            return

        stats["head_aesthetic"] = float(getattr(data, "head_aesthetic", 0.0))
