"""Real-time embedding tracker for debug visualization.

Mirrors the batch extract.py logic but runs per-frame during debug mode,
producing face_identity (ArcFace), face_change and body_change (DINOv2)
for immediate visual feedback.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class EmbedTracker:
    """Tracks ArcFace anchor and DINOv2 EMA for real-time quality/delta display.

    Usage::

        tracker = EmbedTracker()
        stats = tracker.update(face_obs, embed_obs)
        # stats = {"face_identity": 0.92, "face_change": 0.05, ...}
    """

    _ANCHOR_DECAY: float = 0.998  # per-frame decay → half-life ~346 frames

    _ALPHA_FAST: float = 0.3   # ~3 frame response
    _ALPHA_SLOW: float = 0.05  # ~20 frame response

    def __init__(self):
        # ArcFace anchor (best face embedding seen)
        self._anchor_emb: Optional[np.ndarray] = None
        self._anchor_conf: float = 0.0
        # DINOv2 dual-EMA (fast tracks recent, slow = baseline)
        self._fast_face: Optional[np.ndarray] = None
        self._slow_face: Optional[np.ndarray] = None
        self._fast_body: Optional[np.ndarray] = None
        self._slow_body: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._anchor_emb = None
        self._anchor_conf = 0.0
        self._fast_face = None
        self._slow_face = None
        self._fast_body = None
        self._slow_body = None

    def update(
        self,
        face_obs: Any = None,
        face_embed_obs: Any = None,
        body_embed_obs: Any = None,
    ) -> dict[str, float]:
        """Compute embedding stats for this frame.

        Args:
            face_obs: face.detect Observation (contains FaceObservation.embedding).
            face_embed_obs: face.embed Observation (contains FaceEmbedOutput.e_face).
            body_embed_obs: body.embed Observation (contains BodyEmbedOutput.e_body).

        Returns:
            Dict with face_identity, face_change, body_change, anchor_conf.
        """
        stats: dict[str, float] = {
            "face_identity": 0.0,
            "face_change": 0.0,
            "body_change": 0.0,
            "anchor_conf": self._anchor_conf,
            "anchor_updated": 0.0,
        }

        # ArcFace identity from face.detect
        if face_obs is not None:
            self._update_arcface(face_obs, stats)

        # DINOv2 face change from face.embed
        if face_embed_obs is not None:
            self._update_face_embed(face_embed_obs, stats)

        # DINOv2 body change from body.embed
        if body_embed_obs is not None:
            self._update_body_embed(body_embed_obs, stats)

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

        # Use largest face (main)
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

        # Decay anchor confidence so newer good faces can overtake
        self._anchor_conf *= self._ANCHOR_DECAY
        if conf > self._anchor_conf:
            self._anchor_emb = emb
            self._anchor_conf = conf
            stats["anchor_updated"] = 1.0

        # Compute similarity to anchor
        if self._anchor_emb is not None:
            sim = float(np.dot(emb, self._anchor_emb))
            stats["face_identity"] = max(0.0, sim)

    def _dual_ema_update(
        self,
        emb: np.ndarray,
        fast: Optional[np.ndarray],
        slow: Optional[np.ndarray],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Dual-EMA update: delta = 1 - dot(fast, slow)."""
        if fast is None:
            return 0.0, emb.copy(), emb.copy()

        af, asl = self._ALPHA_FAST, self._ALPHA_SLOW
        new_fast = af * emb + (1.0 - af) * fast
        nf = np.linalg.norm(new_fast)
        if nf > 1e-8:
            new_fast = new_fast / nf

        new_slow = asl * emb + (1.0 - asl) * slow
        ns = np.linalg.norm(new_slow)
        if ns > 1e-8:
            new_slow = new_slow / ns

        delta = max(0.0, 1.0 - float(np.dot(new_fast, new_slow)))
        return delta, new_fast, new_slow

    def _update_face_embed(self, face_embed_obs: Any, stats: dict) -> None:
        """Compute DINOv2 face dual-EMA temporal delta."""
        data = getattr(face_embed_obs, "data", None)
        if data is None:
            return

        e_face = getattr(data, "e_face", None)
        if e_face is not None:
            e_face = np.asarray(e_face, dtype=np.float32)
            delta, self._fast_face, self._slow_face = self._dual_ema_update(
                e_face, self._fast_face, self._slow_face
            )
            stats["face_change"] = delta

    def _update_body_embed(self, body_embed_obs: Any, stats: dict) -> None:
        """Compute DINOv2 body dual-EMA temporal delta."""
        data = getattr(body_embed_obs, "data", None)
        if data is None:
            return

        e_body = getattr(data, "e_body", None)
        if e_body is not None:
            e_body = np.asarray(e_body, dtype=np.float32)
            delta, self._fast_body, self._slow_body = self._dual_ema_update(
                e_body, self._fast_body, self._slow_body
            )
            stats["body_change"] = delta
