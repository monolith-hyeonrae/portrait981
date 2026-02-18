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

    def __init__(self):
        # ArcFace anchor (best face embedding seen)
        self._anchor_emb: Optional[np.ndarray] = None
        self._anchor_conf: float = 0.0
        # DINOv2 EMA baselines
        self._ema_face: Optional[np.ndarray] = None
        self._ema_body: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._anchor_emb = None
        self._anchor_conf = 0.0
        self._ema_face = None
        self._ema_body = None

    def update(
        self,
        face_obs: Any = None,
        embed_obs: Any = None,
    ) -> dict[str, float]:
        """Compute embedding stats for this frame.

        Args:
            face_obs: face.detect Observation (contains FaceObservation.embedding).
            embed_obs: vision.embed Observation (contains EmbedOutput.e_face/e_body).

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

        # DINOv2 change from vision.embed
        if embed_obs is not None:
            self._update_dinov2(embed_obs, stats)

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

    def _update_dinov2(self, embed_obs: Any, stats: dict) -> None:
        """Compute DINOv2 temporal deltas from vision.embed observation."""
        data = getattr(embed_obs, "data", None)
        if data is None:
            return

        # Face change
        e_face = getattr(data, "e_face", None)
        if e_face is not None:
            e_face = np.asarray(e_face, dtype=np.float32)
            if self._ema_face is None:
                self._ema_face = e_face.copy()
            else:
                delta = 1.0 - float(np.dot(e_face, self._ema_face))
                stats["face_change"] = max(0.0, delta)
                self._ema_face = 0.1 * e_face + 0.9 * self._ema_face
                norm = np.linalg.norm(self._ema_face)
                if norm > 1e-8:
                    self._ema_face = self._ema_face / norm

        # Body change
        e_body = getattr(data, "e_body", None)
        if e_body is not None:
            e_body = np.asarray(e_body, dtype=np.float32)
            if self._ema_body is None:
                self._ema_body = e_body.copy()
            else:
                delta = 1.0 - float(np.dot(e_body, self._ema_body))
                stats["body_change"] = max(0.0, delta)
                self._ema_body = 0.1 * e_body + 0.9 * self._ema_body
                norm = np.linalg.norm(self._ema_body)
                if norm > 1e-8:
                    self._ema_body = self._ema_body / norm
