"""Face baseline analyzer - per-face identity baseline profiling.

Tracks area_ratio and center position statistics per face identity
using Welford online algorithm for numerically stable mean/std.

depends: ["face.detect", "face.classify"]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from vpx.sdk import Module, Observation
from visualpath.core.capabilities import Capability, ModuleCapabilities
from momentscan.algorithm.analyzers.face_baseline.output import (
    FaceBaselineProfile,
    FaceBaselineOutput,
)

logger = logging.getLogger(__name__)

# Max frames without seeing a face_id before cleaning up its accumulator
_STALE_THRESHOLD = 60


class FaceBaselineAnalyzer(Module):
    """Per-face identity baseline profiling analyzer.

    Tracks area_ratio and center position using Welford online stats.
    Only profiles main and passenger roles (transient/noise ignored).

    depends: ["face.detect", "face.classify"]
    """

    depends = ["face.detect", "face.classify"]

    def __init__(self):
        self._accumulators: Dict[int, Dict[str, Any]] = {}
        self._last_seen: Dict[int, int] = {}
        self._frame_counter: int = 0

    @property
    def name(self) -> str:
        return "face.baseline"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(flags=Capability.STATEFUL)

    def initialize(self) -> None:
        self._accumulators.clear()
        self._last_seen.clear()
        self._frame_counter = 0
        logger.info("FaceBaselineAnalyzer initialized")

    def cleanup(self) -> None:
        if self._accumulators:
            for fid, acc in self._accumulators.items():
                logger.debug(
                    "face.baseline cleanup: face_id=%d n=%d area_mean=%.4f",
                    fid, acc["n"], acc["area_mean"],
                )
        self._accumulators.clear()
        self._last_seen.clear()
        self._frame_counter = 0

    def process(
        self,
        frame: Any,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        deps = deps or {}
        self._frame_counter += 1

        # Extract classified faces from face.classify
        classify_obs = deps.get("face.classify")
        classified_faces: List[Any] = []
        if classify_obs is not None:
            data = getattr(classify_obs, "data", None)
            if data is not None:
                classified_faces = getattr(data, "faces", []) or []

        # Update accumulators for main/passenger faces
        current_ids: set = set()
        for cf in classified_faces:
            role = getattr(cf, "role", "")
            if role not in ("main", "passenger"):
                continue

            face = getattr(cf, "face", None)
            if face is None:
                continue

            face_id = getattr(face, "face_id", 0)
            current_ids.add(face_id)
            self._last_seen[face_id] = self._frame_counter

            bbox = getattr(face, "bbox", (0.0, 0.0, 0.0, 0.0))
            area_ratio = float(getattr(face, "area_ratio", 0.0))
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2

            if face_id not in self._accumulators:
                self._accumulators[face_id] = {
                    "n": 0,
                    "role": role,
                    "area_mean": 0.0, "area_m2": 0.0,
                    "cx_mean": 0.0, "cx_m2": 0.0,
                    "cy_mean": 0.0, "cy_m2": 0.0,
                }

            acc = self._accumulators[face_id]
            acc["role"] = role  # role can change over time
            acc["n"] += 1
            n = acc["n"]

            # Welford online update
            for key, value in [("area", area_ratio), ("cx", cx), ("cy", cy)]:
                old_mean = acc[f"{key}_mean"]
                delta = value - old_mean
                new_mean = old_mean + delta / n
                delta2 = value - new_mean
                acc[f"{key}_mean"] = new_mean
                acc[f"{key}_m2"] += delta * delta2

        # Cleanup stale accumulators
        self._cleanup_old_accumulators()

        # Build profiles
        profiles: List[FaceBaselineProfile] = []
        main_profile: Optional[FaceBaselineProfile] = None
        passenger_profile: Optional[FaceBaselineProfile] = None

        for face_id, acc in self._accumulators.items():
            n = acc["n"]
            if n < 2:
                continue

            profile = FaceBaselineProfile(
                face_id=face_id,
                role=acc["role"],
                n=n,
                area_ratio_mean=acc["area_mean"],
                area_ratio_std=math.sqrt(acc["area_m2"] / (n - 1)),
                center_x_mean=acc["cx_mean"],
                center_x_std=math.sqrt(acc["cx_m2"] / (n - 1)),
                center_y_mean=acc["cy_mean"],
                center_y_std=math.sqrt(acc["cy_m2"] / (n - 1)),
            )
            profiles.append(profile)

            if acc["role"] == "main" and (
                main_profile is None or n > main_profile.n
            ):
                main_profile = profile
            elif acc["role"] == "passenger" and (
                passenger_profile is None or n > passenger_profile.n
            ):
                passenger_profile = profile

        output = FaceBaselineOutput(
            profiles=profiles,
            main_profile=main_profile,
            passenger_profile=passenger_profile,
        )

        frame_id = getattr(frame, "frame_id", 0)
        return Observation(
            source=self.name,
            frame_id=frame_id,
            t_ns=getattr(frame, "t_src_ns", 0),
            signals={
                "profiles_count": len(profiles),
                "main_converged": 1 if main_profile is not None else 0,
                "passenger_converged": 1 if passenger_profile is not None else 0,
            },
            data=output,
        )

    def _cleanup_old_accumulators(self) -> None:
        """Remove accumulators for faces not seen recently."""
        stale = [
            fid for fid, last in self._last_seen.items()
            if self._frame_counter - last > _STALE_THRESHOLD
        ]
        for fid in stale:
            del self._accumulators[fid]
            del self._last_seen[fid]
