"""Face gate analyzer — per-face independent quality gate in the DAG.

Replaces frame.gate with per-face gate judgments using face.classify roles.
Each face is independently evaluated with role-specific thresholds.

Exposure judgment uses local contrast (CV = std/mean) when face.quality
provides mask-based metrics, falling back to absolute brightness otherwise.

depends: ["face.detect", "face.classify"]
optional_depends: ["face.quality", "head.pose"]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from visualbase import Frame

from vpx.sdk import Module, Observation
from visualpath.core.capabilities import Capability, ModuleCapabilities
from momentscan.algorithm.analyzers.face_gate.output import (
    FaceGateConfig,
    FaceGateResult,
    FaceGateOutput,
)

logger = logging.getLogger(__name__)


def _safe_get(obs: Any, attr_path: str, default: float = 0.0) -> float:
    """Get a value from obs by dot path, supporting both attrs and dict keys."""
    if obs is None:
        return default
    obj = obs
    for part in attr_path.split("."):
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    if obj is None:
        return default
    if isinstance(obj, (int, float)):
        return float(obj)
    return default


class FaceGateAnalyzer(Module):
    """Per-face quality gate analyzer.

    Evaluates gate conditions independently for each classified face:

    - main: full gate (area, blur, exposure, yaw, pitch)
    - passenger: relaxed gate (lower area/blur thresholds, no pose check)
    - transient/noise: auto-rejected

    depends: ["face.detect", "face.classify"]
    optional_depends: ["face.quality", "head.pose"]
    """

    depends = ["face.detect", "face.classify"]
    optional_depends = ["face.quality", "frame.quality", "head.pose"]

    def __init__(self, config: FaceGateConfig | None = None):
        self.config = config or FaceGateConfig()
        self._stats_total = 0
        self._stats_fail = 0
        self._stats_reasons: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "face.gate"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.DETERMINISTIC | Capability.THREAD_SAFE,
        )

    def initialize(self) -> None:
        self._stats_total = 0
        self._stats_fail = 0
        self._stats_reasons = {}

    def cleanup(self) -> None:
        if self._stats_total > 0:
            pass_count = self._stats_total - self._stats_fail
            logger.info(
                "face.gate summary: %d/%d passed (%.0f%%), fail reasons: %s",
                pass_count, self._stats_total,
                100.0 * pass_count / self._stats_total,
                dict(self._stats_reasons) if self._stats_reasons else "none",
            )

    def process(
        self,
        frame: "Frame",
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Observation:
        deps = deps or {}
        cfg = self.config

        # Required deps
        face_obs = deps.get("face.detect")
        classify_obs = deps.get("face.classify")

        # Optional deps
        face_q_obs = deps.get("face.quality")
        head_pose_obs = deps.get("head.pose")
        frame_q_obs = deps.get("frame.quality")

        # Extract classified faces
        classified_faces: List[Any] = []
        if classify_obs is not None:
            data = getattr(classify_obs, "data", None)
            if data is not None:
                classified_faces = getattr(data, "faces", []) or []

        # If no classifier output, fall back to face.detect for basic gate
        if not classified_faces and face_obs is not None:
            face_data = getattr(face_obs, "data", None)
            if face_data is not None:
                raw_faces = getattr(face_data, "faces", []) or []
                if raw_faces:
                    # Wrap raw faces as "main" (largest) for backward compat
                    main = max(raw_faces, key=lambda f: getattr(f, "area_ratio", 0.0))
                    classified_faces = [_MockClassified(main, "main")]

        # Extract per-face quality results (face.quality)
        face_quality_map: Dict[int, Any] = {}
        if face_q_obs is not None:
            fq_data = getattr(face_q_obs, "data", None)
            if fq_data is not None:
                for fr in getattr(fq_data, "face_results", []):
                    face_quality_map[fr.face_id] = fr
                # Fallback: main face compat fields
                if not face_quality_map:
                    face_quality_map[-1] = fq_data  # sentinel for main face

        # Extract head.pose estimates (index-aligned with face.detect faces)
        head_estimates: List[Any] = []
        if head_pose_obs is not None:
            hp_data = getattr(head_pose_obs, "data", None)
            if hp_data is not None:
                head_estimates = getattr(hp_data, "estimates", []) or []

        # Frame-level blur/exposure fallback
        frame_blur = _safe_get(frame_q_obs, "signals.blur_score", 0.0) if frame_q_obs else 0.0
        frame_bright = _safe_get(frame_q_obs, "signals.brightness", 0.0) if frame_q_obs else 0.0

        # Build face_id -> pose estimate mapping
        face_id_to_pose: Dict[int, Any] = {}
        if face_obs is not None and head_estimates:
            face_data = getattr(face_obs, "data", None)
            if face_data is not None:
                raw_faces = getattr(face_data, "faces", []) or []
                for i, f in enumerate(raw_faces):
                    if i < len(head_estimates):
                        face_id_to_pose[getattr(f, "face_id", i)] = head_estimates[i]

        # Per-face gate judgment
        results: List[FaceGateResult] = []
        main_result: Optional[FaceGateResult] = None

        for cf in classified_faces:
            face = getattr(cf, "face", cf)
            role = getattr(cf, "role", "main")
            face_id = getattr(face, "face_id", 0)
            fails: List[str] = []

            face_bbox = tuple(getattr(face, "bbox", (0.0, 0.0, 0.0, 0.0)))

            # Auto-reject noise/transient
            if role in ("noise", "transient"):
                results.append(FaceGateResult(
                    face_id=face_id,
                    role=role,
                    gate_passed=False,
                    fail_reasons=("role_rejected",),
                    face_bbox=face_bbox,
                ))
                continue

            # Face metrics
            confidence = float(getattr(face, "confidence", 0.0))
            area_ratio = float(getattr(face, "area_ratio", 0.0))

            # Confidence: main only (passenger trusts detection)
            if role == "main":
                if confidence < cfg.face_confidence_min:
                    fails.append("face_confidence")

            # Blur: face.quality per-face preferred, frame fallback
            fq = face_quality_map.get(face_id) or face_quality_map.get(-1)
            head_blur = float(getattr(fq, "head_blur", 0.0)) if fq else 0.0
            head_exposure = float(getattr(fq, "head_exposure", 0.0)) if fq else 0.0

            blur_min = cfg.head_blur_min if role == "main" else cfg.passenger_blur_min
            if head_blur > 0:
                if head_blur < blur_min:
                    fails.append("blur")
            elif frame_blur > 0:
                if frame_blur < cfg.frame_blur_min:
                    fails.append("blur")

            # Exposure: prefer local contrast metrics, fallback to absolute brightness
            head_contrast = float(getattr(fq, "head_contrast", 0.0)) if fq else 0.0
            clipped_ratio = float(getattr(fq, "clipped_ratio", 0.0)) if fq else 0.0
            crushed_ratio = float(getattr(fq, "crushed_ratio", 0.0)) if fq else 0.0

            if head_contrast > 0:
                # face.quality mask-based local contrast
                if head_contrast < cfg.contrast_min:
                    fails.append("exposure")
                if clipped_ratio > cfg.clipped_max:
                    fails.append("exposure")
                if crushed_ratio > cfg.crushed_max:
                    fails.append("exposure")
            elif head_exposure > 0:
                # face.quality absolute brightness fallback
                if not (cfg.exposure_min <= head_exposure <= cfg.exposure_max):
                    fails.append("exposure")
            elif frame_bright > 0:
                # frame-level absolute brightness fallback
                if not (cfg.exposure_min <= frame_bright <= cfg.exposure_max):
                    fails.append("exposure")

            # Deduplicate exposure failures (multiple contrast conditions may trigger)
            if fails.count("exposure") > 1:
                fails = [f for i, f in enumerate(fails) if f != "exposure" or fails.index("exposure") == i]

            # Pose: main only (passenger doesn't need frontal pose)
            yaw = float(getattr(face, "yaw", 0.0))
            pitch = float(getattr(face, "pitch", 0.0))

            # Override with precise head.pose if available
            pose_est = face_id_to_pose.get(face_id)
            if pose_est is not None:
                est_yaw = getattr(pose_est, "yaw", None)
                if est_yaw is not None:
                    yaw = float(est_yaw)
                est_pitch = getattr(pose_est, "pitch", None)
                if est_pitch is not None:
                    pitch = float(est_pitch)

            gate_passed = len(fails) == 0
            result = FaceGateResult(
                face_id=face_id,
                role=role,
                gate_passed=gate_passed,
                fail_reasons=tuple(fails),
                face_bbox=face_bbox,
                face_area_ratio=area_ratio,
                head_blur=head_blur,
                exposure=head_exposure,
                head_yaw=yaw,
                head_pitch=pitch,
                head_contrast=head_contrast,
                clipped_ratio=clipped_ratio,
                crushed_ratio=crushed_ratio,
            )
            results.append(result)

            if role == "main" and main_result is None:
                main_result = result

        # Frame-level gate (main face based)
        if main_result is not None:
            main_gate_passed = main_result.gate_passed
            main_fail_reasons = main_result.fail_reasons
        elif not classified_faces:
            # No faces at all
            main_gate_passed = False
            main_fail_reasons = ("face_detected",)
        else:
            # All faces are noise/transient
            main_gate_passed = False
            main_fail_reasons = ("no_main_face",)

        output = FaceGateOutput(
            results=results,
            main_gate_passed=main_gate_passed,
            main_fail_reasons=main_fail_reasons,
        )

        # Stats tracking
        self._stats_total += 1
        if not main_gate_passed:
            self._stats_fail += 1
            for reason in main_fail_reasons:
                self._stats_reasons[reason] = self._stats_reasons.get(reason, 0) + 1

        # Logging
        frame_id = getattr(frame, "frame_id", 0)
        faces_passed = sum(1 for r in results if r.gate_passed)
        if self._stats_total == 1:
            dep_names = sorted(deps.keys()) if deps else []
            logger.info(
                "face.gate first frame: deps=%s, main_gate=%s, fails=%s, faces=%d/%d passed",
                dep_names, main_gate_passed, list(main_fail_reasons),
                faces_passed, len(results),
            )
        elif not main_gate_passed:
            logger.debug(
                "face.gate FAIL frame=%d: %s (%d/%d faces passed)",
                frame_id, list(main_fail_reasons), faces_passed, len(results),
            )

        return Observation(
            source=self.name,
            frame_id=frame_id,
            t_ns=getattr(frame, "t_src_ns", 0),
            signals={
                "gate_passed": main_gate_passed,
                "fail_count": len(main_fail_reasons) if not main_gate_passed else 0,
                "faces_gated": len(results),
                "faces_passed": faces_passed,
            },
            data=output,
        )

    def annotate(self, obs):
        """Return LabelMark with gate status at each face's bbox position."""
        if obs is None or obs.data is None:
            return []
        from vpx.sdk.marks import LabelMark

        results = getattr(obs.data, "results", [])
        if not results:
            return []

        marks = []
        for r in results:
            if r.face_bbox == (0.0, 0.0, 0.0, 0.0):
                continue
            bx, by, bw, bh = r.face_bbox
            color = (0, 255, 0) if r.gate_passed else (0, 0, 255)  # green / red
            fail_label = ",".join(r.fail_reasons[:2]) if r.fail_reasons else "OK"
            marks.append(LabelMark(
                text=f"gate:{fail_label}",
                x=bx,
                y=by + bh + 0.01,  # just below face bbox
                color=color,
                font_scale=0.35,
            ))
        return marks


class _MockClassified:
    """Minimal wrapper when face.classify is not available."""

    def __init__(self, face, role: str):
        self.face = face
        self.role = role
