"""Momentscan v2 — 간결한 비디오 분석 앱.

visualpath DAG + vpx/momentscan plugins(측정) + visualbind(판단) 조합.

Usage:
    from momentscan.v2 import MomentscanV2

    app = MomentscanV2(bind_model="models/bind_v4.pkl", pose_model="models/pose_v2.pkl")
    app.initialize()
    results = app.analyze_video("video.mp4", fps=2)
    selected = app.select_frames(results, top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from visualbase.core.frame import Frame
from visualpath.core.path import Path as AnalysisPath

logger = logging.getLogger("momentscan.v2")


@dataclass
class FrameResult:
    """프레임별 분석 결과."""
    frame_idx: int = 0
    timestamp_ms: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False)
    signals: dict = field(default_factory=dict)
    gate_passed: bool = False
    gate_reasons: list = field(default_factory=list)
    expression: str = ""
    expression_conf: float = 0.0
    pose: str = ""
    pose_conf: float = 0.0
    face_detected: bool = False
    face_count: int = 0


def _load_modules():
    """Load all analyzers for the analysis path."""
    modules = []

    def _try(name, mod_path, cls_name):
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            instance = getattr(mod, cls_name)()
            modules.append(instance)
            logger.info("Loaded: %s", name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)

    # vpx analyzers (범용, 순서 = 의존성 순서)
    _try("face.detect", "vpx.face_detect", "FaceDetectionAnalyzer")
    _try("face.au", "vpx.face_au", "FaceAUAnalyzer")
    _try("face.expression", "vpx.face_expression", "ExpressionAnalyzer")
    _try("head.pose", "vpx.head_pose", "HeadPoseAnalyzer")
    _try("face.parse", "vpx.face_parse", "FaceParseAnalyzer")

    # momentscan plugins (도메인 특화)
    _try("face.quality", "momentscan.face_quality", "FaceQualityAnalyzer")
    _try("frame.quality", "momentscan.frame_quality", "QualityAnalyzer")

    return modules


class MomentscanV2:
    """간결한 momentscan — visualpath DAG + plugins + visualbind."""

    def __init__(
        self,
        bind_model: str | Path | None = None,
        pose_model: str | Path | None = None,
    ):
        self._bind_path = Path(bind_model) if bind_model else None
        self._pose_path = Path(pose_model) if pose_model else None
        self._path: Optional[AnalysisPath] = None
        self._bind = None
        self._pose = None
        self._gate = None

    def initialize(self):
        """Analyzer DAG + model 로딩."""
        # visualpath DAG
        modules = _load_modules()
        self._path = AnalysisPath(name="momentscan", modules=modules)
        self._path.initialize()
        logger.info("Analysis path: %d modules", len(modules))

        # visualbind gate
        from visualbind.strategies.heuristic import HeuristicStrategy
        self._gate = HeuristicStrategy()

        # visualbind expression model
        if self._bind_path and self._bind_path.exists():
            from visualbind.strategies.tree import TreeStrategy
            self._bind = TreeStrategy.load(self._bind_path)
            logger.info("Expression model: %s (%d classes)",
                        self._bind_path, len(self._bind.classes))

        # visualbind pose model
        if self._pose_path and self._pose_path.exists():
            from visualbind.strategies.tree import TreeStrategy
            self._pose = TreeStrategy.load(self._pose_path)
            logger.info("Pose model: %s (%d classes)",
                        self._pose_path, len(self._pose.classes))

    def analyze_image(self, image: np.ndarray, frame_id: int = 0) -> FrameResult:
        """단일 이미지 분석 — visualpath DAG 실행."""
        h, w = image.shape[:2]
        frame = Frame(data=image, frame_id=frame_id, t_src_ns=0, width=w, height=h)

        # DAG 실행 — 의존성 자동 해결
        observations = self._path.analyze_all(frame)

        # Observations → signals dict
        signals = {}
        face_detected = False
        face_count = 0

        for obs in observations:
            src = obs.source

            if src == "face.detect" and obs.data:
                faces = getattr(obs.data, "faces", [])
                face_count = len(faces)
                face_detected = face_count > 0
                if face_detected:
                    face = max(faces, key=lambda f: f.area_ratio)
                    signals["face_confidence"] = float(face.confidence)
                    signals["face_area_ratio"] = float(face.area_ratio)
                    signals["face_center_distance"] = float(face.center_distance)
                    signals["head_yaw_dev"] = abs(float(getattr(face, "yaw", 0.0)))
                    signals["head_pitch"] = float(getattr(face, "pitch", 0.0))
                    signals["head_roll"] = float(getattr(face, "roll", 0.0))

            # Merge all observation signals
            if obs.signals:
                for k, v in obs.signals.items():
                    if k not in signals:
                        signals[k] = float(v)

            # face.quality specific
            if src == "face.quality" and obs.signals:
                signals["face_blur"] = float(obs.signals.get("face_blur", 0.0))
                signals["face_exposure"] = float(obs.signals.get("face_exposure", 0.0))
                signals["face_contrast"] = float(obs.signals.get("face_contrast", 0.0))
                signals["clipped_ratio"] = float(obs.signals.get("clipped_ratio", 0.0))
                signals["crushed_ratio"] = float(obs.signals.get("crushed_ratio", 0.0))

            # head.pose override
            if src == "head.pose" and obs.signals:
                if "head_yaw" in obs.signals:
                    signals["head_yaw_dev"] = abs(float(obs.signals["head_yaw"]))
                if "head_pitch" in obs.signals:
                    signals["head_pitch"] = float(obs.signals["head_pitch"])
                if "head_roll" in obs.signals:
                    signals["head_roll"] = float(obs.signals["head_roll"])

            # face.au signal key mapping
            if src == "face.au" and obs.signals:
                au_map = {
                    "au_au1": "au1_inner_brow", "au_au2": "au2_outer_brow",
                    "au_au4": "au4_brow_lowerer", "au_au5": "au5_upper_lid",
                    "au_au6": "au6_cheek_raiser", "au_au9": "au9_nose_wrinkler",
                    "au_au12": "au12_lip_corner", "au_au15": "au15_lip_depressor",
                    "au_au17": "au17_chin_raiser", "au_au20": "au20_lip_stretcher",
                    "au_au25": "au25_lips_part", "au_au26": "au26_jaw_drop",
                }
                for src_key, dst_key in au_map.items():
                    if src_key in obs.signals:
                        signals[dst_key] = float(obs.signals[src_key])

            # face.expression signal key mapping
            if src == "face.expression" and obs.signals:
                expr_map = {
                    "expression_happy": "em_happy", "expression_neutral": "em_neutral",
                    "expression_surprise": "em_surprise", "expression_angry": "em_angry",
                    "expression_contempt": "em_contempt", "expression_disgust": "em_disgust",
                    "expression_fear": "em_fear", "expression_sad": "em_sad",
                }
                for src_key, dst_key in expr_map.items():
                    if src_key in obs.signals:
                        signals[dst_key] = float(obs.signals[src_key])

            # face.parse → segmentation
            if src == "face.parse" and obs.data:
                results = getattr(obs.data, "results", [])
                if results:
                    seg = results[0].class_map
                    total = seg.size
                    if total > 0:
                        face_px = int(np.isin(seg, [1]).sum())
                        eye_px = int(np.isin(seg, [4, 5]).sum())
                        mouth_px = int(np.isin(seg, [11, 12, 13]).sum())
                        mouth_in_px = int(np.isin(seg, [11]).sum())
                        hair_px = int(np.isin(seg, [17]).sum())
                        glasses_px = int(np.isin(seg, [6]).sum())
                        brow_px = int(np.isin(seg, [2, 3]).sum())
                        nose_px = int(np.isin(seg, [10]).sum())

                        signals["seg_face"] = float(face_px / total)
                        signals["seg_eye"] = float(eye_px / total)
                        signals["seg_mouth"] = float(mouth_px / total)
                        signals["seg_hair"] = float(hair_px / total)

                        face_area = max(face_px + eye_px + brow_px + nose_px + mouth_px, 1)
                        signals["eye_visible_ratio"] = float(eye_px / face_area)
                        signals["mouth_open_ratio"] = float(mouth_in_px / face_area)
                        signals["glasses_ratio"] = float(glasses_px / face_area)

        # Frame quality fallback
        if "blur_score" not in signals:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            signals["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            signals["brightness"] = float(gray.mean())
            signals["contrast"] = float(gray.std())

        # Derived signals
        signals["backlight_score"] = max(0.0,
            signals.get("brightness", 0.0) - signals.get("face_exposure", 0.0))

        # Composites
        au6 = signals.get("au6_cheek_raiser", 0.0)
        au12 = signals.get("au12_lip_corner", 0.0)
        signals["duchenne_smile"] = (au6 + au12) / 5.0 * signals.get("warm_smile", 0.0)
        signals["wild_intensity"] = max(signals.get("au25_lips_part", 0.0),
            signals.get("au26_jaw_drop", 0.0)) / 3.0 * signals.get("wild_energy", 0.0)
        clip_max = max(signals.get(a, 0.0) for a in
                       ("warm_smile", "cool_gaze", "playful_face", "wild_energy"))
        signals["chill_score"] = signals.get("em_neutral", 0.0) * (1.0 - clip_max)

        # CLIP placeholders
        for axis in ("warm_smile", "cool_gaze", "playful_face", "wild_energy"):
            signals.setdefault(axis, 0.0)

        # Build result
        result = FrameResult(
            frame_idx=frame_id,
            image=image,
            signals=signals,
            face_detected=face_detected,
            face_count=face_count,
        )

        if not face_detected:
            return result

        # Gate (visualbind HeuristicStrategy)
        result.gate_reasons = self._gate.check_gate_from_signals(signals)
        result.gate_passed = len(result.gate_reasons) == 0

        # Expression (visualbind TreeStrategy)
        if self._bind:
            vec = self._to_vector(signals)
            scores = self._bind.predict(vec)
            if scores:
                result.expression = max(scores, key=scores.get)
                result.expression_conf = scores[result.expression]

        # Pose (visualbind TreeStrategy)
        if self._pose:
            vec = self._to_vector(signals)
            scores = self._pose.predict(vec)
            if scores:
                result.pose = max(scores, key=scores.get)
                result.pose_conf = scores[result.pose]

        return result

    def analyze_video(
        self, video_path: str | Path, fps: int = 2, max_frames: int = 500,
    ) -> list[FrameResult]:
        """비디오 분석."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps))
        results = []
        idx = 0
        extracted = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if idx % interval == 0 and extracted < max_frames:
                result = self.analyze_image(frame_bgr, extracted)
                result.timestamp_ms = idx / video_fps * 1000
                results.append(result)
                extracted += 1
                if extracted % 50 == 0:
                    logger.info("Processing %d frames", extracted)
            idx += 1

        cap.release()
        logger.info("Analyzed %d frames from %s", len(results), video_path.name)
        return results

    def select_frames(self, results: list[FrameResult], top_k: int = 10) -> list[FrameResult]:
        """SHOOT 프레임 중 다양성 기반 선택."""
        shoot = [r for r in results
                 if r.gate_passed and r.face_detected and r.expression and r.expression != "cut"]

        buckets: dict[str, FrameResult] = {}
        for r in shoot:
            key = f"{r.expression}|{r.pose}"
            if key not in buckets or r.expression_conf > buckets[key].expression_conf:
                buckets[key] = r

        return sorted(buckets.values(), key=lambda r: -r.expression_conf)[:top_k]

    def _to_vector(self, signals: dict) -> np.ndarray:
        from visualbind.signals import SIGNAL_FIELDS, normalize_signal
        return np.array([normalize_signal(signals.get(f, 0.0), f) for f in SIGNAL_FIELDS])
