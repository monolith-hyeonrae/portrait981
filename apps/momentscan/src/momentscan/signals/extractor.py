"""SignalExtractor — momentscan DAG 기반 signal 추출 (비디오/이미지 공용).

visualpath의 DAG를 이미지 단위로도 실행할 수 있게 하는 공유 모듈.
scripts/, visualbind 학습, predict_report 모두 이 클래스를 통해 signal을 추출.

momentscan의 face.quality(마스크 기반), face.gate(다중 조건) 포함.

Usage:
    from momentscan.signals import SignalExtractor

    extractor = SignalExtractor()
    result = extractor.extract(image_bgr)
    print(result.signals)       # 49D raw signal dict
    print(result.gate_passed)   # face.gate 판단
    print(result.gate_reasons)  # gate 실패 사유
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("momentscan.signals")


@dataclass
class SignalResult:
    """Signal 추출 결과."""
    signals: dict[str, float] = field(default_factory=dict)
    gate_passed: bool = False
    gate_reasons: list[str] = field(default_factory=list)
    face_detected: bool = False
    face_count: int = 0


class SignalExtractor:
    """momentscan DAG 기반 signal 추출기.

    visualpath의 analyzer들을 직접 호출하여 per-frame signal을 추출.
    비디오 파이프라인과 동일한 analyzer + face.quality + face.gate를 사용.
    """

    def __init__(self):
        self._analyzers: dict[str, object] = {}
        self._gate = None
        self._initialized = False

    def initialize(self):
        """Analyzer 로딩 + 초기화."""
        if self._initialized:
            return

        def _try_load(name: str, module_path: str, class_name: str):
            try:
                mod = __import__(module_path, fromlist=[class_name])
                cls = getattr(mod, class_name)
                instance = cls()
                instance.initialize()
                self._analyzers[name] = instance
                logger.info("SignalExtractor loaded: %s", name)
            except Exception as e:
                logger.warning("SignalExtractor failed to load %s: %s", name, e)

        # vpx analyzers (범용)
        _try_load("face.detect", "vpx.face_detect", "FaceDetectionAnalyzer")
        _try_load("face.au", "vpx.face_au", "FaceAUAnalyzer")
        _try_load("face.expression", "vpx.face_expression", "ExpressionAnalyzer")
        _try_load("head.pose", "vpx.head_pose", "HeadPoseAnalyzer")
        _try_load("face.parse", "vpx.face_parse", "FaceParseAnalyzer")

        # momentscan plugins (도메인 특화)
        _try_load("face.quality", "momentscan.face_quality", "FaceQualityAnalyzer")
        _try_load("frame.quality", "momentscan.frame_quality", "QualityAnalyzer")

        # Quality gate → visualbind HeuristicStrategy
        try:
            from visualbind.strategies.heuristic import HeuristicStrategy
            self._gate = HeuristicStrategy()
            logger.info("SignalExtractor: HeuristicStrategy gate loaded")
        except ImportError:
            self._gate = None
            logger.warning("SignalExtractor: HeuristicStrategy not available")

        self._initialized = True
        logger.info("SignalExtractor initialized: %d analyzers", len(self._analyzers))

    def extract(self, image: np.ndarray, frame_id: int = 0) -> SignalResult:
        """단일 이미지에서 signal 추출.

        momentscan DAG와 동일한 analyzer를 사용하여
        face.quality(마스크 기반) + face.gate(다중 조건) 포함.

        Args:
            image: BGR 이미지.
            frame_id: 프레임 번호.

        Returns:
            SignalResult with 49D signals + gate 판단.
        """
        if not self._initialized:
            self.initialize()

        result = SignalResult()
        signals = result.signals

        # Make frame
        from visualbase.core.frame import Frame
        h, w = image.shape[:2]
        frame = Frame(data=image, frame_id=frame_id, t_src_ns=0, width=w, height=h)

        # --- face.detect ---
        face_detect = self._analyzers.get("face.detect")
        if not face_detect:
            return result

        det_obs = face_detect.analyze(frame)
        if not det_obs or not det_obs.data or not det_obs.data.faces:
            return result

        result.face_detected = True
        result.face_count = len(det_obs.data.faces)

        # Main face (largest)
        face = max(det_obs.data.faces, key=lambda f: f.area_ratio)
        signals["face_confidence"] = float(face.confidence)
        signals["face_area_ratio"] = float(face.area_ratio)
        signals["face_center_distance"] = float(face.center_distance)
        signals["head_yaw_dev"] = abs(float(face.yaw)) if hasattr(face, 'yaw') else 0.0
        signals["head_pitch"] = float(face.pitch) if hasattr(face, 'pitch') else 0.0
        signals["head_roll"] = float(face.roll) if hasattr(face, 'roll') else 0.0

        deps = {"face.detect": det_obs}

        # --- face.au (12D) ---
        self._run_au(frame, deps, signals)

        # --- face.expression (8D) ---
        self._run_expression(frame, deps, signals)

        # --- head.pose (3D, override geometric) ---
        self._run_head_pose(frame, deps, signals)

        # --- face.parse → segmentation + face.quality (mask-based) ---
        parse_obs = self._run_face_parse(frame, deps, signals)

        # --- face.quality (5D, mask-based) ---
        self._run_face_quality(frame, deps, parse_obs, signals)

        # --- frame.quality (3D) ---
        self._run_frame_quality(frame, signals)

        # --- Backlight score (derived) ---
        signals["backlight_score"] = max(0.0,
            signals.get("brightness", 0.0) - signals.get("face_exposure", 0.0))

        # --- CLIP Mood Axes (4D) — placeholder ---
        for axis in ("warm_smile", "cool_gaze", "playful_face", "wild_energy"):
            signals.setdefault(axis, 0.0)

        # --- Composites (3D) ---
        au6 = signals.get("au6_cheek_raiser", 0.0)
        au12 = signals.get("au12_lip_corner", 0.0)
        au25 = signals.get("au25_lips_part", 0.0)
        au26 = signals.get("au26_jaw_drop", 0.0)
        warm = signals.get("warm_smile", 0.0)
        wild = signals.get("wild_energy", 0.0)
        em_neutral = signals.get("em_neutral", 0.0)
        clip_max = max(signals.get(a, 0.0) for a in
                       ("warm_smile", "cool_gaze", "playful_face", "wild_energy"))
        signals["duchenne_smile"] = (au6 + au12) / 5.0 * warm
        signals["wild_intensity"] = max(au25, au26) / 3.0 * wild
        signals["chill_score"] = em_neutral * (1.0 - clip_max)

        # --- Quality gate (visualbind HeuristicStrategy, 모든 signal 수집 후) ---
        self._run_gate(signals, result)

        return result

    # --- Analyzer runners ---

    _AU_KEY_MAP = {
        "au_au1": "au1_inner_brow", "au_au2": "au2_outer_brow",
        "au_au4": "au4_brow_lowerer", "au_au5": "au5_upper_lid",
        "au_au6": "au6_cheek_raiser", "au_au9": "au9_nose_wrinkler",
        "au_au12": "au12_lip_corner", "au_au15": "au15_lip_depressor",
        "au_au17": "au17_chin_raiser", "au_au20": "au20_lip_stretcher",
        "au_au25": "au25_lips_part", "au_au26": "au26_jaw_drop",
    }

    _EXPR_KEY_MAP = {
        "expression_happy": "em_happy", "expression_neutral": "em_neutral",
        "expression_surprise": "em_surprise", "expression_angry": "em_angry",
        "expression_contempt": "em_contempt", "expression_disgust": "em_disgust",
        "expression_fear": "em_fear", "expression_sad": "em_sad",
    }

    def _run_au(self, frame, deps, signals):
        au = self._analyzers.get("face.au")
        if not au:
            return
        try:
            obs = au.analyze(frame, deps=deps)
            if obs and obs.signals:
                for src, dst in self._AU_KEY_MAP.items():
                    if src in obs.signals:
                        signals[dst] = float(obs.signals[src])
        except Exception as e:
            logger.debug("AU failed: %s", e)

    def _run_expression(self, frame, deps, signals):
        expr = self._analyzers.get("face.expression")
        if not expr:
            return
        try:
            obs = expr.analyze(frame, deps=deps)
            if obs and obs.signals:
                for src, dst in self._EXPR_KEY_MAP.items():
                    if src in obs.signals:
                        signals[dst] = float(obs.signals[src])
        except Exception as e:
            logger.debug("Expression failed: %s", e)

    def _run_head_pose(self, frame, deps, signals):
        hp = self._analyzers.get("head.pose")
        if not hp:
            return
        try:
            obs = hp.analyze(frame, deps=deps)
            if obs and obs.signals:
                if "head_yaw" in obs.signals:
                    signals["head_yaw_dev"] = abs(float(obs.signals["head_yaw"]))
                if "head_pitch" in obs.signals:
                    signals["head_pitch"] = float(obs.signals["head_pitch"])
                if "head_roll" in obs.signals:
                    signals["head_roll"] = float(obs.signals["head_roll"])
        except Exception as e:
            logger.debug("HeadPose failed: %s", e)

    def _run_face_parse(self, frame, deps, signals):
        fp = self._analyzers.get("face.parse")
        if not fp:
            return None
        try:
            obs = fp.analyze(frame, deps=deps)
            if obs and obs.data and obs.data.results:
                seg = obs.data.results[0].class_map
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
            return obs
        except Exception as e:
            logger.debug("FaceParse failed: %s", e)
            return None

    def _run_face_quality(self, frame, deps, parse_obs, signals):
        fq = self._analyzers.get("face.quality")
        if not fq:
            # Fallback: simple crop-based
            self._fallback_face_quality(frame, deps, signals)
            return
        try:
            fq_deps = dict(deps)
            if parse_obs:
                fq_deps["face.parse"] = parse_obs
            obs = fq.analyze(frame, deps=fq_deps)
            if obs and obs.signals:
                signals["face_blur"] = float(obs.signals.get("face_blur", 0.0))
                signals["face_exposure"] = float(obs.signals.get("face_exposure", 0.0))
                signals["face_contrast"] = float(obs.signals.get("face_contrast", 0.0))
                signals["clipped_ratio"] = float(obs.signals.get("clipped_ratio", 0.0))
                signals["crushed_ratio"] = float(obs.signals.get("crushed_ratio", 0.0))
        except Exception as e:
            logger.debug("FaceQuality failed: %s, using fallback", e)
            self._fallback_face_quality(frame, deps, signals)

    def _fallback_face_quality(self, frame, deps, signals):
        """Simple crop-based quality when FaceQualityAnalyzer unavailable."""
        det_obs = deps.get("face.detect")
        if not det_obs or not det_obs.data or not det_obs.data.faces:
            return
        face = max(det_obs.data.faces, key=lambda f: f.area_ratio)
        h, w = frame.data.shape[:2]
        bbox = face.bbox
        if all(0 <= v <= 1.1 for v in bbox):
            bx, by, bw, bh = bbox
            px1 = int(bx * w)
            py1 = int(by * h)
            px2 = px1 + int(bw * w)
            py2 = py1 + int(bh * h)
        else:
            px1, py1, px2, py2 = [int(v) for v in bbox]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        if px2 > px1 and py2 > py1:
            gray = cv2.cvtColor(frame.data[py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)
            signals["face_blur"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            signals["face_exposure"] = float(gray.mean())
            mean_val = gray.mean()
            signals["face_contrast"] = float(gray.std() / (mean_val + 1e-6))
            signals["clipped_ratio"] = float((gray > 250).sum() / gray.size)
            signals["crushed_ratio"] = float((gray < 5).sum() / gray.size)

    def _run_gate(self, signals, result):
        """Quality gate via visualbind HeuristicStrategy."""
        if not self._gate:
            result.gate_passed = True
            return
        fails = self._gate.check_gate_from_signals(signals)
        result.gate_passed = len(fails) == 0
        result.gate_reasons = fails

    def _run_frame_quality(self, frame, signals):
        fq = self._analyzers.get("frame.quality")
        if fq:
            try:
                obs = fq.analyze(frame)
                if obs and obs.signals:
                    signals["blur_score"] = float(obs.signals.get("blur_score", 0.0))
                    signals["brightness"] = float(obs.signals.get("brightness", 0.0))
                    signals["contrast"] = float(obs.signals.get("contrast", 0.0))
                    return
            except Exception as e:
                logger.debug("FrameQuality failed: %s", e)

        # Fallback
        gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
        signals["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        signals["brightness"] = float(gray.mean())
        signals["contrast"] = float(gray.std())

    def extract_from_video(
        self, video_path: Path | str, fps: int = 2, max_frames: int = 500,
    ) -> list[tuple[int, np.ndarray, SignalResult]]:
        """비디오에서 프레임별 signal 추출.

        Returns:
            list of (frame_index, image, SignalResult)
        """
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
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0 and extracted < max_frames:
                result = self.extract(frame, frame_id=extracted)
                results.append((extracted, frame, result))
                extracted += 1
            idx += 1

        cap.release()
        logger.info("Extracted signals from %d frames (%s)", len(results), video_path.name)
        return results
