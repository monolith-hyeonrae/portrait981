"""이미지 폴더에서 45D signal 추출 → parquet 저장.

라벨링된 이미지에 vpx analyzer를 실행하여 visualbind 학습용 signal을 추출한다.
momentscan 비디오 파이프라인 없이 이미지 단위로 동작.

Usage:
    uv run python scripts/extract_signals.py data/datasets/portrait-v1
    uv run python scripts/extract_signals.py data/datasets/portrait-v1 --limit 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("extract_signals")

from visualbind.signals import SIGNAL_FIELDS, normalize_signal


def make_frame(img_bgr: np.ndarray, frame_id: int = 0):
    """Create a visualbase Frame from a BGR image."""
    from visualbase.core.frame import Frame
    h, w = img_bgr.shape[:2]
    return Frame(data=img_bgr, frame_id=frame_id, t_src_ns=0, width=w, height=h)


def load_analyzers():
    """Load and initialize all vpx analyzers."""
    loaded = {}

    def try_load(name, cls_path, cls_name):
        try:
            mod = __import__(cls_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            instance = cls()
            instance.initialize()
            loaded[name] = instance
            logger.info("Loaded: %s", name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)

    try_load("face_detect", "vpx.face_detect", "FaceDetectionAnalyzer")
    try_load("face_au", "vpx.face_au", "FaceAUAnalyzer")
    try_load("face_expression", "vpx.face_expression", "ExpressionAnalyzer")
    try_load("head_pose", "vpx.head_pose", "HeadPoseAnalyzer")
    try_load("face_parse", "vpx.face_parse", "FaceParseAnalyzer")

    return loaded


def extract_signals_from_image(img_bgr: np.ndarray, analyzers: dict, frame_id: int = 0) -> dict[str, float]:
    """Run analyzers on a single image and return raw signal dict."""
    signals: dict[str, float] = {}
    frame = make_frame(img_bgr, frame_id)
    h, w = img_bgr.shape[:2]

    # --- Face Detection ---
    det = analyzers.get("face_detect")
    if not det:
        return signals

    det_obs = det.analyze(frame)
    if not det_obs or not det_obs.data or not det_obs.data.faces:
        return signals

    # Largest face = main rider
    faces = det_obs.data.faces
    face = max(faces, key=lambda f: f.area_ratio if hasattr(f, 'area_ratio') else 0)

    # Detection signals from FaceObservation
    signals["face_confidence"] = float(face.confidence)
    signals["face_area_ratio"] = float(face.area_ratio)
    signals["face_center_distance"] = float(face.center_distance)

    # Geometric head pose from face_detect (may be overridden by head_pose)
    signals["head_yaw_dev"] = abs(float(face.yaw)) if hasattr(face, 'yaw') else 0.0
    signals["head_pitch"] = float(face.pitch) if hasattr(face, 'pitch') else 0.0
    signals["head_roll"] = float(face.roll) if hasattr(face, 'roll') else 0.0

    # --- Action Units (12D) ---
    # Signal key mapping: au_au1 → au1_inner_brow
    AU_KEY_MAP = {
        "au_au1": "au1_inner_brow", "au_au2": "au2_outer_brow",
        "au_au4": "au4_brow_lowerer", "au_au5": "au5_upper_lid",
        "au_au6": "au6_cheek_raiser", "au_au9": "au9_nose_wrinkler",
        "au_au12": "au12_lip_corner", "au_au15": "au15_lip_depressor",
        "au_au17": "au17_chin_raiser", "au_au20": "au20_lip_stretcher",
        "au_au25": "au25_lips_part", "au_au26": "au26_jaw_drop",
    }
    face_au = analyzers.get("face_au")
    if face_au:
        try:
            au_obs = face_au.analyze(frame, deps={"face.detect": det_obs})
            if au_obs and au_obs.signals:
                for src_key, dst_key in AU_KEY_MAP.items():
                    if src_key in au_obs.signals:
                        signals[dst_key] = float(au_obs.signals[src_key])
        except Exception as e:
            logger.debug("AU failed: %s", e)

    # --- Emotions (8D) ---
    # Signal key mapping: expression_happy → em_happy
    EXPR_KEY_MAP = {
        "expression_happy": "em_happy", "expression_neutral": "em_neutral",
        "expression_surprise": "em_surprise", "expression_angry": "em_angry",
        "expression_contempt": "em_contempt", "expression_disgust": "em_disgust",
        "expression_fear": "em_fear", "expression_sad": "em_sad",
    }
    face_expr = analyzers.get("face_expression")
    if face_expr:
        try:
            expr_obs = face_expr.analyze(frame, deps={"face.detect": det_obs})
            if expr_obs and expr_obs.signals:
                for src_key, dst_key in EXPR_KEY_MAP.items():
                    if src_key in expr_obs.signals:
                        signals[dst_key] = float(expr_obs.signals[src_key])
        except Exception as e:
            logger.debug("Expression failed: %s", e)

    # --- Head Pose (3D, override geometric) ---
    head_pose = analyzers.get("head_pose")
    if head_pose:
        try:
            pose_obs = head_pose.analyze(frame, deps={"face.detect": det_obs})
            if pose_obs and pose_obs.signals:
                if "head_yaw" in pose_obs.signals:
                    signals["head_yaw_dev"] = abs(float(pose_obs.signals["head_yaw"]))
                if "head_pitch" in pose_obs.signals:
                    signals["head_pitch"] = float(pose_obs.signals["head_pitch"])
                if "head_roll" in pose_obs.signals:
                    signals["head_roll"] = float(pose_obs.signals["head_roll"])
        except Exception as e:
            logger.debug("Head pose failed: %s", e)

    # --- Face Parse / Segmentation — from BiSeNet 19-class map ---
    # Classes: 0=bg, 1=face, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
    #   6=glasses, 7=l_ear, 8=r_ear, 9=earring, 10=nose,
    #   11=mouth_in, 12=u_lip, 13=l_lip, 14=neck, 15=necklace,
    #   16=cloth, 17=hair, 18=hat
    face_parse = analyzers.get("face_parse")
    if face_parse:
        try:
            parse_obs = face_parse.analyze(frame, deps={"face.detect": det_obs})
            if parse_obs and parse_obs.data and parse_obs.data.results:
                seg = parse_obs.data.results[0].class_map
                total = seg.size
                if total > 0:
                    face_px = np.isin(seg, [1]).sum()
                    eye_px = np.isin(seg, [4, 5]).sum()
                    mouth_px = np.isin(seg, [11, 12, 13]).sum()
                    mouth_in_px = np.isin(seg, [11]).sum()
                    hair_px = np.isin(seg, [17]).sum()
                    glasses_px = np.isin(seg, [6]).sum()
                    brow_px = np.isin(seg, [2, 3]).sum()
                    nose_px = np.isin(seg, [10]).sum()

                    # Original 4D segmentation signals
                    signals["seg_face"] = float(face_px / total)
                    signals["seg_eye"] = float(eye_px / total)
                    signals["seg_mouth"] = float(mouth_px / total)
                    signals["seg_hair"] = float(hair_px / total)

                    # New signals: relative to face area
                    face_area = max(face_px + eye_px + brow_px + nose_px + mouth_px, 1)
                    signals["eye_visible_ratio"] = float(eye_px / face_area)
                    signals["mouth_open_ratio"] = float(mouth_in_px / face_area)
                    signals["glasses_ratio"] = float(glasses_px / face_area)
        except Exception as e:
            logger.debug("Face parse failed: %s", e)

    # --- Face Quality (5D) — computed from face crop ---
    bbox = face.bbox  # normalized (cx, cy, w, h) or (x1, y1, w, h)
    # Convert to pixel coords
    if all(0 <= v <= 1.1 for v in bbox):
        # Normalized coords
        bx, by, bw, bh = bbox
        px1 = int((bx - bw/2) * w) if bw < 0.5 else int(bx * w)
        py1 = int((by - bh/2) * h) if bh < 0.5 else int(by * h)
        px2 = px1 + int(bw * w)
        py2 = py1 + int(bh * h)
    else:
        px1, py1, px2, py2 = [int(v) for v in bbox]

    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)
    if px2 > px1 and py2 > py1:
        face_crop = img_bgr[py1:py2, px1:px2]
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        signals["face_blur"] = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
        signals["face_exposure"] = float(gray_face.mean())
        mean_val = gray_face.mean()
        signals["face_contrast"] = float(gray_face.std() / (mean_val + 1e-6))
        signals["clipped_ratio"] = float((gray_face > 250).sum() / gray_face.size)
        signals["crushed_ratio"] = float((gray_face < 5).sum() / gray_face.size)

    # --- Frame Quality (3D) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    signals["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    signals["brightness"] = float(gray.mean())
    signals["contrast"] = float(gray.std())

    # --- Backlight score (derived) ---
    # High frame brightness + low face exposure = backlit
    signals["backlight_score"] = max(0.0, signals.get("brightness", 0.0) - signals.get("face_exposure", 0.0))

    # --- CLIP Mood Axes (4D) — placeholder, needs portrait-score ---
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
    clip_max = max(signals.get(a, 0.0) for a in ("warm_smile", "cool_gaze", "playful_face", "wild_energy"))

    signals["duchenne_smile"] = (au6 + au12) / 5.0 * warm
    signals["wild_intensity"] = max(au25, au26) / 3.0 * wild
    signals["chill_score"] = em_neutral * (1.0 - clip_max)

    return signals


def main():
    parser = argparse.ArgumentParser(description="Extract 45D signals from labeled images")
    parser.add_argument("dataset", help="dataset directory (e.g. data/datasets/portrait-v1)")
    parser.add_argument("--output", "-o", default=None, help="output parquet path")
    parser.add_argument("--limit", type=int, default=0, help="limit number of images (0=all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    output_path = Path(args.output) if args.output else dataset_dir / "signals.parquet"

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)

    # Collect image files
    img_exts = {".jpg", ".jpeg", ".png", ".avif"}
    image_files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in img_exts
    )
    if args.limit > 0:
        image_files = image_files[:args.limit]

    logger.info("Found %d images in %s", len(image_files), images_dir)

    # Load analyzers
    logger.info("Loading analyzers...")
    analyzers = load_analyzers()
    logger.info("Loaded %d analyzers", len(analyzers))

    if "face_detect" not in analyzers:
        logger.error("face_detect is required but not loaded")
        sys.exit(1)

    # Extract signals
    import pandas as pd

    rows = []
    no_face = 0
    for i, img_path in enumerate(image_files):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info("Processing %d/%d: %s", i + 1, len(image_files), img_path.name)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Cannot read: %s", img_path.name)
            continue

        signals = extract_signals_from_image(img_bgr, analyzers, frame_id=i)
        if not signals:
            no_face += 1
            continue

        # Build normalized row
        row = {"filename": img_path.name}
        for f in SIGNAL_FIELDS:
            row[f] = normalize_signal(signals.get(f, 0.0), f)
        rows.append(row)

    if not rows:
        logger.error("No signals extracted")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d signals (%dD) to %s (no_face=%d)", len(df), len(SIGNAL_FIELDS), output_path, no_face)

    # Summary stats
    for col in list(SIGNAL_FIELDS)[:5]:
        vals = df[col]
        logger.info("  %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f", col, vals.mean(), vals.std(), vals.min(), vals.max())
    logger.info("  ... (%d more fields)", len(SIGNAL_FIELDS) - 5)


if __name__ == "__main__":
    main()
