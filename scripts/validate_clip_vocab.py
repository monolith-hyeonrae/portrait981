#!/usr/bin/env python3
"""CLIP axis vs specialist model 검증 스크립트.

같은 프레임에서 전문 모델(HSEmotion, LibreFace AU, 6DRepNet)과
CLIP axis scores를 나란히 비교하여 축 기반 앙상블의 변별력을 검증한다.

Usage:
    uv run python scripts/validate_clip_vocab.py ~/Videos/test.mp4 --samples 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CLIP vocab vs specialist model validation")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--samples", type=int, default=15, help="Number of frames to sample")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for ML models")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    # ── Open video ──
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path.name} ({total_frames} frames, {video_fps:.1f} fps)")

    # Evenly spaced sample indices
    indices = np.linspace(0, total_frames - 1, args.samples, dtype=int)

    # ── Initialize analyzers ──
    print("Loading models...")
    from visualbase import Frame
    from vpx.face_detect import FaceDetectionAnalyzer
    from vpx.face_expression import ExpressionAnalyzer
    from vpx.face_au import FaceAUAnalyzer
    from vpx.head_pose import HeadPoseAnalyzer
    from vpx.vision_embed import ShotQualityAnalyzer

    face_det = FaceDetectionAnalyzer(device=args.device)
    face_exp = ExpressionAnalyzer(device=args.device)
    face_au = FaceAUAnalyzer(device=args.device)
    head_pose = HeadPoseAnalyzer(device=args.device)
    shot_qual = ShotQualityAnalyzer(
        enable_aesthetic=True, enable_caption=False, device=args.device,
    )

    analyzers = [face_det, face_exp, face_au, head_pose, shot_qual]
    for a in analyzers:
        a.initialize()
    print("All models loaded.\n")

    # ── Process frames ──
    results = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, image = cap.read()
        if not ret:
            continue

        t_sec = idx / video_fps
        t_ns = int(t_sec * 1e9)
        frame = Frame.from_array(data=image, frame_id=int(idx), t_src_ns=t_ns)

        # Pipeline: face.detect → deps → downstream
        face_obs = face_det.process(frame)
        if face_obs is None or not face_obs.data or not face_obs.data.faces:
            print(f"[{i+1:2d}/{args.samples}] Frame {idx:5d} (t={t_sec:6.1f}s) — no face detected, skipping")
            continue

        deps = {"face.detect": face_obs}
        exp_obs = face_exp.process(frame, deps=deps)
        au_obs = face_au.process(frame, deps=deps)
        hp_obs = head_pose.process(frame, deps=deps)
        sq_obs = shot_qual.process(frame, deps=deps)

        # ── Extract specialist results ──
        record = {"frame": int(idx), "t_sec": round(t_sec, 1)}

        # Face detection basics
        face = max(face_obs.data.faces, key=lambda f: f.area_ratio)
        record["face_conf"] = round(face.confidence, 2)
        record["face_area"] = round(face.area_ratio, 4)

        # Expression (HSEmotion)
        emotions = {}
        if exp_obs and exp_obs.data and exp_obs.data.faces:
            ef = exp_obs.data.faces[0]
            for k, v in ef.signals.items():
                if k.startswith("em_"):
                    emotions[k[3:]] = round(v, 3)
        record["emotions"] = emotions
        # Top emotion
        if emotions:
            top_emo = max(emotions, key=emotions.get)
            record["top_emotion"] = f"{top_emo}({emotions[top_emo]:.2f})"
        else:
            record["top_emotion"] = "N/A"

        # AU (LibreFace)
        aus = {}
        if au_obs and au_obs.data and au_obs.data.au_intensities:
            aus = {k: round(v, 1) for k, v in au_obs.data.au_intensities[0].items() if v >= 0.5}
        record["aus"] = aus
        # Key AUs summary
        au12 = au_obs.data.au_intensities[0].get("AU12", 0) if au_obs and au_obs.data and au_obs.data.au_intensities else 0
        au25 = au_obs.data.au_intensities[0].get("AU25", 0) if au_obs and au_obs.data and au_obs.data.au_intensities else 0
        au26 = au_obs.data.au_intensities[0].get("AU26", 0) if au_obs and au_obs.data and au_obs.data.au_intensities else 0
        record["au_smile"] = round(au12, 1)  # AU12 = lip corner puller
        record["au_mouth"] = round(max(au25, au26), 1)  # mouth open

        # Head pose (6DRepNet)
        if hp_obs and hp_obs.data and hp_obs.data.estimates:
            hp = hp_obs.data.estimates[0]
            record["head_yaw"] = round(hp.yaw, 1)
            record["head_pitch"] = round(hp.pitch, 1)
        else:
            record["head_yaw"] = 0.0
            record["head_pitch"] = 0.0

        # CLIP axes
        clip_axes = []
        clip_score = 0.0
        if sq_obs and sq_obs.metadata:
            axes = sq_obs.metadata.get("_clip_axes")
            if axes:
                clip_axes = [(ax.name, round(ax.score, 3), ax.active, ax.action) for ax in axes]
            bd = sq_obs.metadata.get("_clip_breakdown")
            if bd:
                clip_score = round(bd.score, 3)

        record["clip_score"] = clip_score
        record["clip_axes"] = clip_axes

        results.append(record)

        # ── Print frame summary ──
        print(f"[{i+1:2d}/{args.samples}] Frame {idx:5d} (t={t_sec:5.1f}s)")
        print(f"  Specialist:")
        print(f"    Expression : {record['top_emotion']}")
        if emotions:
            emo_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(emotions.items(), key=lambda x: -x[1])[:4])
            print(f"                 {emo_str}")
        print(f"    AU(smile)  : AU12={au12:.1f}  AU25={au25:.1f}  AU26={au26:.1f}")
        if aus:
            au_str = "  ".join(f"{k}={v}" for k, v in sorted(aus.items()))
            print(f"    AU(active) : {au_str}")
        print(f"    Head pose  : yaw={record['head_yaw']:+.1f}°  pitch={record['head_pitch']:+.1f}°")

        print(f"  CLIP (score={clip_score:.3f}):")
        for name, score, active, action in clip_axes:
            marker = " ***" if active else ""
            print(f"    {name:16s}  {score:.3f}  ({action}){marker}")
        print()

    cap.release()
    for a in analyzers:
        a.cleanup()

    # ── Summary: correlation analysis ──
    if len(results) < 3:
        print("Not enough frames with faces for analysis.")
        return

    print(f"\n{'='*80}")
    print(f"AXIS CORRELATION ANALYSIS ({len(results)} frames)")
    print(f"{'='*80}\n")

    # 1. smile detection: AU12 vs CLIP disney_smile axis
    print("1. Smile detection: AU12 (LibreFace) vs CLIP disney_smile axis")
    print(f"   {'Frame':>6s}  {'AU12':>5s}  {'HSEmo':>12s}  {'disney_smile':>13s}  {'active':>6s}  Match?")
    print(f"   {'─'*6}  {'─'*5}  {'─'*12}  {'─'*13}  {'─'*6}  {'─'*6}")

    agree_smile = 0
    for r in results:
        au12_val = r["au_smile"]
        is_smiling_au = au12_val >= 1.5  # AU12 threshold for clear smile

        # Find disney_smile axis
        ds_score = 0.0
        ds_active = False
        for name, score, active, action in r["clip_axes"]:
            if name == "disney_smile":
                ds_score = score
                ds_active = active
                break

        match = "✓" if (is_smiling_au == ds_active) else "✗"
        if is_smiling_au == ds_active:
            agree_smile += 1

        emo = r["top_emotion"]
        print(f"   {r['frame']:6d}  {au12_val:5.1f}  {emo:>12s}  {ds_score:13.3f}  {'YES' if ds_active else 'no':>6s}  {match}")

    pct = agree_smile / len(results) * 100
    print(f"\n   Agreement: {agree_smile}/{len(results)} ({pct:.0f}%)\n")

    # 2. Per-axis score distribution
    print("2. Per-axis score statistics across frames")
    axis_names = [name for name, _, _, _ in results[0]["clip_axes"]] if results[0]["clip_axes"] else []
    for ax_name in axis_names:
        scores = []
        active_count = 0
        for r in results:
            for name, score, active, action in r["clip_axes"]:
                if name == ax_name:
                    scores.append(score)
                    if active:
                        active_count += 1
                    break
        if scores:
            print(f"   {ax_name:16s}  min={min(scores):.3f}  max={max(scores):.3f}  "
                  f"mean={np.mean(scores):.3f}  std={np.std(scores):.3f}  "
                  f"active={active_count}/{len(scores)}")
    print()

    # 3. Activation pattern
    print("3. Axis activation pattern per frame")
    print(f"   {'Frame':>6s}  {'Axes active':>40s}")
    print(f"   {'─'*6}  {'─'*40}")
    for r in results:
        active_axes = [name for name, _, active, _ in r["clip_axes"] if active]
        ax_str = ", ".join(active_axes) if active_axes else "(none)"
        print(f"   {r['frame']:6d}  {ax_str}")
    print()

    # 4. Axis score spread (max - min per frame)
    print("4. Per-frame axis score spread (변별력 지표)")
    print(f"   {'Frame':>6s}  {'Max':>6s}  {'Min':>6s}  {'Spread':>7s}  Visual")
    print(f"   {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*20}")
    spreads = []
    for r in results:
        if r["clip_axes"]:
            ax_scores = [score for _, score, _, _ in r["clip_axes"]]
            s_max = max(ax_scores)
            s_min = min(ax_scores)
            spread = s_max - s_min
            spreads.append(spread)
            bar_len = int(spread * 40)
            bar = "█" * bar_len
            print(f"   {r['frame']:6d}  {s_max:.3f}  {s_min:.3f}  {spread:7.3f}  {bar}")

    if spreads:
        print(f"\n   Spread: mean={np.mean(spreads):.3f}  std={np.std(spreads):.3f}")
        if np.mean(spreads) < 0.05:
            print("   ⚠ Very small spread — axes may not discriminate well")
        elif np.mean(spreads) < 0.15:
            print("   △ Moderate spread — some discrimination")
        else:
            print("   ✓ Good spread — meaningful discrimination between axes")


if __name__ == "__main__":
    main()
