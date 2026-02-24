#!/usr/bin/env python3
"""CLIP 감정 성격(emotion quality) 변별력 검증 v2.

EMA 없이, CLIP이 감정의 질적 차이를 구분할 수 있는지 다중 영상으로 검증.

Usage:
    uv run python scripts/validate_clip_emotion_quality.py \
        ~/Videos/reaction_test/test_2.mp4 ~/Videos/reaction_test/test_3.mp4 \
        --samples 25
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ── 감정 성격 vocab ──
_PROMPTS: List[Tuple[str, str]] = [
    # ── 미소의 종류 ──
    ("smile_warm", "a person with a warm genuine heartfelt smile"),
    ("smile_shy", "a person with a shy bashful timid smile"),
    ("smile_confident", "a person with a confident charming grin"),
    ("smile_bright", "a person with a bright radiant beaming smile"),
    ("smile_gentle", "a person with a subtle gentle soft smile"),
    ("smile_disney", "a sweet charming adorable fairy-tale smile"),
    ("smile_teeth", "a person smiling big with teeth showing"),
    ("smile_eyes", "a person smiling with sparkling happy eyes"),
    ("smile_half", "a person with a half smile and knowing look"),
    ("smile_forced", "a person with a forced awkward fake smile"),
    # ── 웃음의 종류 ──
    ("laugh_big", "a person laughing with open mouth full of joy"),
    ("laugh_giggle", "a person giggling and trying not to laugh"),
    ("laugh_tears", "a person laughing so hard with tears of joy"),
    ("laugh_hold", "a person trying to hold back laughter"),
    # ── 카리스마 ──
    ("charisma_cool", "a person with a cool confident smirk"),
    ("charisma_model", "a person looking stylish like a fashion model"),
    ("charisma_mysterious", "a person with a mysterious alluring gaze"),
    ("charisma_star", "a person with movie star charisma and presence"),
    ("charisma_fierce", "a person looking fierce and powerful"),
    ("charisma_brooding", "a person with a brooding smoldering look"),
    ("charisma_frown", "a person frowning intensely with charismatic presence"),
    ("charisma_rebel", "a rebellious James Dean look with squinted eyes"),
    ("charisma_serious", "a serious focused look with strong jawline"),
    # ── 장난 / 재미 ──
    ("playful_tongue", "a person sticking tongue out playfully"),
    ("playful_wink", "a person winking at the camera"),
    ("playful_puff", "a person puffing cheeks like a blowfish"),
    ("playful_pout", "a person making a duck face or pout"),
    ("playful_silly", "a person making a funny silly face"),
    # ── 귀여움 ──
    ("cute_adorable", "a person with a cute adorable puppy-like expression"),
    ("cute_innocent", "a person with big innocent doe eyes"),
    ("cute_tilt", "a person tilting head cutely"),
    # ── 강렬 / 집중 ──
    ("intense_stare", "a person with an intense determined stare"),
    ("intense_frown", "a person frowning with furrowed brow"),
    ("intense_squint", "a person squinting with a tough expression"),
    ("intense_jaw", "a person clenching jaw with determination"),
    # ── 회피 ──
    ("avoid_sneeze", "a person mid-sneeze with distorted face"),
    ("avoid_yawn", "a person yawning with mouth wide open"),
    ("avoid_blur", "a distorted blurry motion-smeared face"),
    ("avoid_chew", "a person caught mid-chew with food in mouth"),
    ("avoid_unflattering", "an awkward unflattering facial expression"),
    # ── 기본 감정 대조군 ──
    ("emo_happy", "a person looking happy and joyful"),
    ("emo_sad", "a person looking sad and upset"),
    ("emo_angry", "a person looking angry and frustrated"),
    ("emo_surprised", "a person looking surprised with wide eyes"),
    ("emo_neutral", "a person with a neutral calm expression"),
    ("emo_disgusted", "a person looking disgusted or repulsed"),
]

_TEXTS = [t for _, t in _PROMPTS]
_LABELS = [l for l, _ in _PROMPTS]
_GROUPS: Dict[str, List[int]] = {}
for _i, _l in enumerate(_LABELS):
    _GROUPS.setdefault(_l.split("_")[0], []).append(_i)


def process_video(
    video_path: Path,
    num_samples: int,
    face_det, face_exp, face_au,
    clip_model, clip_preprocess, clip_dtype, text_embeds,
    device: str,
):
    """Process a single video and return per-frame records."""
    from visualbase import Frame
    from vpx.vision_embed.crop import face_crop
    from PIL import Image
    import torch

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = np.linspace(0, total - 1, num_samples, dtype=int)

    print(f"\n{'━'*80}")
    print(f"  {video_path.name}  ({total} frames, {fps:.1f} fps, {total/fps:.0f}s)")
    print(f"{'━'*80}")

    results = []
    for fi, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, image = cap.read()
        if not ret:
            continue

        t_sec = idx / fps
        frame = Frame.from_array(data=image, frame_id=int(idx), t_src_ns=int(t_sec * 1e9))

        face_obs = face_det.process(frame)
        if not face_obs or not face_obs.data or not face_obs.data.faces:
            print(f"  [{fi+1:2d}/{num_samples}] Frame {idx:5d} — no face")
            continue

        deps = {"face.detect": face_obs}

        # Expression
        exp_obs = face_exp.process(frame, deps=deps)
        emotions = {}
        if exp_obs and exp_obs.data and exp_obs.data.faces:
            for k, v in exp_obs.data.faces[0].signals.items():
                if k.startswith("em_"):
                    emotions[k[3:]] = round(v, 3)

        # AU
        au_obs = face_au.process(frame, deps=deps)
        aus = {}
        if au_obs and au_obs.data and au_obs.data.au_intensities:
            aus = {k: round(v, 2) for k, v in au_obs.data.au_intensities[0].items()}

        # CLIP (no EMA)
        face = max(face_obs.data.faces, key=lambda f: f.area_ratio)
        img_w, img_h = face_obs.data.image_size
        nx, ny, nw, nh = face.bbox
        px, py, pw, ph = int(nx*img_w), int(ny*img_h), int(nw*img_w), int(nh*img_h)

        portrait_img, _ = face_crop(
            image, (px, py, pw, ph), expand=2.2, crop_ratio="1:1", y_shift=0.3,
        )

        rgb = cv2.cvtColor(portrait_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        with torch.no_grad():
            tensor = clip_preprocess(pil_img).unsqueeze(0).to(device=device, dtype=clip_dtype)
            img_embed = clip_model.encode_image(tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            raw_sims = (img_embed @ text_embeds.T).squeeze(0).cpu().numpy()

        # Group stats
        group_maxes = {}
        group_best = {}
        for group, idxs in _GROUPS.items():
            sims = [(raw_sims[i], _LABELS[i], _TEXTS[i]) for i in idxs]
            best = max(sims, key=lambda x: x[0])
            group_maxes[group] = float(best[0])
            group_best[group] = (best[1], best[2], float(best[0]))

        top_idx = np.argsort(raw_sims)[::-1][:8]
        top_prompts = [(_LABELS[i], round(float(raw_sims[i]), 4)) for i in top_idx]

        record = {
            "video": video_path.name, "frame": int(idx), "t_sec": round(t_sec, 1),
            "emotions": emotions, "aus": aus,
            "group_maxes": group_maxes, "group_best": group_best,
            "top_prompts": top_prompts, "raw_sims": raw_sims,
        }
        results.append(record)

        # Print
        top_emo = max(emotions, key=emotions.get) if emotions else "N/A"
        top_emo_val = emotions.get(top_emo, 0)
        au12 = aus.get("AU12", 0)
        au4 = aus.get("AU4", 0)  # brow lowerer (찡그림)

        print(f"  [{fi+1:2d}/{num_samples}] Frame {idx:5d} (t={t_sec:5.1f}s)  "
              f"HSE:{top_emo}={top_emo_val:.2f}  AU4={au4:.1f} AU12={au12:.1f}")

        ranked = sorted(group_maxes.items(), key=lambda x: -x[1])
        for g, s in ranked[:5]:
            bl, _, _ = group_best[g]
            m = "◆" if s > 0.28 else "◇" if s > 0.25 else " "
            print(f"      {m} {g:10s} {s:.4f}  {bl}")

        print(f"      top: {' '.join(l for l,_ in top_prompts[:5])}")

    cap.release()
    return results


def analyze(results: list, video_name: str = ""):
    """Print analysis for one video's results."""
    if len(results) < 3:
        print(f"  Not enough frames for {video_name}.")
        return

    header = f"ANALYSIS: {video_name}" if video_name else f"ANALYSIS ({len(results)} frames)"
    print(f"\n{'='*80}")
    print(f"  {header}")
    print(f"{'='*80}\n")

    # 1. Group spread
    print("  1. Group similarity spread (변별력)")
    print(f"     {'Group':12s}  {'Min':>7s}  {'Max':>7s}  {'Mean':>7s}  {'Spread':>7s}  Visual")

    all_sims = {}
    for r in results:
        for g, s in r["group_maxes"].items():
            all_sims.setdefault(g, []).append(s)

    for g in sorted(all_sims.keys(), key=lambda x: -(max(all_sims[x])-min(all_sims[x]))):
        vals = all_sims[g]
        mn, mx, mean = min(vals), max(vals), np.mean(vals)
        spread = mx - mn
        bar = "█" * int(spread * 400)
        verdict = "✓" if spread > 0.035 else "△" if spread > 0.020 else "✗"
        print(f"     {g:12s}  {mn:7.4f}  {mx:7.4f}  {mean:7.4f}  {spread:7.4f}  {bar} {verdict}")

    # 2. Smile quality detail
    print(f"\n  2. Smile/Laugh quality (happy > 0.5 vs ≤ 0.5)")
    qual_groups = ["smile", "laugh", "playful", "charisma", "cute", "intense"]
    happy = [r for r in results if r["emotions"].get("happy", 0) > 0.5]
    unhappy = [r for r in results if r["emotions"].get("happy", 0) <= 0.5]

    if happy and unhappy:
        print(f"     {'':12s}", end="")
        for qg in qual_groups:
            print(f"  {qg:>10s}", end="")
        print()

        happy_means = {qg: np.mean([r["group_maxes"].get(qg, 0) for r in happy]) for qg in qual_groups}
        unhappy_means = {qg: np.mean([r["group_maxes"].get(qg, 0) for r in unhappy]) for qg in qual_groups}

        print(f"     {'happy mean':12s}", end="")
        for qg in qual_groups:
            print(f"  {happy_means[qg]:10.4f}", end="")
        print()
        print(f"     {'other mean':12s}", end="")
        for qg in qual_groups:
            print(f"  {unhappy_means[qg]:10.4f}", end="")
        print()
        print(f"     {'delta':12s}", end="")
        for qg in qual_groups:
            d = happy_means[qg] - unhappy_means[qg]
            marker = "↑" if d > 0.01 else "↓" if d < -0.01 else "="
            print(f"  {d:+9.4f}{marker}", end="")
        print()

    # 3. Playful group detail
    print(f"\n  3. Playful group detail")
    print(f"     {'Frame':>6s}  {'t':>5s}  {'playful':>8s}  {'best prompt':>25s}  {'HSEmo':>15s}")
    for r in results:
        pv = r["group_maxes"].get("playful", 0)
        if "playful" in r["group_best"]:
            bl, _, _ = r["group_best"]["playful"]
        else:
            bl = "N/A"
        top_emo = max(r["emotions"], key=r["emotions"].get) if r["emotions"] else "N/A"
        ev = r["emotions"].get(top_emo, 0)
        m = "◆" if pv > 0.28 else "◇" if pv > 0.26 else " "
        print(f"     {r['frame']:6d}  {r['t_sec']:5.1f}  {pv:8.4f}  {m} {bl:23s}  {top_emo}={ev:.2f}")

    # 4. Charisma + Intense
    print(f"\n  4. Charisma + Intense (CLIP-only)")
    print(f"     {'Frame':>6s}  {'t':>5s}  {'charism':>8s}  {'intense':>8s}  "
          f"{'best charisma':>20s}  {'best intense':>18s}  {'HSEmo':>15s}")
    for r in results:
        ch = r["group_maxes"].get("charisma", 0)
        it = r["group_maxes"].get("intense", 0)
        ch_bl = r["group_best"].get("charisma", ("N/A","",""))[0]
        it_bl = r["group_best"].get("intense", ("N/A","",""))[0]
        top_emo = max(r["emotions"], key=r["emotions"].get) if r["emotions"] else "N/A"
        ev = r["emotions"].get(top_emo, 0)
        mc = "★" if ch > 0.28 else "☆" if ch > 0.26 else " "
        mi = "★" if it > 0.28 else "☆" if it > 0.26 else " "
        print(f"     {r['frame']:6d}  {r['t_sec']:5.1f}  {ch:8.4f}{mc} {it:8.4f}{mi} "
              f"{ch_bl:>20s}  {it_bl:>18s}  {top_emo}={ev:.2f}")

    # 5. Top-1 prompt distribution
    print(f"\n  5. Top-1 prompt distribution")
    top1 = Counter(r["top_prompts"][0][0] for r in results if r["top_prompts"])
    for label, count in top1.most_common():
        bar = "█" * count
        print(f"     {label:22s}  {count:2d}  {bar}")

    # 6. Top-1 group distribution
    print(f"\n  6. Top-1 group distribution")
    top1g = Counter(r["top_prompts"][0][0].split("_")[0] for r in results if r["top_prompts"])
    for g, count in top1g.most_common():
        bar = "█" * count
        print(f"     {g:12s}  {count:2d}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="CLIP emotion quality validation v2 (no EMA)")
    parser.add_argument("video", nargs="+", help="Path to video file(s)")
    parser.add_argument("--samples", type=int, default=25, help="Frames per video")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    video_paths = [Path(v) for v in args.video]
    for vp in video_paths:
        if not vp.exists():
            print(f"Video not found: {vp}")
            sys.exit(1)

    # ── Load models ──
    print("Loading models...")
    from vpx.face_detect import FaceDetectionAnalyzer
    from vpx.face_expression import ExpressionAnalyzer
    from vpx.face_au import FaceAUAnalyzer

    face_det = FaceDetectionAnalyzer(device=args.device)
    face_exp = ExpressionAnalyzer(device=args.device)
    face_au = FaceAUAnalyzer(device=args.device)
    for a in [face_det, face_exp, face_au]:
        a.initialize()

    import torch
    import open_clip

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(args.device).eval()
    use_fp16 = "cuda" in args.device
    clip_dtype = torch.float16 if use_fp16 else torch.float32
    if use_fp16:
        clip_model = clip_model.half()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        tokens = tokenizer(_TEXTS).to(args.device)
        text_embeds = clip_model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    print(f"CLIP: {len(_TEXTS)} prompts, {len(_GROUPS)} groups ({', '.join(_GROUPS.keys())})")
    print("All models loaded.")

    # ── Process each video ──
    all_results = {}
    for vp in video_paths:
        res = process_video(
            vp, args.samples,
            face_det, face_exp, face_au,
            clip_model, clip_preprocess, clip_dtype, text_embeds,
            args.device,
        )
        all_results[vp.name] = res
        analyze(res, vp.name)

    # ── Cross-video comparison ──
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"  CROSS-VIDEO COMPARISON")
        print(f"{'='*80}\n")

        print(f"  Group mean similarity by video:")
        all_groups_sorted = sorted(_GROUPS.keys())
        print(f"  {'Video':16s}", end="")
        for g in all_groups_sorted:
            print(f"  {g:>8s}", end="")
        print()

        for vname, res in all_results.items():
            print(f"  {vname:16s}", end="")
            for g in all_groups_sorted:
                vals = [r["group_maxes"].get(g, 0) for r in res]
                print(f"  {np.mean(vals):8.4f}", end="")
            print()

    for a in [face_det, face_exp, face_au]:
        a.cleanup()

    print("\nDone.")


if __name__ == "__main__":
    main()
