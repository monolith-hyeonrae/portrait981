"""Video-LLM temporal grounding with visual HTML report.

비디오를 Video-LLM에 입력하고, 응답 + 프레임 타임라인을 HTML로 시각화.

Usage:
    python scripts/test_video_llm_visual.py ~/Videos/reaction_test/test_2.mp4 --output /tmp/llm_test.html
"""

from __future__ import annotations

import argparse
import base64
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("video_llm_visual")


def extract_frames(video_path, n_frames=32):
    """비디오에서 균등 간격 프레임 추출."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            ts = idx / fps if fps > 0 else 0
            frames.append({"index": int(idx), "timestamp": ts, "image": frame})
    cap.release()
    return frames, duration, fps


def frame_to_b64(frame_bgr, max_width=320):
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


N_INPUT_FRAMES = 16  # frames sampled by the model


def calibrate_timestamps(response: str, video_duration: float) -> str:
    """Calibrate model's timestamps to actual video timeline.

    The model sees N_INPUT_FRAMES uniformly sampled frames but doesn't know
    the actual video duration. It outputs timestamps in its own time scale
    (roughly 0 to N_INPUT_FRAMES/fps_assumed). We detect numeric timestamps
    and remap them to actual video time.
    """
    import re

    # Find all float/int numbers that look like timestamps (0.0 - 999.9)
    pattern = r'(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*seconds?'
    matches = list(re.finditer(pattern, response, re.IGNORECASE))
    if not matches:
        return response

    # Estimate model's time scale: max timestamp in response
    all_ts = []
    for m in matches:
        all_ts.append(float(m.group(1)))
        if m.group(2):
            all_ts.append(float(m.group(2)))

    model_max = max(all_ts) if all_ts else 1.0
    # The model likely thinks the video is about model_max seconds long
    # But actual duration is video_duration
    # Simple linear rescale
    if model_max < 1.0:
        return response

    scale = video_duration / max(model_max * 1.2, video_duration * 0.3)  # heuristic

    calibrated = response
    # Replace from end to preserve positions
    for m in reversed(matches):
        t1 = float(m.group(1))
        t1_cal = min(t1 * scale, video_duration)
        if m.group(2):
            t2 = float(m.group(2))
            t2_cal = min(t2 * scale, video_duration)
            replacement = f"{t1_cal:.1f} - {t2_cal:.1f} seconds"
        else:
            replacement = f"{t1_cal:.1f} seconds"
        calibrated = calibrated[:m.start()] + replacement + calibrated[m.end():]

    return calibrated


def query_llm(model, processor, video_path, prompt):
    """Single query to Video-LLM."""
    from qwen_vl_utils import process_vision_info
    import torch

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "nframes": N_INPUT_FRAMES},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="mp4 path")
    parser.add_argument("--output", "-o", default="/tmp/llm_visual.html")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    video_path = args.video
    video_name = Path(video_path).name

    # 1. Extract timeline frames
    logger.info("Extracting frames...")
    frames, duration, fps = extract_frames(video_path, n_frames=32)
    logger.info("Extracted %d frames (%.1fs, %.1f fps)", len(frames), duration, fps)

    # 2. Load model
    logger.info("Loading model...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quant_config, device_map="auto", torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    logger.info("Model loaded")

    # 3. Run prompts
    prompts = [
        "Find the moment where the person has a warm genuine smile looking at the camera. Describe the timestamp.",
        "At what timestamp does the person smile the brightest? Answer with specific seconds.",
        "Describe the person's facial expressions throughout the video, with approximate timestamps.",
        "Which moments in this video would make the best portrait photo? Describe why and at what time.",
        "Is there a moment where the person looks cool or serious but still photogenic? When?",
    ]

    results = []
    for prompt in prompts:
        logger.info("Query: %s", prompt[:60])
        t0 = time.time()
        response = query_llm(model, processor, video_path, prompt)
        elapsed = time.time() - t0
        calibrated = calibrate_timestamps(response, duration)
        results.append({
            "prompt": prompt,
            "response": response,
            "calibrated": calibrated,
            "time": elapsed,
        })
        logger.info("Response (%.1fs): %s", elapsed, response[:100])
        if calibrated != response:
            logger.info("Calibrated: %s", calibrated[:100])

    # 4. Generate HTML
    html = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Video-LLM Test: {video_name}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
h1 {{ color: #e94560; }}
h2 {{ color: #4CAF50; margin-top: 30px; }}
.timeline {{ display: flex; overflow-x: auto; gap: 4px; padding: 10px 0; }}
.timeline img {{ height: 120px; border-radius: 4px; flex-shrink: 0; }}
.timeline .label {{ font-size: 10px; color: #888; text-align: center; }}
.qa {{ background: #16213e; border-radius: 8px; padding: 16px; margin: 12px 0; }}
.qa .prompt {{ color: #e94560; font-weight: bold; margin-bottom: 8px; }}
.qa .response {{ line-height: 1.6; white-space: pre-wrap; }}
.qa .meta {{ font-size: 11px; color: #555; margin-top: 8px; }}
.qa .calibrated {{ color: #4CAF50; margin-top: 8px; padding: 8px; background: #1a3320; border-radius: 4px; }}
.info {{ background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; }}
</style></head><body>
<h1>Video-LLM Temporal Grounding: {video_name}</h1>
<div class="info">
    Model: {args.model} (INT4) | Duration: {duration:.1f}s | FPS: {fps:.0f} | Frames sampled: 16
</div>
<h2>Timeline</h2>
<div class="timeline">"""]

    for f in frames:
        b64 = frame_to_b64(f["image"])
        html.append(f"""<div><img src="data:image/jpeg;base64,{b64}"><div class="label">{f['timestamp']:.1f}s</div></div>""")

    html.append("</div><h2>Q&A</h2>")

    for r in results:
        cal_html = ""
        if r["calibrated"] != r["response"]:
            cal_html = f"""<div class="calibrated"><b>Calibrated ({duration:.0f}s video):</b> {r['calibrated']}</div>"""
        html.append(f"""<div class="qa">
    <div class="prompt">{r['prompt']}</div>
    <div class="response">{r['response']}</div>
    {cal_html}
    <div class="meta">{r['time']:.1f}s</div>
</div>""")

    html.append("</body></html>")

    out = Path(args.output)
    out.write_text("\n".join(html), encoding="utf-8")
    logger.info("Report: %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
