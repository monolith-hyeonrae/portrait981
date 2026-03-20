"""Video context analysis with Qwen2.5-VL.

압축 비디오에서 날씨, 에너지 흐름, 탑승자 정보, 장면 맥락을 추출.

Usage:
    python scripts/test_video_context.py ~/Videos/reaction_test/cap_1.mp4
    python scripts/test_video_context.py data/datasets/portrait-v1/videos/test_2.mp4
    python scripts/test_video_context.py VIDEO --nframes 32 --prompt "custom question"
"""

from __future__ import annotations

import argparse
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("video_context")

CONTEXT_PROMPT_KO = """\
이 영상은 놀이공원 야외 어트랙션 탑승 중 전방 카메라로 촬영된 영상입니다.
다음을 분석하여 JSON으로 응답해주세요:

{
  "weather": "sunny | cloudy | overcast | rainy | sunset 중 택1",
  "sky_condition": "하늘 상태 상세 묘사 (맑은 파란하늘, 얇은 구름, 두꺼운 흐림, 노을빛 등)",
  "emotional_tone": "탑승 전체의 감정 톤 한줄 총평 (예: 처음부터 끝까지 신나고 즐거움, 초반 긴장 후 폭발적 흥분, 차분하고 여유로움)"
}

JSON만 응답하세요."""

CONTEXT_PROMPT_EN = """\
This video is recorded by a forward-facing camera during an outdoor amusement park ride.
Analyze the following and respond in JSON only:

{
  "weather": "sunny | cloudy | overcast | rainy | sunset",
  "sky_condition": "detailed sky description (e.g. clear blue sky, thin clouds, thick overcast, golden sunset glow)",
  "emotional_tone": "overall emotional tone of the ride experience in one sentence (e.g. joyful and energetic throughout, nervous at first then thrilled, calm and relaxed)"
}

Respond with JSON only. No explanation."""

CONTEXT_PROMPT = CONTEXT_PROMPT_EN


def load_model(model_name: str):
    """Load Qwen2.5-VL with 4-bit quantization."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def analyze_video(model, processor, video_path: str, prompt: str, nframes: int = 24):
    """Run a single prompt against the video."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 480 * 640, "nframes": nframes},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Video Context Analysis with Qwen2.5-VL")
    parser.add_argument("video", help="mp4 video path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="model name (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--nframes", type=int, default=24, help="number of frames to sample")
    parser.add_argument("--prompt", "-p", default=None, help="custom prompt (overrides default)")
    parser.add_argument("--lang", choices=["en", "ko"], default="en", help="prompt language (default: en)")
    args = parser.parse_args()

    logger.info("Loading model: %s", args.model)
    t0 = time.time()
    model, processor = load_model(args.model)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # 1. Context analysis
    prompt = args.prompt or (CONTEXT_PROMPT_KO if args.lang == "ko" else CONTEXT_PROMPT_EN)
    logger.info("Analyzing video: %s (%d frames)", args.video, args.nframes)
    t1 = time.time()
    response = analyze_video(model, processor, args.video, prompt, args.nframes)
    logger.info("Analysis done in %.1fs", time.time() - t1)

    print("\n" + "=" * 60)
    print(f"Video: {args.video}")
    print(f"Frames: {args.nframes}")
    print("=" * 60)
    print(f"\n{response}")
    print(f"\n{'=' * 60}")

    # Try to parse JSON
    try:
        # Extract JSON from response (may have markdown code fences)
        json_str = response
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        parsed = json.loads(json_str.strip())
        print("\nParsed JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, IndexError):
        print("\n(Could not parse as JSON — raw response above)")



if __name__ == "__main__":
    main()
