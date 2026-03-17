"""Test Video-LLM temporal grounding on portrait videos.

Qwen2.5-VL-3B로 텍스트 프롬프트 기반 temporal grounding 테스트.

Usage:
    python scripts/test_video_llm.py ~/Videos/reaction_test/test_2.mp4 \
        --prompt "a moment where the person has a warm genuine smile"
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("video_llm")


def main():
    parser = argparse.ArgumentParser(description="Video-LLM Temporal Grounding Test")
    parser.add_argument("video", help="mp4 video path")
    parser.add_argument("--prompt", "-p", default="Find the moment where the person has a warm genuine smile looking at the camera")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    logger.info("Loading model: %s", args.model)
    t0 = time.time()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    # 4-bit quantization for 8GB GPU
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Build message with video
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video, "max_pixels": 360 * 420, "nframes": 16},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Process
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    logger.info("Input prepared, generating...")
    t1 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    # Decode only the generated part
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    logger.info("Generated in %.1fs", time.time() - t1)
    print("\n" + "=" * 60)
    print(f"Video: {args.video}")
    print(f"Prompt: {args.prompt}")
    print(f"=" * 60)
    print(f"\nResponse:\n{response}")
    print(f"\n{'=' * 60}")

    # Try additional prompts
    additional_prompts = [
        "At what timestamp does the person smile the brightest?",
        "Describe the person's facial expressions throughout the video.",
        "Which moments in this video would make the best portrait photo?",
    ]

    for prompt in additional_prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": args.video, "max_pixels": 360 * 420, "nframes": 16},
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
            output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
