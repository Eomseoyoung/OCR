import argparse
import torch
from transformers import AutoProcessor

# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration
# Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


def build_messages(image_path: str, mode: str):
    """
    mode:
      - "plain":  이미지 읽고 설명
      - "layout": 줄바꿈/레이아웃 최대한 유지 + 사용자에게 요약된 정보 제공
      - "json": JSON으로 (필드 추출용; 문서면 유용)
    """
    if mode == "plain":
        prompt = (
            "너는 OCR engine이야. 이미지를 읽고 설명해줘.\n"
            
        )
    elif mode == "layout":
        prompt = (
            "너는 OCR engine이야. 이미지에서 모든 텍스트를 정확히 추출해.\n"
            "줄바꿈, 공백, 읽기 순서를 가능한 한 유지해.\n"
            "사용자에게 요약된 정보를 제공해줘"
        )
    elif mode == "json":
        prompt = (
            "You are an OCR engine. Extract all text and return valid JSON.\n"
            'Schema: {"full_text": string, "lines": [string]}\n'
            "Return ONLY JSON."
        )
    else:
        raise ValueError("mode must be one of: plain, layout, json")

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_model(model_id: str):
    """
    model_id 예시:
      - Qwen/Qwen2-VL-7B-Instruct          (권장)
      - Qwen/Qwen2-VL-2B-Instruct          (가벼움)
      - Qwen/Qwen2.5-VL-7B-Instruct        (문서/인식 강화 계열)
      - Qwen/Qwen2.5-VL-32B-Instruct       (무거움)
    """
    if "Qwen2.5-VL" in model_id or "Qwen2_5" in model_id or "Qwen2.5" in model_id:
        cls = Qwen2_5_VLForConditionalGeneration
    else:
        cls = Qwen2VLForConditionalGeneration

    model = cls.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    return model


@torch.inference_mode()
def run_ocr(model_id: str, image_path: str, mode: str, max_new_tokens: int):
    model = load_model(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    messages = build_messages(image_path, mode)

    # chat template + vision preprocess
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # device_map="auto"일 때는 모델 디바이스로 옮기기만 해주면 됩니다.
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # 프롬프트 부분 제거 디코딩
    gen_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    return gen_text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", default="layout", choices=["plain", "layout", "json"])
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    out = run_ocr(args.model, args.image, args.mode, args.max_new_tokens)
    print(out)


if __name__ == "__main__":
    main()