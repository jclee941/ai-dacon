"""Qwen3-VL 래퍼. RTX 5070 Ti(16GB) 대응을 위해 4-bit 양자화 기본 지원.

의존성: transformers>=4.57, qwen-vl-utils, accelerate, bitsandbytes
무거운 import는 lazy로 처리해 CPU 환경/테스트에서도 모듈 로드가 가능하도록 한다.
"""

from __future__ import annotations

from .base import VLMModel


class Qwen3VLModel(VLMModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        load_in_4bit: bool = True,
        max_pixels: int | None = None,
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        torch_dtype = getattr(torch, dtype)

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant_config,
        )
        processor_kwargs = {}
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    def predict(self, image_path: str | None, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        content = build_message_content(image_path, prompt, self.max_pixels)
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0].strip()


def build_message_content(
    image_path: str | None, prompt: str, max_pixels: int | None = None
) -> list[dict]:
    """Qwen3-VL 메시지 content 구성.

    qwen_vl_utils.process_vision_info가 message item의 max_pixels를 우선 적용하므로,
    16GB GPU에서 vision tower OOM을 막으려면 image item에 직접 한도를 넣어야 한다.
    """
    content: list[dict] = []
    if image_path:
        item: dict = {"type": "image", "image": image_path}
        if max_pixels is not None:
            item["min_pixels"] = 28 * 28 * 4
            item["max_pixels"] = max_pixels
        content.append(item)
    content.append({"type": "text", "text": prompt})
    return content
