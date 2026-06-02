"""Qwen2.5-VL 래퍼. RTX 5070 Ti(16GB) 대응을 위해 4-bit 양자화 기본 지원.

의존성: transformers>=4.49, qwen-vl-utils, accelerate, bitsandbytes
무거운 import는 lazy로 처리해 CPU 환경/테스트에서도 모듈 로드가 가능하도록 한다.
"""

from __future__ import annotations

from .base import VLMModel


class QwenVLModel(VLMModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        load_in_4bit: bool = True,
        max_pixels: int | None = None,
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        content: list[dict] = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": prompt})
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
