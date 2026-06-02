"""LLaVA-OneVision 래퍼 (transformers 백엔드, vLLM fallback).

sm_120(Blackwell)에서 vLLM가 동작하지 않을 때의 폴백. bnb 4bit 지원.
무거운 import는 lazy 처리.

의존성: transformers>=4.45, accelerate, pillow, (bitsandbytes for 4bit)
"""

from __future__ import annotations

from .base import VLMModel


class LlavaOneVisionTransformers(VLMModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            LlavaOnevisionForConditionalGeneration,
        )

        self.max_new_tokens = max_new_tokens
        torch_dtype = getattr(torch, dtype)

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image_path: str | None, prompt: str) -> str:
        content: list[dict] = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        images = None
        if image_path:
            from PIL import Image

            images = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0].strip()
