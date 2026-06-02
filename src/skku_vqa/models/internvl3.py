"""InternVL3 래퍼 (transformers 4.57 네이티브, InternVLForConditionalGeneration).

trust_remote_code 불필요. 16GB GPU 대응을 위해 4-bit 양자화 기본 지원.
InternVL은 Qwen3-VL과 달리 qwen_vl_utils가 필요 없고, 이미지를 processor에 직접 넘긴다.
"""

from __future__ import annotations

from .base import VLMModel


class InternVL3Model(VLMModel):
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3-8B-hf",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        load_in_4bit: bool = True,
        max_pixels: int | None = None,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
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

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image_path: str | None, prompt: str) -> str:
        content: list[dict] = []
        if image_path:
            content.append({"type": "image", "url": image_path})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0].strip()
