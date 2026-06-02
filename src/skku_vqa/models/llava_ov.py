"""LLaVA-OneVision 래퍼 (vLLM 백엔드).

대회 베이스라인과 동일 계열. 오프라인 로컬 가중치 로드만 사용(규칙 #4).
무거운 import는 lazy 처리하여 CPU/테스트 환경에서도 모듈 로드가 가능하도록 한다.

의존성: vllm, transformers, pillow
"""

from __future__ import annotations

from .base import VLMModel


class LlavaOneVisionVLLM(VLMModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.max_new_tokens = max_new_tokens
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    @staticmethod
    def _format(prompt: str, has_image: bool) -> str:
        # LLaVA-OneVision chat 템플릿 (qwen2 기반). 이미지 토큰 포함.
        img = "<image>\n" if has_image else ""
        return (
            "<|im_start|>user\n"
            f"{img}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def predict(self, image_path: str | None, prompt: str) -> str:
        text = self._format(prompt, has_image=image_path is not None)
        if image_path:
            from PIL import Image

            request = {
                "prompt": text,
                "multi_modal_data": {"image": Image.open(image_path).convert("RGB")},
            }
        else:
            request = {"prompt": text}

        out = self.llm.generate([request], self.sampling)
        return out[0].outputs[0].text.strip()
