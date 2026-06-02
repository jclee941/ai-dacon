"""설정 기반 모델 로더. 로컬 가중치 로드만 지원(규칙 #4: 외부 API 금지).

llava + backend=vllm   -> LlavaOneVisionVLLM (기본)
llava + backend=transformers -> LlavaOneVisionTransformers (sm_120 fallback, 4bit)
qwen  -> QwenVLModel (transformers)
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import VLMModel


def load_model(cfg: ModelConfig) -> VLMModel:
    name = cfg.name.lower()
    backend = (getattr(cfg, "backend", "vllm") or "vllm").lower()

    if name.startswith("llava"):
        if backend == "transformers":
            from .llava_ov_transformers import LlavaOneVisionTransformers

            return LlavaOneVisionTransformers(
                model_id=cfg.model_id,
                dtype=cfg.dtype,
                device_map=cfg.device_map,
                max_new_tokens=cfg.max_new_tokens,
                load_in_4bit=getattr(cfg, "load_in_4bit", True),
            )
        if backend == "vllm":
            from .llava_ov import LlavaOneVisionVLLM

            return LlavaOneVisionVLLM(
                model_id=cfg.model_id,
                dtype=cfg.dtype,
                max_new_tokens=cfg.max_new_tokens,
            )
        raise ValueError(f"unsupported backend for llava: {backend}")

    if name.startswith("qwen3_vl"):
        from .qwen3_vl import Qwen3VLModel

        return Qwen3VLModel(
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
            max_new_tokens=cfg.max_new_tokens,
            load_in_4bit=getattr(cfg, "load_in_4bit", True),
            max_pixels=cfg.max_pixels,
        )

    if name.startswith("qwen"):
        from .qwen_vl import QwenVLModel

        return QwenVLModel(
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
            max_new_tokens=cfg.max_new_tokens,
            load_in_4bit=getattr(cfg, "load_in_4bit", True),
            max_pixels=cfg.max_pixels,
        )

    raise ValueError(f"unsupported model: {cfg.name}")
