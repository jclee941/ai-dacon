"""YAML 설정 로딩 및 검증 (Pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    image_dir: str = "data/raw/test/images"
    test_csv: str = "data/raw/test/test.csv"
    sample_submission: str = "data/raw/sample_submission.csv"
    output_dir: str = "outputs"
    artifact_dir: str = "artifacts"


class ModelConfig(BaseModel):
    name: str = "llava_ov"
    model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    backend: str = "vllm"  # vllm | transformers
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 64
    load_in_4bit: bool = False  # 16GB 개발서버에서 큰 모델을 transformers로 돌릴 때만
    max_pixels: int | None = None
    # CPU offload: VRAM이 부족해 4bit 모델이 GPU에 다 안 올라갈 때 사용.
    # cpu_offload=True면 bnb llm_int8_enable_fp32_cpu_offload를 켜고,
    # gpu_max_memory(예 "6GiB")로 GPU 할당을 제한해 나머지를 CPU로 보낸다.
    cpu_offload: bool = False
    gpu_max_memory: str | None = None
    cpu_max_memory: str | None = None


class PromptConfig(BaseModel):
    template: str = "bias_guarded"
    include_cot: bool = False
    output_format: str = "number_only"
    # LLM-합성 앵상블 모드: 이 템플릿들로 K개 분석 수집 후
    # 최종 답은 LLM이 종합 생성(규칙 #5). None이면 단일 템플릿 모드.
    ensemble_templates: list[str] | None = None


class InferenceConfig(BaseModel):
    batch_size: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    num_prompt_variants: int = 3
    cache_predictions: bool = True


class AmbiguityConfig(BaseModel):
    # 규칙 #5: 최종 답은 LLM이 생성. 아래는 프롬프트 전략 선택용이며 룰 override가 아니다.
    prompt_strategy: str = "bias_guarded"  # LLM이 근거 판단을 포함해 직접 답하도록 유도
    diagnose_unknown: bool = True  # 로컬 분석용 (답을 바꾸지 않음)


class SubmissionConfig(BaseModel):
    id_column: str = "sample_id"
    answer_column: str = "label"


class Config(BaseModel):
    seed: int = 42
    experiment_name: str = "default"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    ambiguity: AmbiguityConfig = Field(default_factory=AmbiguityConfig)
    submission: SubmissionConfig = Field(default_factory=SubmissionConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Config:
    """YAML 설정을 로드한다. `defaults:` 키로 베이스 설정을 상속할 수 있다."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    base_ref = raw.pop("defaults", None)
    if base_ref:
        base_raw = yaml.safe_load(Path(base_ref).read_text(encoding="utf-8")) or {}
        raw = _deep_merge(base_raw, raw)

    return Config(**raw)
