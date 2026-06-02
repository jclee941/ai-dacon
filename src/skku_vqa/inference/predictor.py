"""추론 파이프라인: Sample → 프롬프트 → LLM → label(번호).

규칙 #5: 최종 label은 LLM 생성 텍스트에서 파싱한 번호다. 룰 override 없음.
파싱 실패 시 폴백은 부정확하므로, 재질의(retry)를 우선하고 그래도 실패하면
첫 번째 선택지(1)로 둔다(빈 제출 방지용 최소 폴백).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schema import Sample
from ..models.base import VLMModel
from ..prompting.builders import build_prompt
from ..prompting.parsers import parse_label


@dataclass
class Prediction:
    sample_id: str
    label: int
    raw_output: str
    template: str
    parse_ok: bool


def predict_sample(model: VLMModel, sample: Sample, template: str) -> Prediction:
    prompt = build_prompt(sample, template)
    raw = model.predict(sample.image_path, prompt)
    n = len(sample.options) if sample.options else 1
    label = parse_label(raw, n)
    parse_ok = label is not None
    return Prediction(
        sample_id=sample.id,
        label=label if label is not None else 0,
        raw_output=raw,
        template=template,
        parse_ok=parse_ok,
    )
