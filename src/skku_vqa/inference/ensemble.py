"""LLM-합성 앙상블 추론: K개 프롬프트 변이의 분석을 모아, 최종 답을 LLM이 직접 종합 생성.

규칙 #5: 최종 label은 LLM 생성 텍스트(합성 호출의 출력)에서 파싱한다. 코드가 K개
변이의 답을 다수결/카운트/룰로 집계해 label을 결정하지 않는다. 변이 출력은 오직
LLM에게 '근거'로 제공될 뿐이며, 최종 선택은 합성 LLM 호출이 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schema import Sample
from ..models.base import VLMModel
from ..prompting.builders import build_prompt, build_synthesis_prompt
from ..prompting.parsers import parse_label
from .predictor import build_retry_prompt


@dataclass
class EnsemblePrediction:
    sample_id: str
    label: int
    raw_output: str
    templates: list[str]
    variant_outputs: list[str]
    synthesis_output: str
    retry_synthesis_output: str | None
    parse_ok: bool
    retried: bool = False


def ensemble_predict(
    model: VLMModel, sample: Sample, templates: list[str]
) -> EnsemblePrediction:
    n = len(sample.options) if sample.options else 1

    # 1) K개 변이 분석 수집 (각자 독립 호출). 코드는 이 답들을 집계하지 않는다.
    variant_outputs: list[str] = []
    for template in templates:
        prompt = build_prompt(sample, template)
        variant_outputs.append(model.predict(sample.image_path, prompt))

    # 2) 합성 호출: LLM이 변이 근거를 종합해 최종 ANSWER: N 생성 (규칙 #5).
    synthesis_prompt = build_synthesis_prompt(sample, variant_outputs)
    synthesis_output = model.predict(sample.image_path, synthesis_prompt)
    raw = synthesis_output
    label = parse_label(synthesis_output, n)
    retried = False
    retry_synthesis: str | None = None

    if label is None:
        # 규칙 #5: 룰 override 없이, 합성 프롬프트를 포함해 LLM에게 다시 생성하게 한다.
        retried = True
        retry_raw = model.predict(sample.image_path, build_retry_prompt(synthesis_prompt, n))
        retry_synthesis = retry_raw
        retry_label = parse_label(retry_raw, n)
        if retry_label is not None:
            label = retry_label
            raw = retry_raw

    parse_ok = label is not None
    return EnsemblePrediction(
        sample_id=sample.id,
        label=label if label is not None else 0,
        raw_output=raw,
        templates=list(templates),
        variant_outputs=variant_outputs,
        synthesis_output=synthesis_output,
        retry_synthesis_output=retry_synthesis,
        parse_ok=parse_ok,
        retried=retried,
    )
