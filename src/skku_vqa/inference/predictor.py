"""추론 파이프라인: Sample → 프롬프트 → LLM → label(번호).

규칙 #5: 최종 label은 LLM 생성 텍스트에서 파싱한 번호다. 룰 override 없음.
파싱 실패 시 폴백은 부정확하므로, 재질의(retry)를 우선하고 그래도 실패하면
첫 번째 선택지(label 0)로 둔다(빈 제출 방지용 최소 폴백).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schema import Sample
from ..models.base import VLMModel
from ..prompting.builders import build_prompt
from ..prompting.parsers import parse_label


def build_retry_prompt(base_prompt: str, num_options: int) -> str:
    """파싱 실패 시 원래 과제(context/question/options)를 그대로 다시 주고 엄격한
    ANSWER: N 형식만 요청한다. stateless 모델도 풀 수 있어야 하므로 프롬프트 전체를
    재전송한다. 답은 여전히 LLM이 생성한다(규칙 #5)."""
    last = num_options - 1
    return (
        f"{base_prompt}\n\n"
        "Your previous response could not be parsed. Reply with only one line in the "
        f"exact form: ANSWER: <n>\n"
        f"where <n> is a single 0-based option number from 0 to {last}. "
        "Do not explain. Do not add any other text."
    )


@dataclass
class Prediction:
    sample_id: str
    label: int
    raw_output: str
    initial_raw_output: str
    retry_raw_output: str | None
    template: str
    parse_ok: bool
    retried: bool = False


def predict_sample(model: VLMModel, sample: Sample, template: str) -> Prediction:
    prompt = build_prompt(sample, template)
    initial_raw = model.predict(sample.image_path, prompt)
    n = len(sample.options) if sample.options else 1
    label = parse_label(initial_raw, n)
    raw = initial_raw
    retried = False
    retry_raw: str | None = None

    if label is None:
        # 규칙 #5: 룰 override 없이, 원래 과제를 포함해 LLM에게 다시 답을 생성하게 한다.
        retried = True
        retry_raw = model.predict(sample.image_path, build_retry_prompt(prompt, n))
        retry_label = parse_label(retry_raw, n)
        if retry_label is not None:
            label = retry_label
            raw = retry_raw

    parse_ok = label is not None
    return Prediction(
        sample_id=sample.id,
        label=label if label is not None else 0,
        raw_output=raw,
        initial_raw_output=initial_raw,
        retry_raw_output=retry_raw,
        template=template,
        parse_ok=parse_ok,
        retried=retried,
    )
