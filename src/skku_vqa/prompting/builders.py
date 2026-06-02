"""샘플 → 프롬프트 문자열. 선택지는 0-based 번호로 표기(제출 label과 일치)."""

from __future__ import annotations

from ..data.schema import Sample
from .templates import EVIDENCE_GUARD, TEMPLATES


def build_options_block(sample: Sample) -> str:
    lines = []
    for i, opt in enumerate(sample.options):  # 0-based
        lines.append(f"{i}. {opt}")
    return "\n".join(lines)


def build_prompt(sample: Sample, template: str) -> str:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template: {template}")
    return TEMPLATES[template].format(
        evidence_guard=EVIDENCE_GUARD,
        context=sample.context or "",
        question=sample.question,
        options_block=build_options_block(sample),
    )
