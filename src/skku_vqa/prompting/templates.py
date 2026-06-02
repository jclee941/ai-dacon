"""프롬프트 템플릿. 이 대회에서 프롬프트는 솔루션의 핵심 로직이다.

규칙 #5에 따라 최종 답은 LLM이 근거를 종합해 직접 생성해야 한다. 템플릿은
LLM이 (1) 보이는 근거만 사용하고 (2) 근거 부족 시 unknown 선택지를 고르도록
유도하지만, 답 자체는 항상 LLM이 산출한다.

베이스라인 입력 필드: context, question, options(번호). 출력: 선택지 번호(label).
"""

from __future__ import annotations

EVIDENCE_GUARD = (
    "You must answer based ONLY on the given context, the visible evidence in the "
    "image, and the explicit text in the question. Do NOT infer gender, race, "
    "ethnicity, occupation, personality, social role, intent, or ability from "
    "appearance or social stereotypes."
)

# 직접 답변 프롬프트 (번호 출력)
DIRECT = (
    "{evidence_guard}\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Options:\n{options_block}\n\n"
    "Select exactly one option. If the context, image, and question do not provide "
    "enough evidence to determine the answer, choose the option meaning unknown / "
    "cannot be determined.\n"
    "Answer with only the option number (0-based)."
)

# 편향 차단 강조 프롬프트 (번호 출력)
BIAS_GUARDED = (
    "{evidence_guard}\n"
    "If choosing a specific person would require assuming a protected or social "
    "attribute that is not explicitly stated in the context or shown in the image, "
    "you must choose unknown / cannot be determined instead.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Options:\n{options_block}\n\n"
    "Answer with only the option number (0-based)."
)

# 근거 종합형 프롬프트 (규칙 #5 부합: 근거 검토 후 LLM이 최종 번호 생성)
EVIDENCE_THEN_ANSWER = (
    "{evidence_guard}\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Options:\n{options_block}\n\n"
    "Think step by step about what evidence is actually present. Then decide whether "
    "the answer is determined by that evidence. If it is not determined, choose the "
    "unknown / cannot be determined option.\n"
    "End your response with a line exactly in the form: ANSWER: <option number 0-based>"
)

TEMPLATES: dict[str, str] = {
    "direct": DIRECT,
    "bias_guarded": BIAS_GUARDED,
    "evidence_then_answer": EVIDENCE_THEN_ANSWER,
}
