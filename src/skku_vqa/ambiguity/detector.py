"""근거 충분성 진단 (분석/캘리브레이션 전용).

대회 규칙 #5: 최종 답변은 LLM이 후보/근거/검토를 종합해 직접 생성해야 한다.
단순 룰/다수결/조건문/사전정의 목록 기반의 최종 답 결정은 금지된다.

따라서 이 모듈은 *최종 답을 결정하지 않는다*. unknown 선택지 위치 파악과
로컬 진단(LLM이 unknown을 골랐는지 분석)에만 사용한다. 실제 정답 생성은
prompting + LLM 추론(inference/predictor)에서 LLM이 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .unknown_mapper import find_unknown_option


@dataclass
class AmbiguityDiagnosis:
    """로컬 분석용 진단 결과 (제출 답 결정에는 사용하지 않음)."""

    unknown_option_idx: int | None
    llm_chose_unknown: bool


def diagnose(options: list[str], llm_final_idx: int) -> AmbiguityDiagnosis:
    """LLM이 생성한 최종 선택을 사후 분석한다.

    Balanced Accuracy 디버깅 시 'LLM이 ambiguous 문항에서 unknown을 골랐는가'를
    파악하기 위한 용도. 답을 바꾸지 않는다.
    """
    unknown_idx = find_unknown_option(options)
    return AmbiguityDiagnosis(
        unknown_option_idx=unknown_idx,
        llm_chose_unknown=(unknown_idx is not None and llm_final_idx == unknown_idx),
    )
