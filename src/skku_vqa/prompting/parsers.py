"""모델 출력 → 정답 인덱스(label) 파싱.

제출 label은 0-based 선택지 인덱스다 (sample_submission/train 기준).
프롬프트는 선택지를 0..N-1로 번호 매기므로, LLM이 그 번호를 출력한다.
"""

from __future__ import annotations

import re


def parse_label(raw: str, num_options: int) -> int | None:
    """모델 출력에서 0-based 선택지 인덱스를 추출. 범위 밖이면 None."""
    if not raw:
        return None
    text = raw.strip()

    # 'ANSWER: N' 우선
    m = re.search(r"ANSWER\s*[:\-]?\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 0 <= n < num_options:
            return n

    # 마커가 없으면: 끝에서부터 '단독 숫자'(앞뒤가 숫자가 아닌)를 우선 채택.
    # CoT/verbose 출력에서 reasoning 중 언급된 번호 대신 최종 답을 잡기 위함.
    standalone = re.findall(r"(?<!\d)(\d+)(?!\d)", text)
    for token in reversed(standalone):
        n = int(token)
        if 0 <= n < num_options:
            return n
    return None
