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

    # 그 외 첫 번째 유효 범위 정수
    for token in re.findall(r"\d+", text):
        n = int(token)
        if 0 <= n < num_options:
            return n
    return None
