"""선택지 중 '알 수 없음/판단 불가'에 해당하는 항목 탐지.

대회 핵심 로직: unknown 선택지가 보기마다 다른 문구/위치로 등장하므로
프롬프트가 아닌 코드로 신뢰성 있게 매핑한다.
"""

from __future__ import annotations

from ..constants import UNKNOWN_OPTION_PATTERNS


def find_unknown_option(options: list[str]) -> int | None:
    """unknown 의미 선택지의 인덱스를 반환. 없으면 None."""
    for idx, opt in enumerate(options):
        text = opt.strip().lower()
        if any(pat in text for pat in UNKNOWN_OPTION_PATTERNS):
            return idx
    return None


def has_unknown_option(options: list[str]) -> bool:
    return find_unknown_option(options) is not None
