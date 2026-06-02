"""Shared constants for the competition pipeline."""

# 표준화된 답이 "알 수 없음/판단 불가"를 의미하는 선택지 텍스트 패턴 (한/영).
UNKNOWN_OPTION_PATTERNS: tuple[str, ...] = (
    "cannot be determined",
    "cannot determine",
    "not enough information",
    "not enough info",
    "cannot answer",
    "unknown",
    "unclear",
    "undetermined",
    "no information",
    "both",
    "neither",
    "알 수 없",
    "판단할 수 없",
    "판단 불가",
    "결정할 수 없",
    "정보가 부족",
    "알 수 없음",
)

# 평가 그룹 라벨 (테스트셋에는 공개되지 않음; 로컬 캘리브레이션 전용).
GROUP_AMBIGUOUS = "ambiguous"
GROUP_DISAMBIGUATED = "disambiguated"
