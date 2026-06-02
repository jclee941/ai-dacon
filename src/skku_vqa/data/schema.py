"""데이터 스키마 정의."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sample:
    """단일 VQA 샘플."""

    id: str
    question: str
    options: list[str]
    # BBQ 스타일 지문 (모호/명확 맥락). 베이스라인 입력 필드.
    context: str | None = None
    image_path: str | None = None
    # 정답 인덱스 (학습/캘리브레이션 전용; 테스트셋에는 없음)
    answer_idx: int | None = None
    # 평가 그룹 (ambiguous/disambiguated; 테스트셋에는 비공개)
    group: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def option_letters(self) -> list[str]:
        return [chr(ord("A") + i) for i in range(len(self.options))]
