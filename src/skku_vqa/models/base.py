"""모든 VLM 래퍼의 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VLMModel(ABC):
    """이미지 + 프롬프트 → 텍스트 출력."""

    @abstractmethod
    def predict(self, image_path: str | None, prompt: str) -> str:
        """단일 샘플 추론. image_path가 None이면 텍스트 전용."""
        raise NotImplementedError
