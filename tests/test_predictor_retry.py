"""TDD: predict_sample must retry parse failures by re-asking the LLM (규칙 #5).

The retry must re-ask the FULL task (context/question/options), not a bare
"pick a number", so a stateless model can still solve the sample. Initial and
retry raw outputs must both be preserved for audit.
"""

from __future__ import annotations

from skku_vqa.data.schema import Sample
from skku_vqa.inference.predictor import predict_sample
from skku_vqa.models.base import VLMModel


class FakeModel(VLMModel):
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls: list[str] = []

    def predict(self, image_path: str | None, prompt: str) -> str:
        self.calls.append(prompt)
        idx = min(len(self.calls) - 1, len(self._outputs) - 1)
        return self._outputs[idx]


def _sample() -> Sample:
    return Sample(
        id="s1",
        question="Who arrived late?",
        options=["The student", "The teacher", "Cannot be determined"],
        context="ctx",
        image_path=None,
    )


def test_predict_sample_does_not_retry_successful_parse() -> None:
    model = FakeModel(["ANSWER: 1"])
    pred = predict_sample(model, _sample(), "bias_guarded")
    assert pred.label == 1
    assert pred.parse_ok is True
    assert pred.retried is False
    assert pred.retry_raw_output is None
    assert pred.initial_raw_output == "ANSWER: 1"
    assert len(model.calls) == 1


def test_predict_sample_retries_parse_failure_with_full_task_and_strict_form() -> None:
    model = FakeModel(["I cannot pick a number.", "ANSWER: 2"])
    pred = predict_sample(model, _sample(), "bias_guarded")
    assert pred.label == 2
    assert pred.parse_ok is True
    assert pred.retried is True
    assert len(model.calls) == 2
    retry_prompt = model.calls[1]
    low = retry_prompt.lower()
    # strict format requested
    assert "answer:" in low
    assert "only" in low
    # retry re-asks the FULL task so a stateless model can solve it
    assert "Who arrived late?" in retry_prompt
    assert "The teacher" in retry_prompt
    # audit: both raw outputs preserved
    assert pred.initial_raw_output == "I cannot pick a number."
    assert pred.retry_raw_output == "ANSWER: 2"


def test_predict_sample_falls_back_to_zero_after_retry_failure() -> None:
    model = FakeModel(["no number here", "still no number"])
    pred = predict_sample(model, _sample(), "bias_guarded")
    assert pred.label == 0
    assert pred.parse_ok is False
    assert pred.retried is True
    assert len(model.calls) == 2
    assert pred.initial_raw_output == "no number here"
    assert pred.retry_raw_output == "still no number"
