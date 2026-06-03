"""TDD: ensemble_predict runs K variant calls + 1 final LLM synthesis call.

규칙 #5: the final label MUST come from the synthesis LLM call, NOT from a code
vote/count of the K variant answers.
"""

from __future__ import annotations

from skku_vqa.data.schema import Sample
from skku_vqa.inference.ensemble import EnsemblePrediction, ensemble_predict
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


def test_ensemble_makes_k_plus_one_calls_and_uses_final_synthesis() -> None:
    templates = ["bias_guarded_v2", "evidence_receipt", "evidence_then_answer"]
    # All 3 variants say 0, but the SYNTHESIS says 1.
    # If code voted, label would be 0. It must be 1 (synthesis decides) -> proves 규칙 #5.
    model = FakeModel(["ANSWER: 0", "ANSWER: 0", "ANSWER: 0", "ANSWER: 1"])
    pred = ensemble_predict(model, _sample(), templates)
    assert isinstance(pred, EnsemblePrediction)
    assert pred.label == 1  # from synthesis, NOT a majority vote of the variants
    assert pred.parse_ok is True
    assert len(model.calls) == 4  # 3 variants + 1 synthesis
    assert len(pred.variant_outputs) == 3
    assert pred.variant_outputs == ["ANSWER: 0", "ANSWER: 0", "ANSWER: 0"]
    assert pred.synthesis_output == "ANSWER: 1"
    assert pred.templates == templates


def test_ensemble_retries_synthesis_parse_failure() -> None:
    templates = ["bias_guarded_v2", "evidence_receipt"]
    # 2 variants, then a synthesis that fails to parse, then a retry that parses.
    model = FakeModel(["ANSWER: 1", "ANSWER: 2", "I am unsure", "ANSWER: 2"])
    pred = ensemble_predict(model, _sample(), templates)
    assert pred.label == 2
    assert pred.parse_ok is True
    assert pred.retried is True
    assert len(model.calls) == 4  # 2 variants + synthesis + 1 retry
    # audit: first synthesis and retry synthesis both preserved
    assert pred.synthesis_output == "I am unsure"
    assert pred.retry_synthesis_output == "ANSWER: 2"
    assert pred.raw_output == "ANSWER: 2"


def test_ensemble_fallback_is_marked_parse_fail_not_submittable() -> None:
    # When synthesis + retry are both unparseable, the in-memory label is the
    # last-resort 0 BUT parse_ok is False so make_submission refuses it (규칙 #5).
    templates = ["bias_guarded_v2", "evidence_receipt"]
    model = FakeModel(["ANSWER: 1", "ANSWER: 2", "no number", "still none"])
    pred = ensemble_predict(model, _sample(), templates)
    assert pred.label == 0
    assert pred.parse_ok is False
    assert pred.retried is True
    assert pred.retry_synthesis_output == "still none"
