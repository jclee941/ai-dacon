"""TDD: run_inference must route to ensemble_predict when prompt.ensemble_templates is set."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from skku_vqa.config import Config
from skku_vqa.data.schema import Sample
from skku_vqa.models.base import VLMModel

REPO = Path(__file__).resolve().parents[1]


def _load_run_inference():
    spec = importlib.util.spec_from_file_location(
        "run_inference", REPO / "scripts" / "run_inference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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
        question="Who?",
        options=["A", "B", "C"],
        context="ctx",
        image_path=None,
    )


def test_predict_row_single_template_mode() -> None:
    mod = _load_run_inference()
    cfg = Config()
    cfg.prompt.template = "bias_guarded"
    cfg.prompt.ensemble_templates = None
    model = FakeModel(["ANSWER: 1"])
    row = mod.predict_row(model, _sample(), cfg)
    assert row["label"] == 1
    assert row["mode"] == "single"
    assert len(model.calls) == 1


def test_predict_row_ensemble_mode_uses_synthesis() -> None:
    mod = _load_run_inference()
    cfg = Config()
    cfg.prompt.ensemble_templates = ["bias_guarded_v2", "evidence_receipt"]
    # 2 variants say 0, synthesis says 2 -> ensemble label must be 2 (규칙 #5)
    model = FakeModel(["ANSWER: 0", "ANSWER: 0", "ANSWER: 2"])
    row = mod.predict_row(model, _sample(), cfg)
    assert row["label"] == 2
    assert row["mode"] == "ensemble"
    assert len(model.calls) == 3  # 2 variants + 1 synthesis
    assert row["variant_outputs"] == ["ANSWER: 0", "ANSWER: 0"]
    assert row["synthesis_output"] == "ANSWER: 2"


def test_predict_row_ensemble_persists_retry_synthesis() -> None:
    mod = _load_run_inference()
    cfg = Config()
    cfg.prompt.ensemble_templates = ["bias_guarded_v2", "evidence_receipt"]
    # 2 variants, synthesis unparseable, retry parses -> row must persist both
    model = FakeModel(["ANSWER: 1", "ANSWER: 2", "unsure", "ANSWER: 2"])
    row = mod.predict_row(model, _sample(), cfg)
    assert row["label"] == 2
    assert row["mode"] == "ensemble"
    assert row["parse_ok"] is True
    assert row["retried"] is True
    assert row["synthesis_output"] == "unsure"
    assert row["retry_synthesis_output"] == "ANSWER: 2"
