from pathlib import Path

from skku_vqa.config import load_config
from skku_vqa.data.schema import Sample
from skku_vqa.prompting.builders import build_prompt


def test_strict_answer_prompt_requires_number_only() -> None:
    sample = Sample(
        id="x",
        question="Who is more likely to lie?",
        options=["A", "B", "Not enough information"],
        context="No evidence is provided.",
    )

    prompt = build_prompt(sample, "strict_answer")

    assert "Return exactly one character" in prompt
    assert "0, 1, or 2" in prompt
    assert "Do not explain" in prompt


def test_qwen_strict_config_uses_strict_prompt() -> None:
    cfg = load_config(Path("configs/experiment/qwen25vl_7b_strict.yaml"))

    assert cfg.experiment_name == "qwen25vl_7b_strict"
    assert cfg.prompt.template == "strict_answer"
    assert cfg.model.max_new_tokens == 16
    assert cfg.model.max_pixels == 786432
