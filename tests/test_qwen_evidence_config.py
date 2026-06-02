from pathlib import Path

from skku_vqa.config import load_config


def test_qwen_evidence_config_uses_evidence_prompt() -> None:
    cfg = load_config(Path("configs/experiment/qwen25vl_7b_evidence.yaml"))

    assert cfg.experiment_name == "qwen25vl_7b_evidence"
    assert cfg.prompt.template == "evidence_then_answer"
    assert cfg.model.max_pixels == 786432
