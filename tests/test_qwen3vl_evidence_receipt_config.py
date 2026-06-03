from pathlib import Path

from skku_vqa.config import load_config


def test_qwen3vl_evidence_receipt_config() -> None:
    cfg = load_config(Path("configs/experiment/qwen3vl_8b_evidence_receipt.yaml"))

    assert cfg.experiment_name == "qwen3vl_8b_evidence_receipt"
    assert cfg.model.name == "qwen3_vl"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.model.load_in_4bit is True
    assert cfg.model.max_pixels == 200704
    assert cfg.prompt.template == "evidence_receipt"
