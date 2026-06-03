from pathlib import Path

from skku_vqa.config import load_config


def test_qwen3vl_ensemble_config() -> None:
    cfg = load_config(Path("configs/experiment/qwen3vl_8b_ensemble.yaml"))

    assert cfg.experiment_name == "qwen3vl_8b_ensemble"
    assert cfg.model.name == "qwen3_vl"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.model.load_in_4bit is True
    assert cfg.model.max_pixels == 200704
    assert cfg.prompt.ensemble_templates == [
        "bias_guarded_v2",
        "evidence_receipt",
        "evidence_then_answer",
    ]


def test_default_config_has_no_ensemble_templates() -> None:
    cfg = load_config(Path("configs/experiment/qwen3vl_8b.yaml"))
    assert cfg.prompt.ensemble_templates is None
