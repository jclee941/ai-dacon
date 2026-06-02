from pathlib import Path

from skku_vqa.config import load_config


def test_qwen3vl_hires_config_uses_higher_max_pixels() -> None:
    cfg = load_config(Path("configs/experiment/qwen3vl_8b_hires.yaml"))

    assert cfg.experiment_name == "qwen3vl_8b_hires"
    assert cfg.model.name == "qwen3_vl"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.model.load_in_4bit is True
    assert cfg.model.max_pixels == 786432
    assert cfg.prompt.template == "bias_guarded"
