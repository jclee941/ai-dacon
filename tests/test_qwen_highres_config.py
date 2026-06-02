from pathlib import Path

from skku_vqa.config import load_config


def test_qwen_highres_config_keeps_bias_guarded_prompt() -> None:
    cfg = load_config(Path("configs/experiment/qwen25vl_7b_mp1m.yaml"))

    assert cfg.experiment_name == "qwen25vl_7b_mp1m"
    assert cfg.prompt.template == "bias_guarded"
    assert cfg.model.max_pixels == 1048576
