from pathlib import Path

from skku_vqa.config import load_config


def test_qwen3vl_32b_config() -> None:
    cfg = load_config(Path("configs/experiment/qwen3vl_32b.yaml"))

    assert cfg.experiment_name == "qwen3vl_32b"
    assert cfg.model.name == "qwen3_vl"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-32B-Instruct"
    assert cfg.model.backend == "transformers"
    assert cfg.model.load_in_4bit is True
    assert cfg.model.max_pixels == 200704
    assert cfg.prompt.template == "bias_guarded"
