from pathlib import Path

from skku_vqa.config import load_config


def test_qwen_config_loads_max_pixels_for_16gb_gpu() -> None:
    cfg = load_config(Path("configs/experiment/qwen25vl_7b.yaml"))

    assert cfg.model.max_pixels == 786432
