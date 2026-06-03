from pathlib import Path

from skku_vqa.config import load_config


def test_final_config_matches_best_candidate() -> None:
    """artifacts/final/config.yaml must exist and reproduce the winning candidate
    (qwen3vl_8b bias_guarded 200704 = 0.95867) so `make reproduce` works for 2nd-stage."""
    cfg = load_config(Path("artifacts/final/config.yaml"))

    assert cfg.model.name == "qwen3_vl"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.model.max_pixels == 200704
    assert cfg.model.load_in_4bit is True
    assert cfg.prompt.template == "bias_guarded"
    assert cfg.model.max_new_tokens == 64
