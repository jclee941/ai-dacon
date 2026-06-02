from pathlib import Path

import pytest

from skku_vqa.config import ModelConfig, load_config
from skku_vqa.models import loader


def test_qwen3vl_config_loads_expected_fields() -> None:
    cfg = load_config(Path("configs/experiment/qwen3vl_8b.yaml"))

    assert cfg.experiment_name == "qwen3vl_8b"
    assert cfg.model.name == "qwen3_vl"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.model.backend == "transformers"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.model.load_in_4bit is True
    assert cfg.model.max_new_tokens == 64
    assert cfg.model.max_pixels == 200704
    assert cfg.prompt.template == "bias_guarded"


def test_loader_routes_qwen3_vl_before_generic_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeQwen3VLModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    import skku_vqa.models.qwen3_vl as qwen3_mod

    monkeypatch.setattr(qwen3_mod, "Qwen3VLModel", FakeQwen3VLModel)

    cfg = ModelConfig(
        name="qwen3_vl",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        backend="transformers",
        max_pixels=786432,
    )
    model = loader.load_model(cfg)

    assert isinstance(model, FakeQwen3VLModel)
    assert captured["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert captured["max_pixels"] == 786432


def test_loader_still_routes_qwen2_5_to_qwen_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQwenVLModel:
        def __init__(self, **kwargs: object) -> None:
            pass

    import skku_vqa.models.qwen_vl as qwen_mod

    monkeypatch.setattr(qwen_mod, "QwenVLModel", FakeQwenVLModel)

    cfg = ModelConfig(
        name="qwen2_5_vl",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        backend="transformers",
    )
    model = loader.load_model(cfg)

    assert isinstance(model, FakeQwenVLModel)
