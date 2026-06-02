from pathlib import Path

import pytest

from skku_vqa.config import ModelConfig, load_config
from skku_vqa.models import loader


def test_internvl3_config_loads() -> None:
    cfg = load_config(Path("configs/experiment/internvl3_8b.yaml"))

    assert cfg.experiment_name == "internvl3_8b"
    assert cfg.model.name == "internvl3"
    assert cfg.model.model_id == "OpenGVLab/InternVL3-8B-hf"
    assert cfg.model.backend == "transformers"
    assert cfg.model.load_in_4bit is True
    assert cfg.prompt.template == "bias_guarded"


def test_loader_routes_internvl3(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeInternVL3Model:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    import skku_vqa.models.internvl3 as mod

    monkeypatch.setattr(mod, "InternVL3Model", FakeInternVL3Model)

    cfg = ModelConfig(
        name="internvl3",
        model_id="OpenGVLab/InternVL3-8B-hf",
        backend="transformers",
        max_pixels=200704,
    )
    model = loader.load_model(cfg)

    assert isinstance(model, FakeInternVL3Model)
    assert captured["model_id"] == "OpenGVLab/InternVL3-8B-hf"
