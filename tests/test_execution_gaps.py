"""TDD: execution gap tests (sync/setup/subset/fallback). RED before impl."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_run_inference_has_limit_flag():
    """run_inference.py must support --limit for subset smoke runs."""
    src = (REPO / "scripts/run_inference.py").read_text(encoding="utf-8")
    assert '"--limit"' in src or "'--limit'" in src, "run_inference.py missing --limit flag"


def test_loader_routes_transformers_backend():
    """loader must route llava + backend=transformers to a transformers wrapper."""
    src = (REPO / "src/skku_vqa/models/loader.py").read_text(encoding="utf-8")
    assert "transformers" in src, "loader missing transformers backend routing"
    assert "llava_ov_transformers" in src or "LlavaOneVisionTransformers" in src, (
        "loader does not reference llava transformers fallback wrapper"
    )


def test_transformers_fallback_module_exists():
    """The transformers fallback wrapper module must exist."""
    assert (REPO / "src/skku_vqa/models/llava_ov_transformers.py").exists(), (
        "missing src/skku_vqa/models/llava_ov_transformers.py"
    )


def test_fallback_config_exists():
    """A transformers-backend experiment config must exist."""
    cfg = REPO / "configs/experiment/baseline_llava_ov_transformers.yaml"
    assert cfg.exists(), "missing baseline_llava_ov_transformers.yaml"
    text = cfg.read_text(encoding="utf-8")
    assert "backend: transformers" in text


def test_sync_includes_test_data():
    """sync.sh must NOT unconditionally exclude the test data needed for inference."""
    src = (REPO / "scripts/remote/sync.sh").read_text(encoding="utf-8")
    # 데이터 동기화 지원: WITH_DATA 모드 또는 test 데이터 포함 규칙이 있어야 함
    assert "WITH_DATA" in src or "data/raw/test" in src, (
        "sync.sh provides no way to send test data to remote"
    )


def test_setup_installs_cu128_torch_before_vllm():
    """setup.sh must install pre-release cu128 torch BEFORE vllm (Blackwell sm_120)."""
    src = (REPO / "scripts/remote/setup.sh").read_text(encoding="utf-8")
    assert "cu128" in src, "setup.sh does not install cu128 torch for Blackwell"
    torch_idx = src.find("cu128")
    vllm_idx = src.find("vllm")
    assert torch_idx != -1 and vllm_idx != -1 and torch_idx < vllm_idx, (
        "cu128 torch must be installed before vllm"
    )
