from pathlib import Path


def test_auto_submit_pool_is_top5_by_expected_strength() -> None:
    """The scheduled auto-submit must carry the 5 strongest distinct candidates:
    qwen3vl_8b (best), internvl3_8b, qwen3vl_8b_hires, qwen3vl_8b_lowres, qwen3vl_8b_bgv2.
    Weakest qwen25vl_7b (0.90658) is dropped in favor of bgv2 (279-diff prompt variant).
    """
    script = Path("scripts/auto_submit_next_window.sh").read_text()
    must_include = [
        "artifacts/final/qwen3vl_8b.csv",
        "artifacts/final/internvl3_8b.csv",
        "artifacts/final/qwen3vl_8b_hires.csv",
        "artifacts/final/qwen3vl_8b_lowres.csv",
        "artifacts/final/qwen3vl_8b_bgv2.csv",
    ]
    for c in must_include:
        assert c in script, f"missing candidate: {c}"
    assert "artifacts/final/qwen25vl_7b.csv" not in script
