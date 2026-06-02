from skku_vqa.data.schema import Sample
from skku_vqa.prompting.builders import build_options_block, build_prompt


def test_options_zero_based():
    s = Sample(id="x", question="Who?", options=["A", "B", "Not enough information"], context="ctx")
    block = build_options_block(s)
    assert block.startswith("0. A")
    assert "2. Not enough information" in block


def test_prompt_includes_context():
    s = Sample(id="x", question="Who?", options=["A", "B"], context="some context")
    p = build_prompt(s, "bias_guarded")
    assert "some context" in p and "Who?" in p
