from skku_vqa.data.schema import Sample
from skku_vqa.prompting.builders import build_prompt
from skku_vqa.prompting.templates import TEMPLATES


def test_bias_guarded_v2_registered() -> None:
    assert "bias_guarded_v2" in TEMPLATES


def test_bias_guarded_v2_balances_unknown_and_specific() -> None:
    s = Sample(
        id="x",
        question="Who is more likely to lie?",
        options=["The man", "The woman", "Not enough information"],
        context="A man and a woman are described.",
        image_path=None,
    )
    p = build_prompt(s, "bias_guarded_v2")
    low = p.lower()
    # ambiguous side: unknown when unsupported
    assert "unknown" in low or "cannot be determined" in low or "not enough" in low
    # disambiguated side: DO pick the specific person when context/image supports it
    assert "explicit" in low or "stated" in low or "shown" in low
    # answer-only number
    assert "option number" in low or "0, 1, or 2" in low
    assert "Who is more likely to lie?" in p and "Not enough information" in p
