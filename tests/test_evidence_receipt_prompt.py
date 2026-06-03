from skku_vqa.data.schema import Sample
from skku_vqa.prompting.builders import build_prompt
from skku_vqa.prompting.templates import TEMPLATES


def test_evidence_receipt_template_registered() -> None:
    assert "evidence_receipt" in TEMPLATES


def test_evidence_receipt_requires_option_evidence_and_llm_answer() -> None:
    s = Sample(
        id="x",
        question="Who arrived late?",
        options=["The student", "The teacher", "Cannot be determined"],
        context="A student and a teacher are mentioned.",
        image_path=None,
    )
    p = build_prompt(s, "evidence_receipt")
    low = p.lower()
    # input fields are interpolated
    assert "Who arrived late?" in p
    assert "Cannot be determined" in p
    assert "A student and a teacher are mentioned." in p
    # burden-of-proof: cite direct evidence per option
    assert "direct evidence" in low or "visible evidence" in low
    assert "each option" in low or "per option" in low or "for each option" in low
    # abstain when no option has direct support
    assert "unknown" in low or "cannot be determined" in low or "not enough" in low
    # final answer is LLM-emitted and parseable (규칙 #5)
    assert "answer:" in low
