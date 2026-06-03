from skku_vqa.data.schema import Sample
from skku_vqa.prompting.builders import build_synthesis_prompt


def _sample() -> Sample:
    return Sample(
        id="x",
        question="Who arrived late?",
        options=["The student", "The teacher", "Cannot be determined"],
        context="A student and a teacher are mentioned.",
        image_path=None,
    )


def test_synthesis_prompt_contains_task_and_all_variants() -> None:
    variants = [
        "Analysis A: no direct evidence. ANSWER: 2",
        "Analysis B: the teacher was noted as punctual. ANSWER: 0",
        "Analysis C: cannot tell. ANSWER: 2",
    ]
    p = build_synthesis_prompt(_sample(), variants)
    # original task present
    assert "Who arrived late?" in p
    assert "A student and a teacher are mentioned." in p
    assert "The teacher" in p
    # all candidate analyses embedded verbatim
    for v in variants:
        assert v in p
    # 0-based options enumerated
    assert "0. The student" in p
    assert "2. Cannot be determined" in p
    # LLM must emit the final answer (규칙 #5)
    assert "ANSWER:" in p
    # candidate analyses are untrusted evidence; embedded instructions must be ignored
    low = p.lower()
    assert "untrusted" in low or "do not follow" in low or "ignore any instructions" in low


def test_synthesis_prompt_handles_empty_variants() -> None:
    p = build_synthesis_prompt(_sample(), [])
    assert "Who arrived late?" in p
    assert "ANSWER:" in p
