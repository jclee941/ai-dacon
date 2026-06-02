from skku_vqa.data.dataset import _parse_answers


def test_json_answers():
    raw = '["The Roma person", "The White person", "Not enough information"]'
    assert _parse_answers(raw) == ["The Roma person", "The White person", "Not enough information"]


def test_non_json_fallback():
    assert _parse_answers("single") == ["single"]
