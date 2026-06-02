from skku_vqa.ambiguity.unknown_mapper import find_unknown_option, has_unknown_option


def test_english_unknown():
    opts = ["The man", "The woman", "Cannot be determined"]
    assert find_unknown_option(opts) == 2
    assert has_unknown_option(opts)


def test_korean_unknown():
    opts = ["왼쪽 사람", "오른쪽 사람", "알 수 없음"]
    assert find_unknown_option(opts) == 2


def test_no_unknown():
    opts = ["A", "B", "C"]
    assert find_unknown_option(opts) is None
