from skku_vqa.prompting.parsers import parse_label


def test_answer_prefix():
    assert parse_label("ANSWER: 2", 3) == 2


def test_first_valid_int():
    assert parse_label("The answer is 1 because...", 3) == 1


def test_zero_is_valid():
    assert parse_label("0", 3) == 0


def test_out_of_range_returns_none():
    assert parse_label("9", 3) is None


def test_empty():
    assert parse_label("", 3) is None
