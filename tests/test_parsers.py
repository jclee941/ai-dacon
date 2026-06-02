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


def test_ignores_leading_picture_index_prefers_answer_marker():
    # "Picture 1:" style prefix must not be parsed as the label
    assert parse_label("Picture 1: the scene. ANSWER: 2", 3) == 2


def test_prefers_last_standalone_number_when_no_marker():
    # CoT-style: reasoning mentions option 1, final standalone answer is 0
    assert parse_label("Option 1 seems likely, but evidence is missing.\n0", 3) == 0


def test_single_digit_unchanged():
    assert parse_label("2", 3) == 2
    assert parse_label("0", 3) == 0


def test_answer_marker_overrides_earlier_numbers():
    assert parse_label("1 and 2 are tempting. ANSWER: 0", 3) == 0
