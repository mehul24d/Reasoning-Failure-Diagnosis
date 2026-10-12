from rfd.data.loader import _extract_gold_bbh, _extract_gold_gsm8k, _extract_gold_math


def test_extract_gold_gsm8k_strips_marker_and_commas():
    raw = "Natalia sold 48 clips... #### 1,200"
    assert _extract_gold_gsm8k(raw) == "1200"


def test_extract_gold_gsm8k_raises_without_marker():
    import pytest

    with pytest.raises(ValueError):
        _extract_gold_gsm8k("no marker here")


def test_extract_gold_math_extracts_last_boxed():
    raw = r"First \boxed{2} is wrong, actually \boxed{42}."
    assert _extract_gold_math(raw) == "42"


def test_extract_gold_math_falls_back_without_boxed():
    assert _extract_gold_math("the answer is 7") == "the answer is 7"


def test_extract_gold_bbh_strips_whitespace():
    assert _extract_gold_bbh("  (A)  ") == "(A)"
