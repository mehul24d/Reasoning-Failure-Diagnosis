from rfd.generation.answer_extraction import extract_predicted_answer, is_correct


def test_extracts_final_answer_marker():
    text = "First we add 2 and 3 to get 5.\nFinal Answer: 5"
    assert extract_predicted_answer(text) == "5"


def test_extracts_final_answer_with_comma_and_period():
    text = "Final Answer: 1,234."
    assert extract_predicted_answer(text) == "1234"


def test_falls_back_to_last_number_without_marker():
    text = "We compute 2 + 2 = 4 and that's it."
    assert extract_predicted_answer(text) == "4"


def test_returns_none_when_no_number_present():
    assert extract_predicted_answer("I am not sure.") is None


def test_is_correct_numeric_match():
    assert is_correct("42", "42")
    assert is_correct("42.0", "42")
    assert not is_correct("41", "42")


def test_is_correct_handles_fraction():
    assert is_correct("1/2", "0.5")


def test_is_correct_none_prediction():
    assert not is_correct(None, "42")
