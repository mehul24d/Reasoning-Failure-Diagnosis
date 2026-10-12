from rfd.taxonomy.arithmetic_check import has_arithmetic_inconsistency
from rfd.taxonomy.classifier import classify_error


def test_arithmetic_check_flags_wrong_equation():
    assert has_arithmetic_inconsistency("We compute 2 + 2 = 5, so...")


def test_arithmetic_check_passes_correct_equation():
    assert not has_arithmetic_inconsistency("We compute 2 + 2 = 4, so...")


def test_arithmetic_check_handles_multiplication_symbols():
    assert not has_arithmetic_inconsistency("3 x 4 = 12")
    assert has_arithmetic_inconsistency("3 x 4 = 11")


def test_classify_correct_trace():
    record = {"correct": True, "predicted_answer": "4", "generated_trace": "2+2=4"}
    assert classify_error(record) == "correct"


def test_classify_parsing_error_when_no_prediction():
    record = {"correct": False, "predicted_answer": None, "generated_trace": "I'm confused."}
    assert classify_error(record) == "parsing_error"


def test_classify_arithmetic_error():
    record = {"correct": False, "predicted_answer": "5", "generated_trace": "2 + 2 = 5. Final Answer: 5"}
    assert classify_error(record) == "arithmetic_error"


def test_classify_planning_error_when_no_plan_segment():
    record = {"correct": False, "predicted_answer": "7", "generated_trace": "The answer is 7."}
    segmentation = {"segments": [{"start": 0, "end": 5, "phase": "arithmetic_execution"}]}
    assert classify_error(record, segmentation) == "planning_error"


def test_classify_execution_error_fallback():
    record = {"correct": False, "predicted_answer": "7", "generated_trace": "The answer is 7."}
    segmentation = {
        "segments": [
            {"start": 0, "end": 2, "phase": "operation_planning"},
            {"start": 2, "end": 5, "phase": "self_verification"},
        ]
    }
    assert classify_error(record, segmentation) == "execution_error"
