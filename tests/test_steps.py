from rfd.parsing.equations import BinStep
from rfd.parsing.steps import (
    char_to_token,
    content_attributable,
    first_deviation,
    label_steps,
    split_steps,
)

# gold: 20 - 5 = 15 ; 15 * 3 = 45
GOLD = [BinStep("-", 20, 5, 15, 15), BinStep("*", 15, 3, 45, 45)]
QNUMS = [20.0, 5.0, 3.0, 9.0]  # 9 is the distractor


def test_split_steps_stops_at_final_answer():
    text = "Step 1: a 1 + 1 = 2.\nStep 2: b.\nFinal Answer: 2\nStep 3: junk"
    spans = split_steps(text)
    assert [k for k, _, _ in spans] == [1, 2]
    assert text[spans[-1][1] : spans[-1][2]].startswith("Step 2")


def test_correct_trace_all_none():
    text = "Step 1: left 20 - 5 = 15.\nStep 2: total 15 * 3 = 45.\nFinal Answer: 45"
    labs = label_steps(text, GOLD, QNUMS, distractor=9)
    assert [lab.error_type for lab in labs] == ["none", "none"]
    assert [lab.gold_op for lab in labs] == ["-", "*"]
    assert first_deviation(labs) is None


def test_distractor_binding_is_content_attributable():
    text = "Step 1: left 20 - 9 = 11.\nStep 2: total 11 * 3 = 33.\nFinal Answer: 33"
    labs = label_steps(text, GOLD, QNUMS, distractor=9)
    dev = first_deviation(labs)
    assert dev is not None and dev.index == 1 and dev.error_type == "binding"
    assert dev.distractor_used and content_attributable(dev)


def test_arithmetic_slip_is_not_content_attributable():
    text = "Step 1: left 20 - 5 = 14.\nFinal Answer: 42"
    dev = first_deviation(label_steps(text, GOLD, QNUMS, distractor=9))
    assert dev.error_type == "arithmetic" and not content_attributable(dev)


def test_wrong_operation_is_planning_and_content_attributable():
    text = "Step 1: 20 + 5 = 25.\nFinal Answer: 75"
    dev = first_deviation(label_steps(text, GOLD, QNUMS, distractor=9))
    assert dev.error_type == "planning" and dev.op == "+" and dev.gold_op == "-"
    assert content_attributable(dev)


def test_steps_without_equations_are_unparsed():
    labs = label_steps("Step 1: Think about it.\nFinal Answer: 1", GOLD, QNUMS)
    assert labs[0].error_type is None and not labs[0].parsed


def test_char_to_token():
    toks = ["Step", " 1", ":", " 20"]
    assert char_to_token(toks, 0) == 0
    assert char_to_token(toks, 4) == 1
    assert char_to_token(toks, 100) == 3


def test_percent_operand_is_valid():
    gold = [BinStep("*", 400000, 0.07, 28000, 28000)]
    labs = label_steps(
        "Step 1: fee 400000 * 7% = 28000.\nFinal Answer: 28000", gold, [400000.0, 7.0]
    )
    assert labs[0].error_type == "none"
