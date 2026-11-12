from fractions import Fraction as F

import pytest

from harness.splits import split_templates
from parsing.expr import find_equations, parse_expression
from parsing.gold import gold_program, slots
from parsing.steps import parse_trace, segment

# Plan Table 2 style problem: q1 = 4, q2 = 12, q3 = 9, distractor q_delta = 5.
Q = ("Tom buys 4 packs of pencils with 12 pencils in each pack. He gives 9 pencils to friends. "
     "His sister already had 5 pencils. How many pencils does Tom have now?")
SOL_SYM = "Tom buys 4 * 12 = 48 pencils.\nHe has 48 - 9 = 39 pencils left.\n#### 39"
SOL_GSM8K = "He buys 4*12=<<4*12=48>>48 pencils.\nHe has 48-9=<<48-9=39>>39 left.\n#### 39"


# ----------------------------------------------------------------------------- expressions
@pytest.mark.parametrize("expr,ops,value", [
    ("4 * 12", ["*"], 48),
    ("4 x 12", ["*"], 48),
    ("$1,200 / 4", ["/"], 300),
    ("3 + 4 * 2", ["*", "+"], 11),
    ("(3 + 4) * 2", ["+", "*"], 14),
    ("10 - 2 - 3", ["-", "-"], 5),
    ("7 / 2", ["/"], F(7, 2)),
])
def test_parse_expression(expr, ops, value):
    steps = parse_expression(expr)
    assert [s.op for s in steps] == ops and steps[-1].result == value


@pytest.mark.parametrize("expr", ["12", "", "4 *", "5 / 0", "2m - 1", "12 dogs and 3 cats"])
def test_unparseable(expr):
    assert parse_expression(expr) is None


def test_find_equations_chain_latex_and_lists():
    eqs, skipped = find_equations(r"So 4 \times 12 = 48 - 9 = 39. Items: 3, 4 + 5 = 9")
    assert [(e.lhs, e.rhs) for e in eqs] == [(r"4 \times 12", 48), ("48 - 9", 39), ("4 + 5", 9)]
    assert skipped == 0
    eqs, _ = find_equations("Step 1. 4 * 12 = 48\nStep 2) 48 - 9 = 39. He has 3. 2 + 2 = 4")
    assert [(e.lhs, e.rhs) for e in eqs] == [("4 * 12", 48), ("48 - 9", 39), ("2 + 2", 4)]
    eqs, _ = find_equations("Total = 5 + 2 = 8")             # inconsistent is kept, flagged
    assert len(eqs) == 1 and not eqs[0].consistent


# ----------------------------------------------------------------------------- gold
def test_slots_and_distractor():
    span = (Q.index("His sister"), Q.index("How many"))
    sl = slots(Q, span)
    assert [s.value for s in sl] == [4, 12, 9, 5]
    assert [s.distractor for s in sl] == [False, False, False, True]


@pytest.mark.parametrize("source,sol", [("gsm_symbolic", SOL_SYM), ("gsm8k", SOL_GSM8K)])
def test_gold_program(source, sol):
    prog, sl = gold_program(Q, sol, "39", source)
    assert prog.ok, prog.issues
    assert [(g.op, g.a_ref, g.b_ref, g.result) for g in prog.steps] == [
        ("*", ("slot", 0), ("slot", 1), 48), ("-", ("res", 0), ("slot", 2), 39)]


def test_gold_compound_and_constants():
    q = "A show lasts 2 hours and 30 minutes. How many minutes is that?"
    prog, _ = gold_program(q, "2 * 60 + 30 = 150 minutes\n#### 150", 150, "gsm_symbolic")
    assert prog.ok
    assert [(g.op, g.a_ref, g.b_ref) for g in prog.steps] == [
        ("*", ("slot", 0), ("const", 60)), ("+", ("res", 0), ("slot", 1))]


def test_gold_rejects_wrong_answer_and_bad_equation():
    prog, _ = gold_program(Q, SOL_SYM, "40", "gsm_symbolic")
    assert not prog.ok and "answer" in prog.issues[0]
    prog, _ = gold_program(Q, "4 * 12 = 50\n50 - 9 = 41\n#### 41", "41", "gsm_symbolic")
    assert not prog.ok and "inconsistent" in prog.issues[0]


# ----------------------------------------------------------------------------- steps
def _parse(trace, style="marked", span=None):
    prog, sl = gold_program(Q, SOL_SYM, "39", "gsm_symbolic", span)
    return parse_trace(trace, style, sl, prog)


def test_segment_marked_and_terse():
    t = "Sure!\nStep 1: 4 * 12 = 48\n**Step 2:** 48 - 9 = 39\n\nAnswer: 39"
    spans = segment(t, "marked")
    assert [t[s:e] for s, e in spans] == ["Step 1: 4 * 12 = 48", "**Step 2:** 48 - 9 = 39"]
    t2 = "4 * 12 = 48, then 48 - 9 = 39\nAnswer: 39"
    assert [t2[s:e] for s, e in segment(t2, "terse")] == ["4 * 12 = 48", ", then 48 - 9 = 39"]


def test_correct_trace_has_no_errors():
    p = _parse("Step 1: 4 * 12 = 48\nStep 2: 48 - 9 = 39\nAnswer: 39")
    assert [(s.gold, s.error) for s in p.steps] == [(0, "none"), (1, "none")]
    assert p.first_error is None and p.n_unmatched_gold == 0


def test_operands_in_either_order_are_not_binding_errors():
    p = _parse("Step 1: 12 * 4 = 48\nStep 2: 48 - 9 = 39\nAnswer: 39")
    assert [s.error for s in p.steps] == ["none", "none"]


def test_binding_error_on_distractor():
    span = (Q.index("His sister"), Q.index("How many"))
    p = _parse("Step 1: 4 * 12 = 48\nStep 2: 48 - 5 = 43\nAnswer: 43", span=span)
    s = p.steps[1]
    assert (s.gold, s.error, s.b_ref) == (1, "binding", ("slot", 3)) and p.first_error == 1


def test_planning_error_wrong_operation():
    p = _parse("Step 1: 4 + 12 = 16\nStep 2: 16 - 9 = 7\nAnswer: 7")
    assert p.steps[0].error == "planning" and p.first_error == 0


def test_arithmetic_error_and_propagation():
    p = _parse("Step 1: 4 * 12 = 46\nStep 2: 46 - 9 = 37\nAnswer: 37")
    # step 2 uses its own earlier result correctly: only step 1 is an error
    assert [s.error for s in p.steps] == ["arithmetic", "none"] and p.first_error == 0


def test_extra_step_and_non_computational():
    p = _parse("Step 1: Tom has packs.\nStep 2: 4 * 12 = 48\nStep 3: 48 - 9 = 39\n"
               "Step 4: 39 + 5 = 44\nAnswer: 44")
    assert [s.computational for s in p.steps] == [False, True, True, True]
    assert [s.error for s in p.steps] == [None, "none", "none", "planning"]
    assert not p.steps[3].extra_valid and p.first_error == 3     # 44 is not a gold quantity


def test_compound_step_is_flagged_but_feeds_later_steps():
    p = _parse("Step 1: 4 * 12 - 9 = 39\nAnswer: 39")
    assert p.steps[0].compound and p.steps[0].error is None and p.first_error is None


# ----------------------------------------------------------------------------- splits
def test_split_templates_deterministic_and_disjoint():
    ids = [f"t{i}" for i in range(100)]
    a, b = split_templates(ids, seed=3), split_templates(list(reversed(ids)), seed=3)
    assert a == b
    assert [len(a[k]) for k in ("train", "dev", "test")] == [40, 20, 40]
    assert set(a["train"]) | set(a["dev"]) | set(a["test"]) == set(ids)
    assert not set(a["train"]) & set(a["test"])
    with pytest.raises(ValueError):
        split_templates(ids[:50])


# ----------------------------------------------------------------------------- real-trace cases
# (failures found in the manual check of 40 parsed S1c steps, reports/m0/parsing_manual_check.md)
@pytest.mark.parametrize("text,ops,rhs", [
    ("10 baskets * $9/basket = $90", ["*"], 90),
    ("2 pairs * $9.50/pair = $19.00", ["*"], 19),
    ("100 horses + 29 sheep + 9 chickens = 138", ["+", "+"], 138),
    ("200 seats * (3/4) = 150", ["*"], 150),
    ("60*20/100 = 12", ["*"], 12),               # 20/100 is a literal: 60 * 1/5
    ("60 * 20% = 12", ["*"], 12),
    ("21/4÷3/4 = 7", ["/"], 7),
    ("11 1/4 + 3 = 14 1/4", ["+"], F(57, 4)),
    ("2(38+30) = 136", ["+", "*"], 136),
    ("¥20 * 3 = ¥60", ["*"], 60),
    ("3/4 = 75%", ["/"], F(3, 4)),
    ("He has 3 dogs and 10 * 9 = 90", ["*"], 90),
    ("-5 + 3 = -2", ["+"], -2),
])
def test_real_trace_equations(text, ops, rhs):
    eqs, _ = find_equations(text)
    assert len(eqs) == 1
    assert [s.op for s in eqs[0].steps] == ops and eqs[0].rhs == rhs and eqs[0].consistent


def test_algebra_is_rejected_not_misread():
    eqs, skipped = find_equations("2m - 1 + 1 = 2m")
    assert eqs == [] and skipped == 1


def test_spans_point_into_original_text():
    t = r"So **4 \times 12 = 48** pens"
    eq = find_equations(t)[0][0]
    assert t[eq.start:eq.end] == r"4 \times 12 = 48"


def test_gsm8k_gold_uses_unannotated_inline_equations():
    q = "Ann has 20 apples. She buys 2 bags of 20 apples and 4 more. How many apples now?"
    sol = "She buys 2*20 = 40 apples.\nThen 40+4 = 44.\nTotal 44+20=<<44+20=64>>64\n#### 64"
    prog, _ = gold_program(q, sol, 64, "gsm8k")
    assert prog.ok and [g.result for g in prog.steps] == [40, 44, 64]


def test_percent_slot_matches_fraction_operand():
    q = "A shirt costs $60. It is 20% off. How much is the discount?"
    prog, sl = gold_program(q, "60*20/100=<<60*20/100=12>>12\n#### 12", 12, "gsm8k")
    assert prog.ok and prog.steps[0].b_ref == ("slot", 1)
    p = parse_trace("Step 1: 60 * 0.20 = 12\nAnswer: 12", "marked", sl, prog)
    assert p.steps[0].error == "none"


def test_alignment_prefers_exact_step_over_shared_operand():
    # gold: 183*26 = 4758 ; 26/2 = 13 ; 28*13 = 364 (three independent pieces)
    q = "A has 183 rows of 26 seats. Half of 26 is used by 28 classes. How many?"
    sol = "183*26=4758\n26/2=13\n28*13=364\n4758-364=4394\n#### 4394"
    prog, sl = gold_program(q, sol, 4394, "gsm_symbolic")
    assert prog.ok
    p = parse_trace("Step 1: 26/2 = 13\nStep 2: 28 * 13 = 364\nStep 3: 183 * 26 = 4758\n"
                    "Step 4: 4758 - 364 = 4394\nAnswer: 4394", "marked", sl, prog)
    assert [(s.gold, s.error) for s in p.steps] == [(1, "none"), (2, "none"), (0, "none"),
                                                    (3, "none")]


def test_compound_result_feeds_later_step():
    p = _parse("Step 1: 4 * 12 - 9 = 39\nStep 2: 39 + 0 = 39\nAnswer: 39")
    assert p.steps[0].compound and p.steps[0].gold == 1
    assert p.steps[1].a_ref == ("res", 1)


def test_redundant_valid_step_is_extra_valid():
    p = _parse("Step 1: 4 * 12 = 48\nStep 2: 48 - 9 = 39\nStep 3: 48 - 9 = 39\nAnswer: 39")
    s3 = p.steps[2]
    assert s3.error == "planning" and s3.extra_valid
    assert p.first_error is None and p.first_error_strict == 2


def test_terse_line_with_two_equations_and_markdown():
    t = "**4 * 12 = 48** and **48 - 9 = 39**\nAnswer: 39"
    p = _parse(t, style="terse")
    assert [(s.gold, s.error) for s in p.steps] == [(0, "none"), (1, "none")]
