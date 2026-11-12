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
    assert p.steps[0].compound and p.steps[0].error == "none" and p.first_error is None


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


# ----------------------------------------------------------------------------- v3 cases
def test_division_by_constant_equals_multiplication_by_reciprocal():
    q = "A cake has 50 slices. Half are eaten. How many are left?"
    prog, sl = gold_program(q, "1/2*50 = 25\n#### 25", 25, "gsm_symbolic")
    assert prog.ok
    p = parse_trace("Step 1: 50 / 2 = 25\nAnswer: 25", "marked", sl, prog)
    assert (p.steps[0].gold, p.steps[0].error) == (0, "none")


def test_regrouping_is_a_valid_rederivation():
    q = "Tom had $300. He spent $225 on a bike and $45 on a helmet. How much is left?"
    prog, sl = gold_program(q, "300 - 225 = 75\n75 - 45 = 30\n#### 30", 30, "gsm_symbolic")
    p = parse_trace("Step 1: 225 + 45 = 270\nStep 2: 300 - 270 = 30\nAnswer: 30",
                    "marked", sl, prog)
    assert p.first_error is None
    assert p.steps[1].error == "none" and p.steps[1].rederived


def test_weak_match_does_not_use_up_gold_step():
    # step 2 is redundant and shares only an operand with gold 1; gold 1 must stay free
    p = _parse("Step 1: 4 * 12 = 48\nStep 2: 48 - 0 = 48\nStep 3: 48 - 9 = 39\nAnswer: 39")
    assert (p.steps[2].gold, p.steps[2].tier, p.steps[2].error) == (1, 1, "none")


def test_wrong_compound_step_is_arithmetic_error():
    p = _parse("Step 1: 4 * 12 - 9 = 40\nAnswer: 40")
    assert p.steps[0].compound and p.steps[0].error == "arithmetic" and p.first_error == 0


def test_chain_registers_final_value():
    p = _parse("Step 1: 4 * 12 = 40 + 8 = 48\nStep 2: 48 - 9 = 39\nAnswer: 39")
    assert p.steps[1].a_ref == ("res", 0) and p.first_error is None


@pytest.mark.parametrize("text,ops", [
    ("5 plants * 14 strawberries/plant = 70", ["*"]),
    ("300 km / 15 km/liter = 20 liters", ["/"]),
    ("150 km (each way) * 2 = 300 km", ["*"]),
    ("$10 (salad) + $25 (hamburgers) = $35", ["+"]),
    ("20 emails/day * 30 days = 600", ["*"]),
])
def test_more_unit_forms(text, ops):
    eqs, _ = find_equations(text)
    assert len(eqs) == 1 and [s.op for s in eqs[0].steps] == ops and eqs[0].consistent


def test_powers_and_truncated_inconsistent_sides_are_rejected():
    assert find_equations("A = 10000(1 + 0.20/1)^(1*3) = 10000")[0] == []
    eqs, _ = find_equations("8 minutes/day * 5 days + 90 minutes = 40 minutes + 90 minutes = 130")
    assert eqs[0].chained and eqs[0].consistent and eqs[-1].rhs == 130


# ----------------------------------------------------------------------------- v4 (hand check 3)
def _gsm(q, sol, trace, style):
    from parsing.answers import gsm8k_gold
    prog, sl = gold_program(q, sol, gsm8k_gold(sol), "gsm8k")
    assert prog.ok
    return parse_trace(trace, style, sl, prog)


Q_ELIZA = ("Eliza can iron a blouse in 15 minutes and a dress in 20 minutes.  If she spends 2 hours "
           "ironing blouses and 3 hours ironing dresses, how many pieces of clothes did she iron?")
S_ELIZA = ("In 2 hours, Eliza ironed blouses for 2 x 60 = <<2*60=120>>120 minutes.\n"
           "Eliza can iron 120 / 15 = <<120/15=8>>8 blouses.\n"
           "In 3 hours, Eliza ironed dresses for 3 x 60 = <<3*60=180>>180 minutes.\n"
           "Eliza can iron 180 / 20 = <<180/20=9>>9 dresses.\n"
           "In total, Eliza ironed 8 + 9 = <<8+9=17>>17 pieces of clothes.\n#### 17")


def test_reusing_an_earlier_wrong_result_is_not_a_new_error():
    p = _gsm(Q_ELIZA, S_ELIZA,
             "Step 1: 2 hours * 60 minutes/hour = 120 minutes\n"
             "Step 2: 120 minutes / 15 minutes/blouse = 8 blouses\n"
             "Step 3: 120 minutes / 20 minutes/dress = 6 dresses\n"
             "Step 4: 8 blouses + 6 dresses = 14 pieces of clothes\nAnswer: 14", "marked")
    assert [s.error for s in p.steps] == ["none", "none", "binding", "none"]
    assert p.first_error == 2 and not p.steps[2].extra_valid    # 14 is not a gold value


Q_LISA = ("Lisa makes breakfast for her family every morning, Monday through Friday. She makes 2 eggs "
          "for each of her 4 children and 3 eggs for her husband. Then, she makes 2 eggs for "
          "herself. How many eggs does Lisa cook for her family for breakfast in a year?")
S_LISA = ("Lisa makes 2 eggs/child x 4 children = 8 eggs for her children.\n"
          "She makes 3 eggs for her husband + 2 for herself + 8 for her children = "
          "<<3+2+8=13>>13 eggs.\nLisa makes breakfast 5 days a week for 52 weeks a year, so "
          "5 * 52 = <<5*52=260>>260 breakfasts.\nSo Lisa makes 13 eggs/breakfast x 260 breakfasts"
          " = <<13*260=3380>>3,380 eggs.\n#### 3,380")


def test_valid_regrouping_through_a_weak_match_is_extra_valid():
    p = _gsm(Q_LISA, S_LISA,
             "* Children: 4 children * 2 eggs/child = 8 eggs\n* Total: 8 + 3 + 2 = 13 eggs\n"
             "* Total: 13 eggs/day * 5 days/week = 65 eggs\n"
             "* Total: 65 eggs/week * 52 weeks/year = 3380 eggs", "terse")
    week = p.steps[2]
    assert week.error == "binding" and week.extra_valid       # 65 feeds 65 * 52 = 3380
    assert p.steps[3].error == "none" and p.first_error is None


Q_DIANA = ("Diana needs to bike 10 miles to get home.  She can bike 3 mph for two hours before she "
           "gets tired, and she can bike 1 mph until she gets home.  How long will it take Diana "
           "to get home?")
S_DIANA = ("In the first part of her trip, Diana will cover 2 hours * 3 mph = <<2*3=6>>6 miles.\n"
           "Diana will need to cover an additional 10 miles - 6 miles = <<10-6=4>>4 miles.\n"
           "To cover 4 miles * 1 mph = will take Diana <<4*1=4>>4 hours.\n"
           "Total biking time will be 2 hours + 4 hours = <<2+4=6>>6 hours.\n#### 6")


def test_unit_before_x_and_label_equations():
    eqs, _ = find_equations("Distance = Speed x Time = 3 mph x 2 hours = 6 miles")
    assert [(e.lhs, e.rhs) for e in eqs] == [("3 mph x 2 hours", 6)]
    eqs, _ = find_equations("Total Time = Time biking at 3 mph + Time biking at 1 mph = "
                            "2 hours + 4 hours = 6 hours")
    assert [(e.lhs, e.rhs) for e in eqs] == [("2 hours + 4 hours", 6)]
    # a real slip in a chain is kept (no words after the operator)
    eqs, _ = find_equations("5 + 3 = 9 + 2 = 11")
    assert [(e.lhs, e.rhs, e.consistent) for e in eqs] == [("5 + 3", 9, False), ("9 + 2", 11, True)]
    p = _gsm(Q_DIANA, S_DIANA,
             "Step 1: Distance = Speed x Time = 3 mph x 2 hours = 6 miles\n"
             "Step 2: Remaining = 10 miles - 6 miles = 4 miles\n"
             "Step 3: Time = 4 miles / 1 mph = 4 hours\n"
             "Step 4: Total Time = Time biking at 3 mph + Time biking at 1 mph = 2 hours + 4 hours"
             " = 6 hours\nAnswer: 6", "marked")
    assert p.first_error is None and all(s.error == "none" for s in p.steps)


def test_title_abbreviation_is_not_a_sentence_end():
    eqs, _ = find_equations("Total: 34 cupcakes + 1 cupcake (for Ms. Delmont) + 1 cupcake (for Mrs. "
                            "Donnelly) + 1 cupcake (for the school nurse) + 1 cupcake (for the "
                            "school principal) = 38 cupcakes")
    assert len(eqs) == 1 and eqs[0].rhs == 38 and eqs[0].consistent


Q_ANNIE = ("Annie walked 5 blocks from her house to the bus stop. She rode the bus 7 blocks to the "
           "coffee shop. Later, she came home the same way. How many blocks did Annie travel in "
           "all?")
S_ANNIE = ("Annie traveled 5 + 7 = <<5+7=12>>12 blocks to the coffee shop.\n"
           "Her round trip would take 12 x 2 = <<12*2=24>>24 blocks.\n#### 24")


def test_unaligned_compound_step_is_a_planning_error():
    p = _gsm(Q_ANNIE, S_ANNIE, "Step 1: Total Distance = 5 + 7 + 5 = 17 blocks\nAnswer: 17",
             "marked")
    assert p.steps[0].compound and p.steps[0].error == "planning" and p.first_error == 0
    p = _gsm(Q_ANNIE, S_ANNIE, "Step 1: Total = 5 + 7 + 5 + 7 = 24 blocks\nAnswer: 24", "marked")
    assert p.steps[0].error == "none" and p.first_error is None
    p = _gsm(Q_ANNIE, S_ANNIE, "Step 1: Total = 5 + 7 + 5 + 7 = 25 blocks\nAnswer: 25", "marked")
    assert p.steps[0].error == "arithmetic" and p.steps[0].gold == 1   # aligned by exact value


Q_ANDY = ("Andy is checking to see if all the cars in the parking lot paid for their parking. 75% of "
          "the cars have valid tickets, and 1/5th that number have permanent parking passes. If "
          "there are 300 cars in the parking lot, how many people tried to park in the lot without "
          "paying?")
S_ANDY = ("First find how many cars have valid tickets: 75% * 300 cars = <<75*.01*300=225>>225 cars\n"
          "Then find the percentage of cars with parking passes: 75% / 5 = 15%\n"
          "Then multiply that number by the total number of cars to find the number with parking "
          "passes: 15% * 300 cars = <<15*.01*300=45>>45 cars\n"
          "Then subtract all the cars with valid forms of parking to find the number parked "
          "illegally: 300 cars - 225 cars - 45 cars = <<300-225-45=30>>30 cars\n#### 30")


def test_reciprocal_constant_is_valid():
    p = _gsm(Q_ANDY, S_ANDY,
             "Step 1: 75% of 300 cars = (75/100) * 300 = 225 cars\n"
             "Step 2: 1/5th of 225 cars = (1/5) * 225 = 45 cars\n"
             "Step 3: 225 cars + 45 cars = 270 cars\n"
             "Step 4: 300 cars - 270 cars = 30 cars\nAnswer: 30", "marked")
    assert p.steps[1].error == "none" and p.steps[1].rederived
    assert p.first_error is None


Q_JESSIE = ("Jessie invited 4 friends over to play. They made muffins for a snack and Jesse wants to "
            "divide them equally between herself and her friends. If they made 20 muffins in total, "
            "how many will each person have?")
S_JESSIE = "There are 1 + 4 = <<1+4=5>>5 people.\nEach has 20 / 5 = <<20/5=4>>4 muffins.\n#### 4"


def test_last_step_off_the_answer_gets_no_exemption():
    # 20 / 4 = 5 equals gold's 1 + 4 = 5 by value only, and it ends the trace off the answer
    p = _gsm(Q_JESSIE, S_JESSIE, "Step 1: 20 muffins / 4 people = 5 muffins per person\n"
             "Answer: 5", "marked")
    assert p.steps[0].error == "planning" and not p.steps[0].rederived and p.first_error == 0


def test_swapped_operands_of_subtraction_are_a_binding_error():
    p = _parse("Step 1: 4 * 12 = 48\nStep 2: 9 - 48 = -39\nAnswer: -39")
    assert p.steps[1].tier == 4 and p.steps[1].error == "binding" and p.first_error == 1


def test_two_line_label_equation_units_and_hyphens():
    p = _parse("Step 1: Pencils = 4 packs x 12\nPencils = 48\n"
               "Step 2: Pencils left = 48 - 9\nPencils left = 39\nAnswer: 39")
    assert [s.error for s in p.steps] == ["none", "none"] and p.first_error is None
    eqs, _ = find_equations("Wall: 3m * 2m = 6 square meters")
    assert [(e.lhs, e.rhs) for e in eqs] == [("3m * 2m", 6)]
    assert find_equations("2m - 1 = 5")[0] == []                    # still algebra
    eqs, _ = find_equations("5 time-outs * 5 = 25 time-outs")
    assert [(e.lhs, e.rhs) for e in eqs] == [("5 time-outs * 5", 25)]


Q_PCT = ("There are 400 students. 120 students take dance as their elective. 200 students take art "
         "as their elective. The rest take music. What percentage of students take music?")
S_PCT = ("There are 400 - 120 - 200 = <<400-120-200=80>>80 students in music.\n"
         "Thus, music students make up 80/400 * 100 = <<80/400*100=20>>20% of the students.\n#### 20")


def test_percent_result_matches_gold_percentage():
    p = _gsm(Q_PCT, S_PCT, "Step 1: 120 + 200 = 320\nStep 2: 400 - 320 = 80\n"
             "Step 3: (80 / 400) * 100% = 20%\nAnswer: 20", "marked")
    assert p.steps[2].error == "none" and p.first_error is None


def test_compound_support_feeding_a_gold_value_is_extra_valid():
    p = _parse("Step 1: 4 * 12 - 5 = 43\nStep 2: 43 - 4 = 39\nAnswer: 39")
    assert p.steps[0].compound and p.steps[0].error == "planning"     # 43 is no gold value
    assert p.steps[0].extra_valid and p.first_error is None           # 48 - 9 regrouped
