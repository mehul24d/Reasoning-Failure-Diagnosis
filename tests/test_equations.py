from rfd.parsing.equations import decompose, parse_equations, program_from_text


def _exprs(text, validate=True):
    return [(e.expr, e.stated) for e in parse_equations(text, validate=validate)]


def test_units_and_dangling_operator():
    text = (
        "72 inches/remora * 7 remoras = 504 inches.\n"
        "Then 504 inches / 12 inches/foot = 42 foot. 42 foot / 210 feet * 100% = 20%"
    )
    assert _exprs(text) == [("72.0*7.0", 504.0), ("504.0/12.0", 42.0), ("42.0/210.0*100.0", 20.0)]


def test_leading_prose_number_is_not_an_operand():
    assert _exprs("In the second 30 seconds of cooking, 4* 60 = 240 kernels pop.") == [
        ("4.0*60.0", 240.0)
    ]


def test_latex_and_currency():
    assert _exprs(r"Step 3: \(16 - 3 = 13\) eggs, \(9 \times \$2 = \$18\)") == [
        ("16.0-3.0", 13.0),
        ("9.0*2.0", 18.0),
    ]
    assert _exprs(r"\[ \frac{350}{50} = 7 \]") == [("(350.0)/(50.0)", 7.0)]


def test_thousands_separator_and_x_operator():
    assert _exprs("He paid 1,200 x 3 = 3,600 dollars") == [("1200.0*3.0", 3600.0)]


def test_validate_false_keeps_wrong_stated_result():
    eqs = parse_equations("So 7 * 8 = 54 apples.", validate=False)
    assert len(eqs) == 1
    step = eqs[0].steps[-1]
    assert step.stated if hasattr(step, "stated") else True
    assert step.result == 54.0 and step.computed == 56.0 and not step.arithmetic_ok
    assert parse_equations("So 7 * 8 = 54 apples.", validate=True) == []


def test_decompose_evaluation_order():
    steps = decompose("1/10*100")
    assert [(s.op, s.a, s.b, s.result) for s in steps] == [
        ("/", 1.0, 10.0, 0.1),
        ("*", 0.1, 100.0, 10.0),
    ]


def test_program_from_text_skips_restatements():
    prog = program_from_text("She has 48=48 emails. 150-48=102 left. 102*6= 612")
    assert [(s.op, s.result) for s in prog] == [("-", 102.0), ("*", 612.0)]


def test_parenthesised_unit_words_are_dropped():
    assert _exprs("70 (building blocks) + 46 (stuffed animals) + 29 (rings) = 145 toys") == [
        ("70.0+46.0+29.0", 145.0)
    ]


def test_model_text_does_not_cross_clause_boundaries():
    text = "- Sunday: $189 + $306 = $495. - Monday: $408 + $560 = $968."
    eqs = parse_equations(text, validate=False)
    assert [(e.expr, e.stated) for e in eqs] == [("189.0+306.0", 495.0), ("408.0+560.0", 968.0)]


def test_slip_takes_minimal_binary_expression():
    eqs = parse_equations("So the total is 12 + 30 + 7 = 50 apples.", validate=False)
    assert eqs[0].expr == "30.0+7.0" and not eqs[0].steps[-1].arithmetic_ok


def test_fraction_on_rhs():
    assert _exprs("The probability is 6/12 = 1/2.") == [("6.0/12.0", 0.5)]
