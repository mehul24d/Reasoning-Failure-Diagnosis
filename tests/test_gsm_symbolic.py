from rfd.data.distractor import NOOP_CLAUSES, insert_clause, make_variant, sample_distractor_value
from rfd.data.gsm_symbolic import parse_instance, template_crosscheck, template_splits

import numpy as np

TEMPLATE_0 = {
    "question_annotated": (
        "{n, Benny} saw a {x,10}-foot {big_fish,shark} with {k,2} {y,6}-inch remoras attached "
        "to it. What percentage of the {big_fish,shark}'s body length is the combined length of "
        "the remoras?\n\n#init:\n- n = sample(names)\n\n#answer: int((k * y)/ (x * 12) * 100)"
    ),
    "answer_annotated": (
        "First, find the combined length of the remoras in inches: {y} inches/remora * {k} "
        "remoras = <<{y}*{k}={k*y}>>{k*y} inches.\nThen divide that number by 12 to convert it "
        "to feet: {k*y} inches / 12 inches/foot = <<{k*y}/12>>{k*y//12} foot.\nThen divide the "
        "combined remora length in feet by the {big_fish}'s length and multiply by 100% to "
        "express the answer as a percentage: {k*y//12} foot / {x} feet * 100% = "
        "{int(k*y/(12*x)*100)}%\n\n#### {int(k*y/(12*x)*100)}"
    ),
}
ROW_0 = {
    "id": 0,
    "instance": 0,
    "question": (
        "Rania saw a 210-foot whale with 7 72-inch remoras attached to it. What percentage of "
        "the whale's body length is the combined length of the remoras?"
    ),
    "answer": (
        "First, find the combined length of the remoras in inches: 72 inches/remora * 7 remoras "
        "= 504 inches.\nThen divide that number by 12 to convert it to feet: 504 inches / 12 "
        "inches/foot = 42 foot.\nThen divide the combined remora length in feet by the whale's "
        "length and multiply by 100% to express the answer as a percentage: 42 foot / 210 feet "
        "* 100% = 20%\n\n#### 20"
    ),
}


def test_parse_instance_recovers_full_program():
    prob = parse_instance(ROW_0)
    assert prob is not None
    assert prob.uid == "t000-i00" and prob.gold_answer == 20.0
    assert [(s.op, s.result) for s in prob.gold_steps] == [
        ("*", 504.0),
        ("/", 42.0),
        ("/", 0.2),
        ("*", 20.0),
    ]


def test_parse_instance_rejects_program_not_ending_at_gold():
    row = dict(ROW_0, answer=ROW_0["answer"].replace("#### 20", "#### 21"))
    assert parse_instance(row) is None


def test_template_crosscheck_matches_annotations():
    steps = template_crosscheck(TEMPLATE_0, ROW_0)
    assert steps is not None
    assert [s.result for s in steps] == [504.0, 42.0]  # template omits the last line's <<>>


def test_template_splits_disjoint_and_deterministic():
    ids = list(range(97))
    a = template_splits(ids, {"train": 0.5, "dev": 0.2, "test": 0.3}, seed=1)
    b = template_splits(ids, {"train": 0.5, "dev": 0.2, "test": 0.3}, seed=1)
    assert a == b
    assert sum(len(v) for v in a.values()) == 97
    assert not set(a["train"]) & set(a["test"]) and not set(a["dev"]) & set(a["test"])


def test_insert_clause_before_question_and_handles_abbreviations():
    q = "Mrs. Lee saw a house. How much more is it than Mrs. Lee's budget?"
    assert (
        insert_clause(q, "X.")
        == "Mrs. Lee saw a house. X. How much more is it than Mrs. Lee's budget?"
    )
    assert insert_clause("If she eats 5, how many?", "X.") == "X. If she eats 5, how many?"


def test_distractor_value_avoids_forbidden_and_reserved():
    rng = np.random.default_rng(0)
    forbidden = [float(v) for v in range(2, 19)]
    for _ in range(20):
        assert sample_distractor_value(forbidden, rng) == 19


def test_make_variant_carries_explicit_slot():
    v = make_variant(0, ROW_0["question"], [210, 7, 72, 504, 42, 20], seed=3)
    assert v is not None and str(v.d) in v.clause and v.clause in v.question
    assert v.d not in (7, 10, 12)
    assert 0 in NOOP_CLAUSES and make_variant(11, "Q?", [], 0) is None
