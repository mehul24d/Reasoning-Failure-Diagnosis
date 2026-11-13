"""Instance-level clause checks, the independent solvers and the final clause table (prereg/S2.md A18-A20)."""
import csv
import importlib.util
import json
from fractions import Fraction

import pytest

from harness import distractor as hd
from harness.clause_solvers import SOLVERS, ratio, solve
from harness.config import REPO_ROOT, experiment_config
from harness.instance_checks import PARTITIVE_RE, grounding_check, partitive_check, stem

Q52 = "Qasim has 5 $50 bills. He buys 6 coloring books for $8 each. He also buys 5 packs of trading cards for $4 each. How much money does he have left?"
Q52b = "Leon has 8 $50 bills. He buys 7 toy cars for $3 each. He also buys 4 packs of stickers for $1 each. How much money does he have left?"


def test_partitive_check_contradiction_absent_group_and_unlisted():
    c = "{d} of the coloring books have animals on the cover."
    assert partitive_check(52, c, Q52, "182", 5)["ok"]
    bad = partitive_check(52, c, Q52, "182", 18)
    assert not bad["ok"] and bad["size"] == 6                    # 18 of the 6 coloring books
    absent = partitive_check(52, c, Q52b, "382", 3)              # no coloring books in this instance
    assert not absent["ok"] and "not read" in absent["note"]
    assert not partitive_check(12345, "{d} of the cats are grey.", Q52, "1", 2)["ok"]   # no referent entry
    assert partitive_check(52, "He has been saving for {d} weeks.", Q52, "1", 99)["note"] == "not partitive"
    assert partitive_check(60, "{d} of his coworkers also work the same shift.", "x", "1", 19)["ok"]  # never stated


def test_grounding_checks():
    qs = [Q52, Q52b]
    assert not grounding_check("{d} of the coloring books are new.", qs, hd.insert_clause)["variable_entity"]
    assert grounding_check("He has been saving for {d} weeks.", qs, hd.insert_clause)["note"] == ""
    assert not grounding_check("She has been saving for {d} weeks.", qs, hd.insert_clause)["pronoun_gender"]
    q = ["Nia eats 5 eggs a day. How many eggs will Nia need for the trip?"]
    assert not grounding_check("The trip lasts {d} days.", q, hd.insert_clause)["no_forward_ref"]
    assert stem("pieces") == "piece" and stem("berries") == "berry" and stem("boxes") == "box"


def test_ratio_words():
    assert ratio("twice") == 2 and ratio("three times") == 3 and ratio("quintuple") == 5
    assert ratio("two-thirds") == Fraction(2, 3) and ratio("a quarter") == Fraction(1, 4)
    assert ratio("1/3") == Fraction(1, 3) and ratio("one-half") == Fraction(1, 2) and ratio("an eighth") == Fraction(1, 8)


def test_solver_ignores_the_clause():
    q = hd.insert_clause(Q52, "He has been saving for 18 weeks.")
    assert solve(52, Q52) == solve(52, q) == 182


def test_final_table_invariants():
    exp = experiment_config()["data"]
    assert set(hd.FINAL_CLAUSES) == set(exp["study_templates"])
    assert set(hd.PREREVIEW_REWRITES) <= set(SOLVERS)            # every rewrite has an independent solver
    assert set(hd.REWRITE_TEMPTATION_AI) == set(hd.PREREVIEW_REWRITES)
    assert set(hd.REWRITE_TEMPTATION_AI.values()) <= {"PASS", "WEAK", "UNSURE"}
    for t, (c, why) in hd.PREREVIEW_REWRITES.items():
        assert why and c != hd.DRAFT_CLAUSES[t]
        if why.startswith("partitive"):
            assert not PARTITIVE_RE.search(c), t                 # failing partitive clauses become non-partitive
    assert exp["clause_check_instances"] == 16


def test_committed_check_record_is_clean():
    rep = json.loads((REPO_ROOT / "reports" / "m1" / "distractor_checks_final.json").read_text())
    assert rep["all_pass"] and rep["templates"] == 89 and rep["instances_checked_per_template"] == 16
    assert rep["rewritten"] == sorted(hd.PREREVIEW_REWRITES)
    sheet = REPO_ROOT.parent / "docs" / "review" / "human_review_final.csv"
    rows = list(csv.DictReader(open(sheet)))
    assert [int(r["template_id"]) for r in rows] == exp_templates()
    assert list(rows[0]) == ["template_id", "split", "original_problem", "problem_with_clause", "rewritten",
                             "reviewer1_is_noop", "reviewer1_grammatical", "reviewer2_is_noop",
                             "reviewer2_grammatical", "notes"]
    assert not any(k.startswith("ai_") for k in rows[0])        # no AI verdicts on the human sheet
    tt = json.loads(hd.TEMPTATION.read_text())["templates"]
    assert len(tt) == 89 and all(v["rating"] in {"PASS", "WEAK", "UNSURE"} for v in tt.values())


def exp_templates():
    return experiment_config()["data"]["study_templates"]


def test_top_up_by_budget():
    from harness.estimate import top_up_by_budget

    old = json.loads((REPO_ROOT / "reports" / "m1" / "estimate_gemma-2-2b-it.json").read_text())
    tu = top_up_by_budget(old["measured"], len(old["gpus"]) or 1)
    assert tu["decision"] == "skip" and tu["reported_as"] == "underpowered"
    assert tu["t4_gpu_hours_total_with_safety"] > tu["budget_t4_gpu_hours"]
    cfg = experiment_config()["data"]["power"]["top_up"]
    assert cfg["decision"] == "skip" and cfg["projected_with_safety_16"] == pytest.approx(
        tu["t4_gpu_hours_total_with_safety"], abs=1e-3)


def _ra():
    spec = importlib.util.spec_from_file_location("review_agreement", REPO_ROOT / "scripts" / "review_agreement.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_new_sheet_columns_are_read():
    ra = _ra()
    rows = [{"template_id": "1", "reviewer1_is_noop": "yes", "reviewer1_grammatical": "yes",
             "reviewer2_is_noop": "no", "reviewer2_grammatical": "yes", "notes": ""}]
    assert ra.reviewer_labels(rows, 1)[1]["verdict"] == "OK"
    assert ra.reviewer_labels(rows, 2)[1]["verdict"] == "FIX"
