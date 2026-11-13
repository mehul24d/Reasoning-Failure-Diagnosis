"""The distractor condition (prereg/S2.md A14, A17) and the review reconciliation (D39)."""
import importlib.util
import sys

import pytest

from harness import distractor as hd
from harness.config import REPO_ROOT, experiment_config

np = pytest.importorskip("numpy")

ROW = {"problem_id": "gsms-main-4-3", "template_id": "gsms-4", "source": "gsm_symbolic",
       "question": "Liam buys 4 boxes of pencils. Each box holds 13 pencils. How many pencils does he buy?",
       "solution": "He buys 4 * 13 = 52 pencils.\n#### 52", "answer": "52"}


def test_clause_table_covers_the_study_and_matches_the_baseline():
    study = experiment_config()["data"]["study_templates"]
    assert sorted(hd.DRAFT_CLAUSES) == sorted(study)
    root = REPO_ROOT.parent / "src"
    if not (root / "rfd" / "data" / "distractor.py").exists():
        pytest.skip("baseline package not present")
    sys.path.insert(0, str(root))
    from rfd.data import distractor as base

    assert {t: c for t, c in hd.DRAFT_CLAUSES.items() if t != 50} == {t: base.NOOP_CLAUSES[t] for t in study if t != 50}
    assert 50 not in base.NOOP_CLAUSES                       # drafted for the study (A17)
    q = "Ann has 3 cats. Mrs. Lee has 2. How many cats are there?"
    for d in (5, 9):
        assert hd.insert_clause(q, f"{d} of them are black.") == base.insert_clause(q, f"{d} of them are black.")
    for seed in range(20):
        assert hd.sample_distractor_value([3.0, 5.0], np.random.default_rng(seed)) == \
            base.sample_distractor_value([3.0, 5.0], np.random.default_rng(seed))


def test_pair_rows_builds_both_conditions():
    out = hd.pair_rows([ROW], {4: "{d} of the boxes are blue."})
    clean, dis = out
    assert clean["condition"] == "clean" and clean["question"] == ROW["question"]
    assert dis["condition"] == "distractor" and dis["problem_id"] == "gsms-noop-4-3"
    assert dis["pair_id"] == clean["pair_id"] == "gsms-main-4-3"
    a, b = dis["distractor_span"]
    assert dis["question"][a:b] == dis["clause"] and str(dis["distractor"]) in dis["clause"]
    assert dis["answer"] == ROW["answer"] and dis["solution"] == ROW["solution"]
    # the gold program sees the clause's number as the distractor slot and is unchanged
    from parsing.gold import gold_program

    p0, _ = gold_program(ROW["question"], ROW["solution"], ROW["answer"], "gsm_symbolic")
    p1, sl = gold_program(dis["question"], dis["solution"], dis["answer"], "gsm_symbolic", distractor_span=(a, b))
    assert p1.answer == p0.answer and sum(s.distractor for s in sl) == 1
    # a template without a reviewed clause keeps only the clean condition
    assert [r["condition"] for r in hd.pair_rows([ROW], {})] == ["clean"]


def test_problems_stage_needs_the_reconciled_review(tmp_path):
    with pytest.raises(RuntimeError, match="reviewed"):
        hd.reviewed_clauses(tmp_path / "missing.json")
    (tmp_path / "r.json").write_text('{"clauses": {"4": {"action": "keep", "clause_template": "A {d}."},'
                                     ' "5": {"action": "drop", "clause_template": null},'
                                     ' "6": {"action": "rewrite", "clause_template": "B {d}."}}}')
    assert hd.reviewed_clauses(tmp_path / "r.json") == {4: "A {d}.", 6: "B {d}."}


def _ra():
    spec = importlib.util.spec_from_file_location("review_agreement", REPO_ROOT / "scripts" / "review_agreement.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_review_labels_and_kappa():
    ra = _ra()
    rows = []
    for t, (n1, g1, n2, g2, v2) in enumerate([("yes", "yes", "yes", "yes", ""), ("no", "yes", "no", "yes", ""),
                                             ("yes", "yes", "yes", "yes", "FIX"), ("yes", "no", "yes", "yes", "")]):
        rows.append({"template_id": str(t), "reviewer1_is_noop (yes/no)": n1, "reviewer1_grammatical (yes/no)": g1,
                     "reviewer2_is_noop (yes/no)": n2, "reviewer2_grammatical (yes/no)": g2,
                     "reviewer2_verdict": v2, "reviewer1_initials": "AB", "reviewer2_initials": "CD"})
    r1, r2 = ra.reviewer_labels(rows, 1), ra.reviewer_labels(rows, 2)
    assert [r1[t]["verdict"] for t in range(4)] == ["OK", "FIX", "OK", "FIX"]
    assert [r2[t]["verdict"] for t in range(4)] == ["OK", "FIX", "FIX", "OK"]
    ag = ra.agreement(r1, r2)
    assert ag["verdict"]["percent_agreement"] == 0.5 and ag["verdict"]["kappa"] == pytest.approx(0.0)
    assert ag["is_noop"]["kappa"] == pytest.approx(1.0)
    # an unanswered row stops the script
    rows[0]["reviewer2_is_noop (yes/no)"] = ""
    with pytest.raises(SystemExit, match="not complete"):
        ra.reviewer_labels(rows, 2)
    # in the combined sheet, reviewer 2 never falls back to reviewer 1's columns
    del rows[0]["reviewer2_is_noop (yes/no)"]
    with pytest.raises(SystemExit):
        ra.reviewer_labels(rows, 2)
