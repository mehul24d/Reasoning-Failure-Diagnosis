"""The automated distractor-clause checks can fail (DECISIONS D34)."""
import importlib.util
import sys

import pytest

from harness.config import REPO_ROOT

pytest.importorskip("numpy")
sys.path.insert(0, str(REPO_ROOT.parent / "src"))
spec = importlib.util.spec_from_file_location("distractor_review", REPO_ROOT / "scripts" / "distractor_review.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)

ROW = {"question": "Liam buys 4 boxes of pencils. Each box holds 13 pencils. How many pencils does he buy?",
       "solution": "He buys 4 * 13 = 52 pencils.\n#### 52", "answer": "52"}


def run(clause, monkeypatch):
    monkeypatch.setitem(dr.NOOP_CLAUSES, 999, clause)
    return dr.check_instance(999, ROW, 0)["checks"]


def test_good_clause_passes(monkeypatch):
    assert all(run("{d} of the boxes are blue.", monkeypatch).values())


def test_bad_clauses_fail(monkeypatch):
    assert not run("Some of the boxes are blue.", monkeypatch)["contains_number"]
    assert not run("{d} of the boxes hold 13 pencils.", monkeypatch)["contains_number"]
    assert not run("{d} of the boxes are blue. Some are red.", monkeypatch)["clause_one_sentence"]
    assert not run("{d} of the boxes are blue", monkeypatch)["clause_one_sentence"]


def test_final_sentence_rule():
    assert dr._final_sentence_ok("How much more is it than Mrs. Lee's budget?")
    assert dr._final_sentence_ok("If there were 25, calculate the total number of insects.")
    assert not dr._final_sentence_ok("He is happy. How many does he have?")
    assert not dr._final_sentence_ok("He has many pencils.")
