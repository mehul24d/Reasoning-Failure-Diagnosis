"""E3/E4 analysis scripts: the test lock, the per-failure statistics and the S5 kappa gate."""
import json

import numpy as np
import pytest

from analysis import e3, e4


def test_e3_refuses_test_while_locked(monkeypatch):
    monkeypatch.setattr(e3, "test_unlocked", lambda: False)
    with pytest.raises(PermissionError):
        e3.guard("test")
    e3.guard("dev")


def test_e4_refuses_test_while_locked_or_without_kappa(monkeypatch):
    monkeypatch.setattr(e4, "test_unlocked", lambda: False)
    with pytest.raises(PermissionError):
        e4.guard("test", {"met": True})
    monkeypatch.setattr(e4, "test_unlocked", lambda: True)
    with pytest.raises(SystemExit):
        e4.guard("test", {"met": False, "status": "pending"})
    e4.guard("dev", {"met": False})


def test_per_failure_top1_gob_and_prior_correction():
    p = np.array([[0.6, 0.4, 0.0, 0.0],      # gold 0 top-1, over bound 1
                  [0.2, 0.5, 0.3, 0.0],      # bound 1 is top-1, gold 2 below it
                  [0.25, 0.25, 0.25, 0.25]])  # tie: gob 1/2
    gold, bound = np.array([0, 2, 1]), np.array([1, 1, 3])
    r = e3.per_failure(p, gold, bound, pi=np.array([0.7, 0.1, 0.1, 0.1]))
    assert r["top1"].tolist() == [1, 0, 0]
    assert r["gob"].tolist() == [1, 0, 0.5]
    assert r["bound_top1"].tolist() == [0, 1, 0]
    # prior-corrected: 0.6/0.7 < 0.4/0.1, so gold no longer wins in row 0
    assert r["gob_prior_corrected"][0] == 0.0


def test_mean_ci_weights_by_row():
    x = np.array([1.0, 0.0, 0.0, 1.0])
    t = np.array(["a", "a", "b", "c"])
    r = e3.mean_ci(x, t)
    assert r["estimate"] == pytest.approx(0.5) and r["n"] == 4


def test_kappa_status(tmp_path):
    assert e4.kappa_status(tmp_path / "missing.json")["met"] is False
    rec = {"first_wrong_step": {"kappa": {"estimate": 0.7}, "meets_kappa_min": True},
           "error_type_on_same_step": {"kappa": {"estimate": 0.5}, "meets_kappa_min": False}}
    (tmp_path / "a.json").write_text(json.dumps(rec))
    assert e4.kappa_status(tmp_path / "a.json")["met"] is False
    rec["error_type_on_same_step"]["meets_kappa_min"] = True
    (tmp_path / "a.json").write_text(json.dumps(rec))
    assert e4.kappa_status(tmp_path / "a.json")["met"] is True
