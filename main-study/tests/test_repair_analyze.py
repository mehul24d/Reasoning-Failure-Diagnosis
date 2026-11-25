"""Calibration rule, random-null p-value and the verdict words of prereg/R1.md."""
from __future__ import annotations

import numpy as np

from repair.analyze import GRID, decide, summarize, verdict


def _cur(rates):
    return {str(m): {"oracle": {"failure": {"answer_changed": c, "broken": b}}}
            for m, (c, b) in zip(GRID, rates)}


def test_decide_smallest_multiple_meeting_the_criterion():
    r = decide(_cur([(0, 0), (0.1, 0), (0.2, 0.05), (0.5, 0.0), (0.9, 0.5), (1, 1), (1, 1)]))
    assert r["m_star"] == 1.0 and len(r["trail"]) == 3


def test_decide_stops_when_outputs_break_first():
    r = decide(_cur([(0, 0), (0.05, 0.2), (0.5, 0.0), (1, 0), (1, 0), (1, 1), (1, 1)]))
    assert r["m_star"] is None and "break first" in r["reason"]


def test_decide_change_with_too_much_breakage_is_not_accepted():
    r = decide(_cur([(0, 0), (0.3, 0.3)] + [(1, 1)] * 5))
    assert r["m_star"] is None


def test_decide_no_change_on_the_grid():
    r = decide(_cur([(0.0, 0.0)] * 7))
    assert r["m_star"] is None and "no multiple" in r["reason"]


def _items(n_t=6):
    items = {}
    k = 0
    for t in range(n_t):
        for role in ("failure", "solved"):
            items[k] = {"item": k, "template_id": f"t{t}", "role": role}
            k += 1
    return items


def test_summarize_rates_and_p_value():
    items = _items()
    rows = []
    for i, it in items.items():
        f = it["role"] == "failure"
        rows.append({"item": i, "cond": "oracle", "correct": f or True, "answer_changed": f, "broken": False})
        rows.append({"item": i, "cond": "eval", "correct": not f, "answer_changed": False, "broken": False})
        for j in range(100):
            rows.append({"item": i, "cond": f"rand{j}", "correct": not f, "answer_changed": False, "broken": False})
    s = summarize(rows, items, n_boot=200)
    o = s["conditions"]["oracle"]
    assert o["repair"]["estimate"] == 1.0 and o["damage"]["estimate"] == 0.0
    assert np.isclose(o["net"]["estimate"], 0.5)
    assert o["p_vs_random"] == 1 / 101
    assert s["conditions"]["eval"]["p_vs_random"] == 1.0
    assert verdict(o, True) == "supported"
    assert verdict(s["conditions"]["eval"], True) == "not supported"
