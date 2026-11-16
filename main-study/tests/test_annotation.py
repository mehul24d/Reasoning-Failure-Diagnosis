"""S5 annotation sheets (scripts/build_annotation_sheets.py) and agreement (scripts/annotation_agreement.py)."""
import csv
import importlib.util
import json
from collections import Counter

import pytest

from analysis.stats import cohens_kappa
from harness.config import REPO_ROOT


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pool():
    rows, i = [], 0
    for split, sizes in (("test", (7, 3, 11, 1)), ("train", (5, 9)), ("dev", (4,))):
        for t, n in enumerate(sizes):
            for pop in ("primary", "natural"):
                for _ in range(n if pop == "natural" else max(n // 2, 1)):
                    rows.append({"trace_id": f"x{i}", "split": split, "template_id": f"{split}-{t}", "population": pop})
                    i += 1
    return rows


def test_draw_is_stratified_fixed_and_twenty_percent_per_split():
    b = _mod("build_annotation_sheets")
    pool = _pool()
    s1, s2 = b.draw(pool, 0.2, 7), b.draw(pool, 0.2, 7)
    assert [r["trace_id"] for r in s1] == [r["trace_id"] for r in s2]
    n_split, got = Counter(r["split"] for r in pool), Counter(r["split"] for r in s1)
    assert all(got[k] == round(0.2 * n) for k, n in n_split.items())
    strata = Counter((r["split"], r["template_id"], r["population"]) for r in pool)
    took = Counter((r["split"], r["template_id"], r["population"]) for r in s1)
    assert all(int(0.2 * n) <= took[k] <= int(0.2 * n) + 1 for k, n in strata.items())
    assert len({r["trace_id"] for r in s1}) == len(s1)


def test_model_solution_is_numbered_and_complete():
    b = _mod("build_annotation_sheets")
    marked = {"trace_id": "m", "prompt": "marked_p0",
              "text": "Sure.\nStep 1: 2 + 3 = 5\nStep 2: 5 * 4 = 20\nAnswer: 20"}
    out = b.render_model(marked)
    assert out.splitlines() == ["[before step 1] Sure.", "[1] Step 1: 2 + 3 = 5", "[2] Step 2: 5 * 4 = 20",
                                "[answer line] Answer: 20"]
    terse = {"trace_id": "t", "prompt": "terse_p1", "text": "Cost: 3 * 4 = 12\n\nLeft: 20 - 12 = 8\nAnswer: 8"}
    assert b.render_model(terse).splitlines()[:2] == ["[1] Cost: 3 * 4 = 12", "[2] Left: 20 - 12 = 8"]
    assert b.render_gold("A is 2 + 3 = 5.\nB is 5 * 4 = 20.\n#### 20") == "(1) A is 2 + 3 = 5.\n(2) B is 5 * 4 = 20."


def test_sheet_has_no_model_or_parser_labels():
    b = _mod("build_annotation_sheets")
    rec = {"trace_id": "m", "prompt": "marked_p0", "text": "Step 1: 2 + 3 = 6\nAnswer: 6", "pred_answer": "6",
           "answer_extracted": True, "s0": 0, "s0_type": "arithmetic", "correct": False, "population": "natural"}
    row = b.sheet_row("F0001", rec, {"question": "What is 2 + 3?", "solution": "2 + 3 = 5\n#### 5", "answer": "5"})
    assert list(row) == b.COLUMNS
    assert all(row[c] == "" for c in b.LABEL_COLS)
    text = " ".join(row.values()).lower()
    assert "arithmetic" not in text and "natural" not in text and "s0" not in text


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "model_solution", "first_wrong_step", "error_type", "confidence", "notes"])
        w.writeheader()
        w.writerows(rows)


def _setup(tmp_path, la, lb):
    sol = "[1] a\n[2] b\n[3] c\n[answer line] Answer: 1"
    items = [{"item_id": "P01", "template_id": "gsms-1", "part": 0}]
    items += [{"item_id": f"F{i:04d}", "template_id": f"gsms-{i % 4}", "part": 1 if i < 6 else 2} for i in range(len(la))]
    rec = tmp_path / "sample.json"
    rec.write_text(json.dumps({"items": items}))
    for name, lab in (("A", la), ("B", lb)):
        rows = [{"item_id": "P01", "model_solution": sol, "first_wrong_step": "", "error_type": "", "confidence": ""}]
        rows += [{"item_id": f"F{i:04d}", "model_solution": sol, "first_wrong_step": s, "error_type": t, "confidence": "2"}
                 for i, (s, t) in enumerate(lab)]
        _write(tmp_path / f"{name}.csv", rows)
    return rec


LA = [("1", "planning"), ("2", "binding"), ("none", ""), ("3", "arithmetic"), ("answer", "other"),
      ("2", "binding"), ("1", "planning"), ("3", "arithmetic"), ("2", "planning"), ("none", "")]
LB = [("1", "planning"), ("Step 2", "arithmetic"), ("none", ""), ("[3]", "arithmetic"), ("answer", "other"),
      ("1", "binding"), ("1", "planning"), ("3", "binding"), ("2", "planning"), ("2", "binding")]


def test_agreement_matches_cohens_kappa_and_skips_practice(tmp_path):
    m = _mod("annotation_agreement")
    rec = _setup(tmp_path, LA, LB)
    out = tmp_path / "agr.json"
    assert m.main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--record", str(rec), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["n_items"] == 10                               # P01 (blank) excluded
    na = [m.norm_step(s, 3) for s, _ in LA]
    nb = [m.norm_step(s, 3) for s, _ in LB]
    assert r["first_wrong_step"]["kappa"]["estimate"] == pytest.approx(cohens_kappa(na, nb))
    assert r["first_wrong_step"]["raw"]["estimate"] == pytest.approx(8 / 10)
    k = r["first_wrong_step"]["k_categories"]
    assert k == 5 and r["first_wrong_step"]["pabak"]["estimate"] == pytest.approx((k * 0.8 - 1) / (k - 1))
    same = [i for i in range(10) if na[i] == nb[i] and na[i] != "none"]
    assert r["error_type_on_same_step"]["n"] == len(same) == 7
    ta, tb = [LA[i][1] for i in same], [LB[i][1] for i in same]
    assert r["error_type_on_same_step"]["kappa"]["estimate"] == pytest.approx(cohens_kappa(ta, tb))
    assert r["first_wrong_step"]["kappa"]["ci_low"] <= r["first_wrong_step"]["kappa"]["estimate"] <= r["first_wrong_step"]["kappa"]["ci_high"]
    assert r["kappa_min"] == 0.6


def test_agreement_refuses_incomplete_or_invalid_sheets(tmp_path):
    m = _mod("annotation_agreement")
    for bad in (("", ""), ("4", "planning"), ("2", "misc"), ("2", "")):
        rec = _setup(tmp_path, LA, [bad] + LB[1:])
        with pytest.raises(SystemExit):
            m.main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--record", str(rec), "--out", str(tmp_path / "o.json")])
    # part 1 only needs part 1 complete
    rec = _setup(tmp_path, LA, LB[:9] + [("", "")])
    assert m.main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--record", str(rec), "--part", "1",
                   "--out", str(tmp_path / "o.json")]) == 0
