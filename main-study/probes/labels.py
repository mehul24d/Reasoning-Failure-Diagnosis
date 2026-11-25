"""Per-step automatic labels for E1/E2 (and the repair pilot), from the S2 greedy traces.

Every label comes from parser v4 (parsing.steps.parse_trace, frozen in S2) on the S2 scored
traces; nothing is labelled by hand here. The human check of these labels (S5, kappa) is still
pending, so every result built on them is a DEVELOPMENT result.

Test templates are never returned: `step_rows(..., splits=("train", "dev"))` is the only entry
point and it refuses "test" unless `allow_test=True`, which only the E3/E4 scripts pass after the
test set is unlocked (prereg/UNLOCK_TEST.md must exist; see `test_unlocked`).

One row per parsed step of a trace with a gold program (`gold_ok`) that did not stop at
max_new_tokens:
  trace_id, template_id, split, condition, population, prompt, correct (trace), s0, s0_strict,
  step, n_steps, step_pos ("1".."5" or "6+"), computational, compound, ambiguous, extra_valid,
  op (+ - * /, simple steps only), gold_op (op of the aligned gold step; for an unaligned step,
  of the first gold step not yet matched), distractor_used (distractor condition, simple,
  unambiguous steps only), step_correct (computational steps: error == "none"), error_type
  (none | planning | binding | arithmetic, computational steps), a_slot / b_slot (operand slot
  index or None), result (stated result as a string), content_attributable (s0 step only: the
  distractor is bound there, or its op differs from gold_op).

The 2x2 core variables (prereg/E1.md):
  content/binary      distractor_used      (distractor-condition traces)
  content/4-class     op                   (+ - * /)
  evaluation/binary   step_correct
  evaluation/4-class  error_type
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from harness.config import REPO_ROOT

S2_ROOT = REPO_ROOT / "data" / "generated" / "kaggle_outputs" / "btp-minor-s2" / "btp"
GEN = S2_ROOT / "data" / "generated" / "gemma-2-2b-it"
PROBLEMS = S2_ROOT / "data" / "s2_gsms_pairs_i8.jsonl"
SCORED = GEN / "s2_traces_scored.jsonl"
POSITIONS = GEN / "s2_positions.jsonl"
TOP2 = GEN / "s2_top2.jsonl"
STORE = REPO_ROOT / "data" / "generated" / "kaggle_outputs" / "btp-minor-s4" / "btp" / "cache" / \
    "store" / "gemma-2-2b-it"
OPS = ("+", "-", "*", "/")
ERROR_TYPES = ("none", "planning", "binding", "arithmetic")
UNLOCK = REPO_ROOT / "prereg" / "UNLOCK_TEST.md"


def test_unlocked() -> bool:
    """The held-out test templates may be analysed only after the PI's written unlock."""
    return UNLOCK.exists()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_inputs() -> dict:
    """The local S2 copy must hash to the committed S2 records (reports/m1/s2/)."""
    rec = REPO_ROOT / "reports" / "m1" / "s2"
    score = json.loads((rec / "score.json").read_text())
    probs = json.loads((rec / "problems.json").read_text())
    got = {"scored": _sha(SCORED), "positions": _sha(POSITIONS), "problems": _sha(PROBLEMS)}
    want = {"scored": score["sha256_scored"], "positions": score["sha256_positions"],
            "problems": probs["sha256"]}
    bad = [k for k in got if got[k] != want[k]]
    if bad:
        raise SystemExit(f"S2 inputs do not match the committed records: {bad}")
    return got


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def step_pos(i: int) -> str:
    return str(i + 1) if i < 5 else "6+"


def _next_unmatched(matched: set[int], n_gold: int) -> int | None:
    return next((k for k in range(n_gold) if k not in matched), None)


def label_trace(rec: dict, prob: dict) -> list[dict]:
    from parsing.gold import gold_program
    from parsing.steps import parse_trace

    span = prob.get("distractor_span")
    prog, sl = gold_program(prob["question"], prob["solution"], prob["answer"], prob["source"],
                            distractor_span=tuple(span) if span else None)
    if not prog.ok:
        return []
    pt = parse_trace(rec["text"], rec["prompt"].split("_")[0], sl, prog)
    distr = {s.index for s in sl if s.distractor}
    rows, matched = [], set()
    for st in pt.steps:
        comp = st.computational
        simple = comp and not st.compound
        gk = st.gold if st.gold is not None and st.tier is not None and st.tier <= 3 else None
        ref_k = st.gold if st.gold is not None else _next_unmatched(matched, len(prog.steps))
        gold_op = prog.steps[ref_k].op if ref_k is not None else None
        a_slot = st.a_ref[1] if simple and st.a_ref and st.a_ref[0] == "slot" else None
        b_slot = st.b_ref[1] if simple and st.b_ref and st.b_ref[0] == "slot" else None
        du = None
        if rec["condition"] == "distractor" and simple and not st.ambiguous:
            du = bool(distr & {a_slot, b_slot} - {None})
        row = {
            "trace_id": rec["trace_id"], "template_id": rec["template_id"],
            "split": rec["split"], "condition": rec["condition"],
            "population": rec.get("population"), "prompt": rec["prompt"],
            "correct": bool(rec["correct"]), "s0": pt.first_error, "s0_strict": pt.first_error_strict,
            "step": st.index, "n_steps": len(pt.steps), "step_pos": step_pos(st.index),
            "computational": comp, "compound": bool(st.compound), "ambiguous": bool(st.ambiguous),
            "extra_valid": bool(st.extra_valid),
            "op": st.op if simple and st.op in OPS else None,
            "gold_op": gold_op if gold_op in OPS else None,
            "distractor_used": du,
            "step_correct": None if not comp else st.error == "none",
            "error_type": st.error if comp and st.error in ERROR_TYPES else None,
            "a_slot": a_slot, "b_slot": b_slot,
            "result": None if st.result is None else str(st.result),
            "gold_step": st.gold, "tier": st.tier,
        }
        rows.append(row)
        if gk is not None:
            matched.add(gk)
    s0 = pt.first_error
    for r in rows:
        r["is_s0"] = s0 is not None and r["step"] == s0
        r["content_attributable"] = None
        if r["is_s0"]:
            r["content_attributable"] = bool(r["distractor_used"]) or (
                r["op"] is not None and r["gold_op"] is not None and r["op"] != r["gold_op"])
    return rows


@lru_cache(maxsize=4)
def _traces(splits: tuple) -> tuple:
    check_inputs()
    probs = {p["problem_id"]: p for p in read_jsonl(PROBLEMS)}
    keep = []
    for r in read_jsonl(SCORED):
        if r.get("split") not in splits:
            continue                                  # test rows are dropped before parsing
        if not r.get("gold_ok") or r.get("stop_reason") == "max_new_tokens":
            continue
        keep.append((r, probs[r["problem_id"]]))
    return tuple(keep)


def traces(splits=("train", "dev"), allow_test: bool = False) -> list[dict]:
    """Scored S2 records (gold_ok, not truncated) of the given splits."""
    _guard(splits, allow_test)
    return [r for r, _ in _traces(tuple(sorted(splits)))]


def step_rows(splits=("train", "dev"), allow_test: bool = False) -> list[dict]:
    _guard(splits, allow_test)
    out = []
    for r, p in _traces(tuple(sorted(splits))):
        out.extend(label_trace(r, p))
    return out


def _guard(splits, allow_test: bool) -> None:
    if "test" in splits and not (allow_test and test_unlocked()):
        raise PermissionError("test templates are locked (prereg/UNLOCK_TEST.md is absent)")


def positions_by_trace(trace_ids: set[str]) -> dict[str, dict]:
    out = {}
    with POSITIONS.open() as f:
        for line in f:
            d = json.loads(line)
            if d["trace_id"] in trace_ids:
                out[d["trace_id"]] = d
    return out
