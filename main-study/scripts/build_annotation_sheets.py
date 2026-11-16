"""Build the two-annotator failure-annotation sheets for S5 (docs/annotation/).

Population ("study failures"): every S2 greedy trace in a failure population (A14: primary =
distractor failure whose clean twin is right; natural = clean-condition failure), with a gold
program (`gold_ok`) and not stopped by max_new_tokens (those are excluded from the failure sets,
prereg/S2.md acceptance table). Failures without a parser s0 are kept: the human labels must
not depend on the parser.

Sample: 20% (experiment.yaml annotation.fraction) of that population, stratified by split x
template x population. Each stratum gets floor(0.2 n) items; the remaining items of each split
go to the strata with the largest remainders (ties broken by the seeded RNG), so every split
gets round(0.2 N_split). Items are drawn within a stratum by the seeded RNG (SEED below).

Sheets: annotator_A.csv and annotator_B.csv hold the same items. Practice items P01-P10 come
first, in the same order in both sheets (they are discussed together). Then part 1 (test split)
and part 2 (train and dev), each in a different random order per annotator. The sheets carry no
parser output, model scores, population, condition or split: only the problem text, the
model's solution with numbered steps, the gold solution steps and both answers.

Outputs:
  docs/annotation/annotator_A.csv, annotator_B.csv   (UTF-8, every field quoted)
  reports/m1/s5_annotation/sample.json               (seed, counts, item -> trace map, sheet sha256)
  and it reads docs/annotation/private/practice_key.csv (gitignored, never written by this script;
  its sha256 goes into sample.json)

    python -I scripts/build_annotation_sheets.py --root data/generated/kaggle_outputs/btp-minor-s2/btp

The scored traces are checked against the committed S2 record (reports/m1/s2/score.json) first.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # works under `python -I`

import numpy as np                                                # noqa: E402

from harness.config import REPO_ROOT, experiment_config           # noqa: E402
from harness.pipeline import Run                                  # noqa: E402
from parsing.steps import segment                                 # noqa: E402

MODEL = "gemma-2-2b-it"
SEED = 20261116                    # fixed before drawing; the date the sheets were built
ANNOT = REPO_ROOT.parent / "docs" / "annotation"
RECORD = REPO_ROOT / "reports" / "m1" / "s5_annotation" / "sample.json"
PRIVATE_KEY = ANNOT / "private" / "practice_key.csv"

INPUT_COLS = ["item_id", "problem", "model_solution", "gold_solution", "gold_answer", "model_final_answer"]
LABEL_COLS = ["first_wrong_step", "error_type", "confidence", "notes"]
COLUMNS = INPUT_COLS + LABEL_COLS
ERROR_TYPES = ("planning", "binding", "arithmetic", "other")

# Practice items P01-P10, in this order: chosen by hand from eligible failures OUTSIDE the
# sample (train split) so that every error type occurs at least twice. Their answer key is NOT in
# this repository: it lives in the gitignored PRIVATE_KEY file (item_id, trace_id,
# first_wrong_step, error_type, why), held by the coordinator until the practice discussion is
# over. The build checks the key against this list and records its sha256.
PRACTICE: list[str] = [
    "gsms-main-54-3|marked_p0|0",
    "gsms-noop-20-7|marked_p2|0",
    "gsms-main-81-4|marked_p2|0",
    "gsms-main-83-3|terse_p1|0",
    "gsms-noop-8-2|marked_p2|0",
    "gsms-main-81-3|marked_p2|0",
    "gsms-main-55-2|marked_p0|0",
    "gsms-noop-13-6|terse_p1|0",
    "gsms-main-89-6|marked_p2|0",
    "gsms-noop-98-2|terse_p0|0",
]

# Worked examples printed in ANNOTATION_GUIDE.md (also outside the sample and the practice set).
WORKED: list[str] = [
    "gsms-main-41-0|terse_p1|0", "gsms-noop-64-4|terse_p0|0",      # arithmetic
    "gsms-noop-46-4|marked_p0|0", "gsms-main-37-7|marked_p1|0",    # binding
    "gsms-main-54-5|marked_p0|0", "gsms-main-54-2|marked_p2|0",    # planning
    "gsms-main-77-7|marked_p0|0", "gsms-main-80-4|marked_p0|0",    # other
    "gsms-main-81-3|terse_p0|0",                                   # answer line, planning (stops early)
]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(root: Path) -> tuple[list[dict], dict]:
    run = Run.study(MODEL, root=root)
    scored = run.base / "s2_traces_scored.jsonl"
    rec = json.loads((REPO_ROOT / "reports" / "m1" / "s2" / "score.json").read_text())
    if sha(scored) != rec["sha256_scored"]:
        raise SystemExit(f"{scored} does not hash to the committed S2 record; refusing to build")
    prec = json.loads((REPO_ROOT / "reports" / "m1" / "s2" / "problems.json").read_text())
    if sha(run.problems) != prec["sha256"]:
        raise SystemExit(f"{run.problems} does not hash to the committed S2 record; refusing to build")
    return read_jsonl(scored), {p["problem_id"]: p for p in read_jsonl(run.problems)}


def eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("population") in ("primary", "natural")
            and r["gold_ok"] and not r["hit_max_tokens"]]


def draw(pool: list[dict], frac: float, seed: int) -> list[dict]:
    """Stratified sample: split x template x population, largest remainder within each split."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["trace_id"]):
        strata[(r["split"], r["template_id"], r["population"])].append(r)
    take = {k: int(np.floor(frac * len(v))) for k, v in strata.items()}
    for split in sorted({k[0] for k in strata}):
        keys = sorted(k for k in strata if k[0] == split)
        n_split = sum(len(strata[k]) for k in keys)
        short = int(round(frac * n_split)) - sum(take[k] for k in keys)
        tie = rng.random(len(keys))
        order = sorted(range(len(keys)), key=lambda i: (-(frac * len(strata[keys[i]]) - take[keys[i]]), tie[i]))
        for i in order[:short]:
            take[keys[i]] += 1
    out = []
    for k in sorted(strata):
        v = strata[k]
        idx = rng.choice(len(v), size=take[k], replace=False) if take[k] else []
        out += [v[i] for i in sorted(idx)]
    return out


def answer_stop(text: str) -> int:
    from parsing.steps import _ANSWER_LINE
    ans = list(_ANSWER_LINE.finditer(text))
    return ans[-1].start() if ans else len(text)


def model_steps(rec: dict) -> tuple[list[str], str, str]:
    """(numbered steps, text before step 1, the answer line and anything after it).

    The steps are the parser's step segmentation (parsing.steps.segment): "Step k:" blocks for
    the marked prompts, one per line (one per equation on a line with several) for the terse
    prompts. Step [n] is parser step n-1. Only the segmentation is used, nothing the parser
    derives from it."""
    text = rec["text"]
    spans = segment(text, rec["prompt"].split("_")[0])
    steps = [text[s:e].strip("\n") for s, e in spans]
    before = text[:spans[0][0]].strip() if spans else ""
    stop = answer_stop(text)
    after = text[stop:].strip() if stop < len(text) else ""
    if not spans:
        before = text[:stop].strip()
    return steps, before, after


def render_model(rec: dict) -> str:
    steps, before, after = model_steps(rec)
    parts = []
    if before:
        parts.append("[before step 1] " + before)
    parts += [f"[{i}] {s}" for i, s in enumerate(steps, 1)]
    parts.append("[answer line] " + after if after else "[answer line] (no separate answer line)")
    out = "\n".join(parts)
    squash = lambda x: re.sub(r"\s+", "", x)
    if squash(rec["text"]) not in squash(re.sub(r"\[(before step 1|answer line|\d+)\] ", "", out)):
        raise SystemExit(f"{rec['trace_id']}: the numbered solution does not reproduce the model's text")
    return out


def gold_steps(solution: str) -> list[str]:
    lines = [ln.strip() for ln in solution.split("\n")]
    return [ln for ln in lines if ln and not ln.startswith("####")]


def render_gold(solution: str) -> str:
    return "\n".join(f"({i}) {s}" for i, s in enumerate(gold_steps(solution), 1))


# Planning estimate of annotation time, before any timing exists: careful reading of worked maths
# at ~120 words a minute (problem + gold solution + model solution), 0.15 min to check each
# numbered model step, 0.75 min to fill the four label columns. Replace it with the practice
# session's measured minutes per item.
READ_WPM, PER_STEP_MIN, FIXED_MIN = 120, 0.15, 0.75


def minutes_estimate(rec: dict, prob: dict) -> float:
    words = len(prob["question"].split()) + len(" ".join(gold_steps(prob["solution"])).split()) + len(rec["text"].split())
    return FIXED_MIN + words / READ_WPM + PER_STEP_MIN * len(model_steps(rec)[0])


def sheet_row(item_id: str, rec: dict, prob: dict) -> dict:
    pred = rec["pred_answer"] if rec["answer_extracted"] else "(no answer found)"
    return {"item_id": item_id, "problem": prob["question"], "model_solution": render_model(rec),
            "gold_solution": render_gold(prob["solution"]), "gold_answer": prob["answer"],
            "model_final_answer": pred, **{c: "" for c in LABEL_COLS}}


def to_csv(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


_FORMULA = re.compile(r"^[=+\-@]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding data/ (e.g. the S2 Kaggle output's btp/)")
    ap.add_argument("--out", default=str(ANNOT))
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--key", default=str(PRIVATE_KEY))
    a = ap.parse_args(argv)
    frac = experiment_config()["annotation"]["fraction"]
    rows, probs = load(Path(a.root))
    by_id = {r["trace_id"]: r for r in rows}
    pool = eligible(rows)
    sample = draw(pool, frac, SEED)
    in_sample = {r["trace_id"] for r in sample}
    practice_ids = list(PRACTICE)
    for t in practice_ids + WORKED:
        r = by_id[t]
        if t in in_sample or r not in pool or (t in practice_ids and r["split"] != "train"):
            raise SystemExit(f"{t}: practice items must be eligible train failures outside the sample, "
                             "worked examples eligible failures outside the sample")
    if len(set(practice_ids + WORKED)) != len(practice_ids + WORKED):
        raise SystemExit("practice and worked items overlap")

    rng = np.random.default_rng(SEED + 1)
    ids = rng.permutation(len(sample)) + 1                       # opaque item ids
    items = [{"item_id": f"F{ids[i]:04d}", "trace_id": r["trace_id"], "template_id": r["template_id"],
              "split": r["split"], "population": r["population"], "condition": r["condition"],
              "part": 1 if r["split"] == "test" else 2} for i, r in enumerate(sample)]
    practice = [{"item_id": f"P{i:02d}", "trace_id": t, "template_id": by_id[t]["template_id"],
                 "split": "train", "population": by_id[t]["population"],
                 "condition": by_id[t]["condition"], "part": 0} for i, t in enumerate(practice_ids, 1)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for who, seed in (("A", SEED + 2), ("B", SEED + 3)):
        r_ = np.random.default_rng(seed)
        order = list(practice)
        for part in (1, 2):
            block = sorted((it for it in items if it["part"] == part), key=lambda it: it["item_id"])
            order += [block[i] for i in r_.permutation(len(block))]
        sheet = [sheet_row(it["item_id"], by_id[it["trace_id"]], probs[by_id[it["trace_id"]]["problem_id"]])
                 for it in order]
        for row in sheet:
            for c in INPUT_COLS:
                if _FORMULA.match(row[c]):
                    row[c] = " " + row[c]                       # never read as a formula
        f = out / f"annotator_{who}.csv"
        f.write_text(to_csv(sheet, COLUMNS), encoding="utf-8")
        sheets[who] = {"file": str(f.relative_to(REPO_ROOT.parent)), "sha256": sha(f),
                       "order": [it["item_id"] for it in order]}
    if sheets["A"]["order"][len(practice):] == sheets["B"]["order"][len(practice):]:
        raise SystemExit("the two sheets came out in the same order")

    key = Path(a.key)
    if not key.exists():
        raise SystemExit(f"{key}: the practice answer key is missing (it is kept out of git; ask the coordinator)")
    with open(key, encoding="utf-8", newline="") as f:
        krows = list(csv.DictReader(f))
    if [(r["item_id"], r["trace_id"]) for r in krows] != [(p["item_id"], p["trace_id"]) for p in practice]:
        raise SystemExit(f"{key} does not list the practice items P01-P10 in order")
    if any(r["first_wrong_step"] != "none" and r["error_type"] not in ERROR_TYPES for r in krows):
        raise SystemExit(f"{key}: an error_type is not one of {ERROR_TYPES}")
    (key.parent / ".gitignore").write_text("*\n")

    def counts(xs):
        c = defaultdict(int)
        for it in xs:
            c[f"{it['split']}/{it['population']}"] += 1
        return dict(sorted(c.items()))
    words = [len(by_id[it["trace_id"]]["text"].split()) for it in items]
    mins = {it["item_id"]: minutes_estimate(by_id[it["trace_id"]], probs[by_id[it["trace_id"]]["problem_id"]])
            for it in practice + items}
    est = lambda xs: round(sum(mins[it["item_id"]] for it in xs), 1)
    record = {
        "note": "S5 failure-annotation sample (scripts/build_annotation_sheets.py). The sheets show no "
                "parser output; do not read the item -> trace map while annotating.",
        "seed": SEED, "fraction": frac,
        "population_rule": "population in {primary, natural}, gold_ok, not hit_max_tokens",
        "strata": "split x template x population; largest remainder within split",
        "scored_sha256": sha(Path(a.root) / "data" / "generated" / MODEL / "s2_traces_scored.jsonl"),
        "eligible": {"n": len(pool), "by": counts(pool)},
        "sample": {"n": len(items), "part1_test": sum(it["part"] == 1 for it in items),
                   "part2_train_dev": sum(it["part"] == 2 for it in items), "by": counts(items),
                   "templates": len({it["template_id"] for it in items}),
                   "words_per_trace_median": float(np.median(words)) if words else None},
        "time_estimate": {
            "rule": f"{FIXED_MIN} min + words/{READ_WPM} wpm + {PER_STEP_MIN} min per numbered model step "
                    "(problem, gold and model words); a planning figure until the practice session is timed",
            "minutes_per_item_median": round(float(np.median([mins[it["item_id"]] for it in items])), 2),
            "minutes_per_item_mean": round(float(np.mean([mins[it["item_id"]] for it in items])), 2),
            "hours_per_annotator": {"practice": round(est(practice) / 60, 1),
                                    "part1_test": round(est([it for it in items if it["part"] == 1]) / 60, 1),
                                    "part2_train_dev": round(est([it for it in items if it["part"] == 2]) / 60, 1),
                                    "all": round(est(practice + items) / 60, 1)}},
        "practice": {"n": len(practice), "excluded_from_kappa": True,
                     "key_sha256": sha(key), "key_file": "docs/annotation/private/practice_key.csv (not committed)"},
        "sheets": {k: {"file": v["file"], "sha256": v["sha256"]} for k, v in sheets.items()},
        "items": practice + items,
    }
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("eligible", "sample", "time_estimate", "practice")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
