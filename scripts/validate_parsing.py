"""Check the gold extractor and the step parser on real data (no GPU needed).

1. Gold programs for every GSM8K train row and every apple/GSM-Symbolic row (main, p1, p2):
   share that are `ok`, and the issue categories of the rest.
2. Step parsing of the S1c traces (GSM8K train, gemma-2-2b-it) against the GSM8K gold
   programs: steps per trace, computational / compound / aligned shares, error types split by
   trace correctness. A correct trace should rarely have an error step; a failed one should
   usually have one. Also writes 40 random parsed steps for checking by hand.

Usage:
    python scripts/validate_parsing.py --traces data/generated/gemma-2-2b-it/m0_traces_scored.jsonl \
        --out reports/m0/parsing_validation.json --sample-out reports/m0/parsing_sample.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parsing.answers import gsm8k_gold  # noqa: E402
from parsing.gold import gold_program  # noqa: E402
from parsing.steps import parse_trace  # noqa: E402


def issue_kind(issue: str) -> str:
    return re.sub(r"[-\d./]+", "N", issue.split(":")[0])


def gold_report(rows, source) -> dict:
    c, kinds, n_steps = Counter(), Counter(), []
    for q, sol in rows:
        prog, _ = gold_program(q, sol, gsm8k_gold(sol), source)
        c["n"] += 1
        c["ok"] += prog.ok
        c["ambiguous_steps"] += sum(g.ambiguous for g in prog.steps)
        c["const_operands"] += sum((g.a_ref[0] == "const") + (g.b_ref[0] == "const")
                                   for g in prog.steps)
        c["steps"] += len(prog.steps)
        if prog.ok:
            n_steps.append(len(prog.steps))
        for i in prog.issues:
            kinds[issue_kind(i)] += 1
    return {"n": c["n"], "ok_rate": c["ok"] / max(c["n"], 1),
            "mean_steps_ok": sum(n_steps) / max(len(n_steps), 1),
            "ambiguous_step_rate": c["ambiguous_steps"] / max(c["steps"], 1),
            "const_operand_rate": c["const_operands"] / max(2 * c["steps"], 1),
            "issues": dict(kinds.most_common())}


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed of the hand-check sample")
    a = ap.parse_args()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    report = {"gold": {"gsm8k_train": gold_report([(r["question"], r["answer"]) for r in gsm],
                                                  "gsm8k")}}
    for cfg in ("main", "p1", "p2"):
        ds = load_dataset("apple/GSM-Symbolic", cfg, split="test")
        report["gold"][f"gsm_symbolic_{cfg}"] = gold_report(
            [(r["question"], r["answer"]) for r in ds], "gsm_symbolic")
        report["gold"][f"gsm_symbolic_{cfg}"]["n_templates"] = len(set(ds["id"]))

    traces = [json.loads(line) for line in open(a.traces) if line.strip()]
    stats = {}
    sample_pool = []
    for r in traces:
        idx = int(r["problem_id"].rsplit("-", 1)[1])
        row = gsm[idx]
        prog, sl = gold_program(row["question"], row["answer"], gsm8k_gold(row["answer"]), "gsm8k")
        style = r["prompt"].split("_")[0]
        key = (style, "correct" if r["correct"] else "failed")
        s = stats.setdefault(key, Counter())
        s["traces"] += 1
        if not prog.ok:
            s["gold_not_ok"] += 1
            continue
        p = parse_trace(r["text"], style, sl, prog)
        s["traces_parsed"] += 1
        s["steps"] += len(p.steps)
        s["computational"] += len(p.computational)
        s["compound"] += sum(st.compound for st in p.steps)
        s["aligned"] += sum(st.gold is not None for st in p.steps)
        s["ambiguous"] += sum(st.ambiguous for st in p.steps)
        s["no_computational_steps"] += not p.computational
        s["has_error_step"] += p.first_error is not None
        s["has_error_step_strict"] += p.first_error_strict is not None
        s["extra_valid"] += sum(st.extra_valid for st in p.steps)
        for st in p.steps:
            if st.tier is not None:
                s[f"tier_{st.tier}"] += 1
        s["gold_steps"] += len(prog.steps)
        s["gold_unmatched"] += p.n_unmatched_gold
        for st in p.steps:
            if st.error is not None:
                s[f"err_{st.error}"] += 1
            if st.computational:
                sample_pool.append((r["problem_id"], r["prompt"], r["correct"], st, prog))

    out = {}
    for (style, grp), s in sorted(stats.items()):
        n = max(s["traces_parsed"], 1)
        typed = sum(v for k, v in s.items() if k.startswith("err_"))
        out[f"{style}/{grp}"] = {
            **dict(s),
            "steps_per_trace": s["steps"] / n,
            "computational_share": s["computational"] / max(s["steps"], 1),
            "compound_share_of_computational": s["compound"] / max(s["computational"], 1),
            "aligned_share_of_computational": s["aligned"] / max(s["computational"], 1),
            "gold_steps_matched_share": 1 - s["gold_unmatched"] / max(s["gold_steps"], 1),
            "trace_has_error_step": s["has_error_step"] / n,
            "trace_has_error_step_strict": s["has_error_step_strict"] / n,
            "error_type_shares": {k[4:]: v / max(typed, 1) for k, v in s.items()
                                  if k.startswith("err_")},
        }
    report["s1c_traces"] = out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))

    rng = random.Random(a.seed)
    lines = ["# Random parsed steps (S1c traces) for manual checking", ""]
    for pid, prompt, ok, st, prog in rng.sample(sample_pool, min(40, len(sample_pool))):
        g = prog.steps[st.gold] if st.gold is not None else None
        lines += [f"## {pid} {prompt} trace_correct={ok}",
                  f"* step text: `{st.text.strip()[:200]}`",
                  f"* parsed: op={st.op} a={st.a} b={st.b} result={st.result} "
                  f"a_ref={st.a_ref} b_ref={st.b_ref} compound={st.compound} "
                  f"ambiguous={st.ambiguous}",
                  f"* aligned gold step: {st.gold} (tier {st.tier}) "
                  + (f"({g.op}, {g.a_ref}, {g.b_ref}, {g.result})" if g else ""),
                  f"* error type: {st.error}  extra_valid={st.extra_valid}",
                  "* gold program: " + "; ".join(f"{k}: {x.a} {x.op} {x.b} = {x.result}"
                                                 for k, x in enumerate(prog.steps)),
                  "* manual check: [ ] correct  [ ] wrong (note):", ""]
    Path(a.sample_out).write_text("\n".join(lines))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
