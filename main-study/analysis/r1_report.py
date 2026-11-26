"""Readable summary of the R1 repair pilot (prereg/R1.md) from its committed JSON results.

    python -m analysis.r1_report [--dir reports/m2/r1]  -> r1.md, calibration.png

CPU only; reads r1_results.json, calibration.json, projection.json, session_status.json and
inputs_meta.json. No number is recomputed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.config import REPO_ROOT

DIR = REPO_ROOT / "reports" / "m2" / "r1"
SETS = ("all (train + dev)", "train", "dev", "content/binary items", "content/4-class items",
        "stable failures", "borderline failures")


def ci(c: dict) -> str:
    return f"{100 * c['estimate']:+.1f} [{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]"


def pct(c: dict) -> str:
    return f"{100 * c['estimate']:.1f} [{100 * c['ci_low']:.1f}, {100 * c['ci_high']:.1f}]"


def plot(cal: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = sorted(float(m) for m in cal["curve"])
    o = [cal["curve"][str(m)]["oracle"]["failure"] for m in ms]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ms, [x["answer_changed"] for x in o], "o-", label="oracle: answer changed")
    ax.plot(ms, [x["broken"] for x in o], "s-", label="oracle: broken")
    ax.plot(ms, [x["correct"] for x in o], "^-", label="oracle: correct")
    rnd = [(m, cal["curve"][str(m)]["random"]["failure"]["answer_changed"]) for m in ms
           if "random" in cal["curve"][str(m)]]
    if rnd:
        ax.plot(*zip(*rnd), "x--", label="random (10 dirs): answer changed")
    ax.axhline(0.20, color="grey", lw=0.6, ls=":")
    ax.axhline(0.10, color="grey", lw=0.6, ls=":")
    ax.axvline(cal["m_star"], color="k", lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("edit strength m (× class-mean distance)")
    ax.set_ylabel("share of 41 dev failures")
    ax.set_title(f"R1 calibration (dev): m* = {cal['m_star']}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def markdown(r: dict, cal: dict, proj: dict, sess: dict, meta: dict) -> str:
    L = ["# R1: redesigned repair pilot (train + dev templates)", "",
         f"**Development result.** Labels: {r['labels']}. No test template is used. "
         f"Code {sess['commit'][:7]}; Kaggle 2 × T4, {sess['session_hours']:.2f} h session = "
         f"**{sess['t4_hours_this_session']:.2f} T4-hours** (cap 8).", "",
         "## Calibration (41 dev failures, oracle edit)", "",
         "| m | answer changed | broken | correct |", "|---|---|---|---|"]
    for m in sorted(cal["curve"], key=float):
        o = cal["curve"][m]["oracle"]["failure"]
        L.append(f"| {m} | {o['answer_changed']:.3f} | {o['broken']:.3f} | {o['correct']:.3f} |")
    L += ["", f"m\\* = **{cal['m_star']}** ({cal['reason']}): the smallest m with ≥ 20% changed "
          "and ≤ 10% broken. ![calibration](calibration.png)", "",
          f"Projection before the pilot: {proj['projected_t4h_with_safety']:.2f} T4-h with the 1.25 "
          f"safety factor, {proj['used_t4h']:.2f} used; within budget.", "",
          f"Items: {meta['n_failures']} failures ({meta['failures_by_split']}), {meta['n_solved']} "
          f"matched solved items, {meta['templates']} templates; solved drawn with replacement "
          f"{meta['fallbacks']['with_replacement']}×, from other templates "
          f"{meta['fallbacks']['other_template']}×. Self-sourced gate open on "
          f"{meta['gate_open_failures']}/{meta['n_failures']} failures and "
          f"{meta['gate_open_solved']}/{meta['n_solved']} solved items.", "",
          "## Pilot at m\\* (percentage points; 95% template-clustered bootstrap CIs)", "",
          f"Verdicts (all items, words fixed in prereg/R1.md): {r['verdicts']}.", ""]
    for s in SETS:
        v = r[s]
        L += [f"### {s}: {v['n_failures']} failures + {v['n_solved']} solved, {v['n_templates']} templates", "",
              "| condition | repair | damage | net gain | p vs random | net − random mean | changed | broken |",
              "|---|---|---|---|---|---|---|---|"]
        for c, d in v["conditions"].items():
            L.append(f"| {c} | {pct(d['repair'])} | {pct(d['damage'])} | {ci(d['net'])} | "
                     f"{d['p_vs_random']:.3f} | {ci(d['minus_random_mean'])} | "
                     f"{100 * d['changed_answers']:.1f} | {100 * d['broken']:.1f} |")
        q = v["random"]
        L += [f"| random (100 dirs) | {100 * q['repair_mean']:.1f} | {100 * q['damage_mean']:.1f} | "
              f"{100 * q['net_mean']:+.1f} (2.5–97.5%: {100 * q['net_q025']:+.1f}, {100 * q['net_q975']:+.1f}) "
              f"| | | {100 * q['changed_answers']:.1f} | {100 * q['broken']:.1f} |", ""]
    L += ["Oracle damage is 0 by construction (the oracle edit is zero on solved items, gold = bound)."]
    return "\n".join(L) + "\n"


def exploratory_md(x: dict) -> str:
    L = ["", "## EXPLORATORY (not pre-registered, D51): repair and damage each against random", "",
         "Net gain mixes repair with damage, and the oracle edit is never applied to solved items. "
         "Here each part is compared with the 100 random directions on its own. Points; CI of "
         "condition − random mean (template-clustered, 2,000 resamples); p = (1 + #random as good) / 101.", "",
         "| part | set | n (templates) | random mean | oracle | self | eval |", "|---|---|---|---|---|---|---|"]
    for part in ("repair (failures)", "damage (solved)"):
        for k, d in x[part].items():
            cells = [f"{100 * d[c]['estimate']:.1f}; {ci(d[c]['minus_random_mean'])}; p {d[c]['p_vs_random']:.3f}"
                     for c in ("oracle", "self", "eval")]
            L.append(f"| {part} | {k} | {d['n']} ({d['templates']}) | {100 * d['random_mean']:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DIR)
    d = ap.parse_args().dir
    j = lambda n: json.loads((d / n).read_text())  # noqa: E731
    r, cal = j("r1_results.json"), j("calibration.json")
    md = markdown(r, cal, j("projection.json"), j("session_status.json"), j("inputs_meta.json"))
    if (d / "r1_exploratory.json").exists():
        md += exploratory_md(j("r1_exploratory.json"))
    (d / "r1.md").write_text(md)
    plot(cal, d / "calibration.png")
    print((d / "r1.md").read_text())


if __name__ == "__main__":
    main()
