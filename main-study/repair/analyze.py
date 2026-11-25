"""Calibration rule, budget projection and the analysis of the redesigned repair pilot
(prereg/R1.md). CPU only; reads the worker result files.

    python -m repair.analyze calibrate --inputs DIR --out DIR      -> calibration.json (m*)
    python -m repair.analyze project   --inputs DIR --out DIR --used-t4h X   -> projection.json
    python -m repair.analyze report    --inputs DIR --out DIR      -> r1_results.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.stats import cluster_bootstrap

GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CHANGE_MIN = 0.20        # oracle changes the final answer of >= 20% of dev failures ...
BREAK_MAX = 0.10         # ... with <= 10% of them broken (no answer, or the 512-token cap)
CAL_RANDOM = 10          # random directions in the calibration (rand0..rand9)
BUDGET_T4H = 8.0
SAFETY = 1.25
MEI = 0.05               # minimal effect of interest: 5 points of net gain
CONDS = ("oracle", "self", "eval")


def read_rows(out: Path, tag: str) -> list[dict]:
    rows = []
    for p in sorted(out.glob(f"{tag}_w*.jsonl")):
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    seen, uniq = set(), []
    for r in rows:                       # a resumed worker may repeat a key: keep the first
        k = (r["item"], r.get("cond"), r.get("m"), r.get("sample"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def read_items(inputs: Path) -> dict[int, dict]:
    return {d["item"]: d for d in (json.loads(x) for x in (inputs / "items.jsonl").read_text().splitlines() if x.strip())}


def group(cond: str) -> str:
    return "random" if cond.startswith("rand") else cond


def curve(rows: list[dict], items: dict) -> dict:
    """Per multiple and condition group: change / break / correct rates on dev failures and
    dev solved items."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        it = items[r["item"]]
        acc[(r["m"], group(r["cond"]))][it["role"]].append(r)
    out = {}
    for (m, g), by in sorted(acc.items()):
        e = {}
        for role, rs in by.items():
            e[role] = {"n": len(rs), "answer_changed": float(np.mean([x["answer_changed"] for x in rs])),
                       "broken": float(np.mean([x["broken"] for x in rs])),
                       "continuation_changed": float(np.mean([x["diverged"] for x in rs])),
                       "correct": float(np.mean([x["correct"] for x in rs]))}
        out.setdefault(str(m), {})[g] = e
    return out


def decide(cur: dict, grid=GRID) -> dict:
    """m* = the smallest multiple at which the oracle edit changes the final answer of >= 20% of
    dev failures while <= 10% of them are broken. Stop climbing at the first multiple whose
    oracle break rate exceeds 10% (outputs break before they change)."""
    trail = []
    for m in grid:
        o = cur.get(str(m), {}).get("oracle", {}).get("failure")
        if o is None:
            return {"m_star": None, "reason": f"oracle not run at {m}", "trail": trail}
        trail.append({"m": m, "answer_changed": o["answer_changed"], "broken": o["broken"]})
        if o["answer_changed"] >= CHANGE_MIN and o["broken"] <= BREAK_MAX:
            return {"m_star": m, "reason": "criterion met", "trail": trail}
        if o["broken"] > BREAK_MAX:
            return {"m_star": None, "reason": f"outputs break first (broken {o['broken']:.2f} at {m}x "
                                               f"with answer changes {o['answer_changed']:.2f})",
                    "trail": trail}
    return {"m_star": None, "reason": "no multiple on the grid changes >= 20% of answers", "trail": trail}


def project(out: Path, inputs: Path, m_star: float, used_t4h: float) -> dict:
    """Projected T4-hours of the main run at m*, from the measured calibration timings and the
    calibration divergence rates at m* (random rate for the 100 random directions)."""
    items = read_items(inputs)
    tim = []
    for p in out.glob("cal_*_w*_timing.json"):
        tim += json.loads(p.read_text())
    scan_s = sum(t["scan_s"] for t in tim)
    scan_rows = max(sum(t["scan_rows"] for t in tim), 1)
    regen_s = sum(t["regen_s"] for t in tim)
    regen_rows = max(sum(t["regen_rows"] for t in tim), 1)
    rows = read_rows(out, "cal_oracle") + read_rows(out, "cal_rest")
    rate = {}
    for g in ("oracle", "self", "eval", "random"):
        rs = [r for r in rows if r["m"] == m_star and group(r["cond"]) == g and r["edited"]]
        rate[g] = float(np.mean([r["diverged"] for r in rs])) if rs else 1.0
    n_f = sum(1 for it in items.values() if it["role"] == "failure")
    n = len(items)
    gate = sum(1 for it in items.values() if it["gate_open"])
    edited = {"oracle": n_f, "self": gate, "eval": n, "random": 100 * n}
    exp_scan = sum(edited.values())
    exp_regen = sum(edited[g] * rate[g] for g in edited)
    sec = exp_scan * scan_s / scan_rows + exp_regen * regen_s / regen_rows
    t4h = sec / 3600
    ok = used_t4h + SAFETY * t4h <= BUDGET_T4H
    return {"m_star": m_star, "divergence_rate_at_m_star": rate, "edited_runs": edited,
            "expected_regenerations": exp_regen, "sec_per_scan_row": scan_s / scan_rows,
            "sec_per_regen_row": regen_s / regen_rows, "projected_t4h": t4h,
            "projected_t4h_with_safety": SAFETY * t4h, "used_t4h": used_t4h,
            "budget_t4h": BUDGET_T4H, "within_budget": ok}


def _rates(new: np.ndarray, solved: np.ndarray, w: np.ndarray, tidx: np.ndarray) -> tuple:
    """Weighted (by template resample weight) repair, damage and net gain."""
    ww = w[tidx]
    f, s = ~solved, solved
    rep = (ww[f] * new[f]).sum() / max(ww[f].sum(), 1e-12)
    dmg = (ww[s] * (1 - new[s])).sum() / max(ww[s].sum(), 1e-12)
    net = (ww * (new - solved)).sum() / max(ww.sum(), 1e-12)
    return rep, dmg, net


def summarize(rows: list[dict], items: dict, subset=None, n_boot: int = 10_000) -> dict:
    """Repair, damage, net gain per condition with template-clustered CIs, and the random null."""
    keys = sorted(i for i in items if subset is None or subset(items[i]))
    if not keys:
        return {"n_items": 0}
    templates = np.array([items[i]["template_id"] for i in keys])
    uniq, tidx = np.unique(templates, return_inverse=True)
    solved = np.array([items[i]["role"] == "solved" for i in keys])
    pos = {i: k for k, i in enumerate(keys)}
    by = defaultdict(lambda: np.full(len(keys), np.nan))
    for r in rows:
        if r["item"] in pos:
            by[r["cond"]][pos[r["item"]]] = float(r["correct"])
    out = {"n_items": len(keys), "n_failures": int((~solved).sum()), "n_solved": int(solved.sum()),
           "n_templates": int(len(uniq)), "conditions": {}}
    rand = sorted([c for c in by if c.startswith("rand")], key=lambda c: int(c[4:]))
    R = np.stack([by[c] for c in rand]) if rand else None
    complete_rand = R is not None and not np.isnan(R).any()
    for c in CONDS:
        if c not in by or np.isnan(by[c]).any():
            out["conditions"][c] = {"missing": int(np.isnan(by[c]).sum()) if c in by else len(keys)}
            continue
        new = by[c]
        e = {}
        for j, name in enumerate(("repair", "damage", "net")):
            e[name] = cluster_bootstrap(templates, lambda w, j=j: _rates(new, solved, w, tidx)[j],
                                        n_resamples=n_boot, seed=0)
        if complete_rand:
            nets = np.array([_rates(R[k], solved, np.ones(len(uniq)), tidx)[2] for k in range(len(R))])
            e["p_vs_random"] = float((1 + (nets >= e["net"]["estimate"]).sum()) / (1 + len(nets)))
            e["minus_random_mean"] = cluster_bootstrap(
                templates, lambda w: _rates(new, solved, w, tidx)[2]
                - float(np.mean([_rates(R[k], solved, w, tidx)[2] for k in range(len(R))])),
                n_resamples=min(n_boot, 2000), seed=0)
        e["changed_answers"] = float(np.mean([r["answer_changed"] for r in rows
                                              if r["cond"] == c and r["item"] in pos]))
        e["broken"] = float(np.mean([r["broken"] for r in rows if r["cond"] == c and r["item"] in pos]))
        out["conditions"][c] = e
    if complete_rand:
        nets = np.array([_rates(R[k], solved, np.ones(len(uniq)), tidx) for k in range(len(R))])
        out["random"] = {"n_directions": len(R), "net_mean": float(nets[:, 2].mean()),
                         "net_q025": float(np.quantile(nets[:, 2], 0.025)),
                         "net_q975": float(np.quantile(nets[:, 2], 0.975)),
                         "repair_mean": float(nets[:, 0].mean()), "damage_mean": float(nets[:, 1].mean()),
                         "changed_answers": float(np.mean([r["answer_changed"] for r in rows
                                                           if r["cond"].startswith("rand") and r["item"] in pos])),
                         "broken": float(np.mean([r["broken"] for r in rows
                                                  if r["cond"].startswith("rand") and r["item"] in pos]))}
    else:
        out["random"] = {"complete": False}
    return out


def verdict(e: dict, random_ok: bool) -> str:
    """Words fixed in prereg/R1.md: supported only if the net CI excludes 0, p <= 0.05 against the
    random null and the estimate is >= 5 points; not supported if the net CI's upper end is below
    5 points; otherwise inconclusive."""
    if "net" not in e:
        return "not run"
    net = e["net"]
    if random_ok and net["ci_low"] > 0 and e.get("p_vs_random", 1) <= 0.05 and net["estimate"] >= MEI:
        return "supported"
    if net["ci_high"] < MEI:
        return "not supported"
    return "inconclusive"


def stability_labels(out: Path) -> dict[int, str]:
    rows = read_rows(out, "stability")
    by = defaultdict(list)
    for r in rows:
        by[r["item"]].append(r["correct"])
    return {i: ("stable" if not any(v) else "borderline") for i, v in by.items() if len(v) == 5}


def report(out: Path, inputs: Path) -> dict:
    items = read_items(inputs)
    rows = read_rows(out, "main")
    st = stability_labels(out)
    for i, it in items.items():
        it["stability"] = st.get(i)
    subsets = {
        "all (train + dev)": None,
        "train": lambda it: it["split"] == "train",
        "dev": lambda it: it["split"] == "dev",
        "content/binary items": lambda it: it["variable"] == "content/binary",
        "content/4-class items": lambda it: it["variable"] == "content/4-class",
    }
    res = {k: summarize(rows, items, f) for k, f in subsets.items()}
    # stable vs borderline: failures of that kind + all solved items of the same templates
    for lab in ("stable", "borderline"):
        fs = {i for i, it in items.items() if it["role"] == "failure" and it["stability"] == lab}
        tr = {items[i]["trace_id"] for i in fs}
        res[f"{lab} failures"] = summarize(
            rows, items, lambda it, fs=fs, tr=tr: it["item"] in fs or it.get("matched_failure") in tr)
    rok = res["all (train + dev)"].get("random", {}).get("n_directions") == 100
    res["verdicts"] = {c: verdict(res["all (train + dev)"]["conditions"].get(c, {}), rok) for c in CONDS}
    res["stability_counts"] = {lab: sum(1 for v in st.values() if v == lab) for lab in ("stable", "borderline")}
    res["labels"] = "automatic (parser v4); S5 annotator kappa pending: development result"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["calibrate", "project", "report"])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--used-t4h", type=float, default=0.0)
    a = ap.parse_args()
    items = read_items(a.inputs)
    if a.what == "calibrate":
        cur = curve(read_rows(a.out, "cal_oracle") + read_rows(a.out, "cal_rest"), items)
        res = {"curve": cur, **decide(cur)}
        (a.out / "calibration.json").write_text(json.dumps(res, indent=2))
    elif a.what == "project":
        cal = json.loads((a.out / "calibration.json").read_text())
        res = project(a.out, a.inputs, cal["m_star"], a.used_t4h)
        (a.out / "projection.json").write_text(json.dumps(res, indent=2))
    else:
        res = report(a.out, a.inputs)
        (a.out / "r1_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":
    main()
