"""EXPLORATORY (not pre-registered; DECISIONS D51): R1 repair and damage, each against the random
null on its own.

The pre-registered R1 comparison is on net gain. The oracle edit is zero on solved items, so its
damage is 0 by construction while random directions are applied to solved items too; the net-gain
comparison therefore mixes "repairs more" with "is not applied to solved items". This script
separates the two: repair (failures only) and damage (solved items only) of each condition against
the per-direction distribution of the 100 random directions, with p = (1 + #random >= cond) / 101
(for damage: #random <= cond) and a template-clustered CI (2,000 resamples) of cond − random mean.

    python -m analysis.r1_exploratory --out-dir DIR_WITH_main_w*.jsonl --inputs DIR_WITH_items.jsonl
    -> reports/m2/r1/r1_exploratory.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.stats import cluster_bootstrap
from harness.config import REPO_ROOT
from repair.analyze import CONDS, read_items, read_rows, stability_labels

OUT = REPO_ROOT / "reports" / "m2" / "r1" / "r1_exploratory.json"


def role_vs_random(rows, items, role, subset=None) -> dict:
    keys = sorted(i for i, it in items.items() if it["role"] == role and (subset is None or subset(it)))
    pos = {i: k for k, i in enumerate(keys)}
    templates = np.array([items[i]["template_id"] for i in keys])
    uniq, tidx = np.unique(templates, return_inverse=True)
    by = defaultdict(lambda: np.full(len(keys), np.nan))
    for r in rows:
        if r["item"] in pos:
            by[r["cond"]][pos[r["item"]]] = float(r["correct"])
    rand = sorted([c for c in by if c.startswith("rand")], key=lambda c: int(c[4:]))
    R = np.stack([by[c] for c in rand])
    # repair = share made correct (failures); damage = share made wrong (solved)
    stat = (lambda x, w: (w[tidx] * x).sum() / w[tidx].sum()) if role == "failure" else \
           (lambda x, w: (w[tidx] * (1 - x)).sum() / w[tidx].sum())
    one = np.ones(len(uniq))
    rnd = np.array([stat(R[k], one) for k in range(len(R))])
    out = {"n": len(keys), "templates": int(len(uniq)), "random_mean": float(rnd.mean()),
           "random_q025": float(np.quantile(rnd, 0.025)), "random_q975": float(np.quantile(rnd, 0.975))}
    for c in CONDS:
        x = by[c]
        est = stat(x, one)
        cnt = (rnd >= est).sum() if role == "failure" else (rnd <= est).sum()
        out[c] = {"estimate": float(est), "p_vs_random": float((1 + cnt) / (1 + len(rnd))),
                  "minus_random_mean": cluster_bootstrap(
                      templates, lambda w, x=x: stat(x, w) - float(np.mean([stat(R[k], w) for k in range(len(R))])),
                      n_resamples=2000, seed=0)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=REPO_ROOT / "data" / "generated" / "repair_r1_inputs")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    items, rows = read_items(a.inputs), read_rows(a.out_dir, "main")
    st = stability_labels(a.out_dir)
    sets = {"all": None, "train": lambda it: it["split"] == "train", "dev": lambda it: it["split"] == "dev",
            "borderline failures": lambda it: st.get(it["item"]) == "borderline",
            "stable failures": lambda it: st.get(it["item"]) == "stable"}
    res = {"status": "EXPLORATORY, not pre-registered (D51); development result, automatic labels",
           "repair (failures)": {k: role_vs_random(rows, items, "failure", f) for k, f in sets.items()},
           "damage (solved)": {k: role_vs_random(rows, items, "solved", f) for k, f in list(sets.items())[:3]}}
    a.out.write_text(json.dumps(res, indent=1, default=float))
    for part, v in res.items():
        if part == "status":
            continue
        for k, d in v.items():
            print(part, k, f"n {d['n']} ({d['templates']} t)", f"random {d['random_mean']:.3f}",
                  *[f"{c} {d[c]['estimate']:.3f} p {d[c]['p_vs_random']:.3f} "
                    f"diff {d[c]['minus_random_mean']['estimate']:+.3f} "
                    f"[{d[c]['minus_random_mean']['ci_low']:+.3f}, {d[c]['minus_random_mean']['ci_high']:+.3f}]"
                    for c in CONDS])


if __name__ == "__main__":
    main()
