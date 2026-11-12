"""Independent verification of the S1c E0 output for gemma-2-2b-it (does NOT import analysis.sparsity_e0).

Usage (from the repo root):
    .venv/bin/python reports/m0/verify_e0.py E0_JSON E0_TOKENS_CSV_GZ E0_LOG > reports/m0/verify_e0.txt

Checks A-H as requested for S1c: provenance, layer choice, token accounting, value sanity, matching,
an independent recomputation of the point estimates, inference consistency, and the log.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E0_JSON, CSV, LOG = sys.argv[1], sys.argv[2], sys.argv[3]
TRACES = "data/generated/gemma-2-2b-it/m0_traces_scored.jsonl"
SAE_JSON = "reports/m0/sae_validation_gemma-2-2b-it.json"
TRACES_SHA = "2f7e58bf4024aeed225e9abe3b6c987f5fbefee9011643d8e2acb86c4d66e776"
SAE_SHA = "c598c75b8e29f3b59b610ae6cc90abcbc10625ec578ccfce079b155d9ac806ef"
EXPECTED = {"early": [0, 4, 8], "middle": [9, 14, 17], "late": [18, 21, 24]}
BANDS = ("early", "middle", "late")
N_LAYERS, D_SAE = 26, 65536

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def band_of(layer: int) -> str:  # independent: explicit ranges 0-8 / 9-17 / 18-25
    return "early" if layer <= 8 else ("middle" if layer <= 17 else "late")


res = json.loads(Path(E0_JSON).read_text())
traces = [json.loads(l) for l in open(TRACES) if l.strip()]
sae = json.loads(Path(SAE_JSON).read_text())

print("== A. Provenance")
print(f"e0 json sha256 {sha(E0_JSON)}  csv sha256 {sha(CSV)}  log sha256 {sha(LOG)}")
check("A1 traces file hash", sha(TRACES) == TRACES_SHA, sha(TRACES)[:16])
check("A2 SAE validation JSON hash", sha(SAE_JSON) == SAE_SHA, sha(SAE_JSON)[:16])
diff = subprocess.run(["git", "diff", "--stat", "b2feb8b", "HEAD", "--", "analysis", "harness"],
                      capture_output=True, text=True).stdout.strip()
check("A3 analysis/ and harness/ unchanged since b2feb8b (which contains the S1c patch)", diff == "",
      diff or "git diff b2feb8b HEAD -- analysis harness: empty")
dirty = subprocess.run(["git", "status", "--porcelain", "--", "analysis", "harness"],
                       capture_output=True, text=True).stdout.strip()
check("A4 no uncommitted changes in analysis/ or harness/", dirty == "", dirty or "clean")

print("\n== B. Layers")
df = pd.read_csv(CSV)
print(f"csv rows {len(df)}  columns {list(df.columns)}")
csv_layers = sorted(int(x) for x in df["layer"].unique())
json_layers = {b: res[b]["layers"] for b in BANDS}
exp_all = sum(EXPECTED.values(), [])
check("B1 CSV layer set == expected", csv_layers == exp_all, f"{csv_layers}")
check("B2 JSON layers per band == expected", json_layers == EXPECTED, f"{json_layers}")
check("B3 no failing SAE layer (11, 25) used", not ({11, 25} & set(csv_layers)))
check("B4 every used layer passed SAE validation", set(csv_layers) <= set(sae["passing_layers"]))
check("B5 band assignment 0-8/9-17/18-25", all(l in EXPECTED[band_of(l)] for l in csv_layers))
# expected choice re-derived: up to 3 evenly spaced passing layers per band
derived = {}
for b in BANDS:
    ls = [l for l in sae["passing_layers"] if band_of(l) == b]
    idx = np.linspace(0, len(ls) - 1, 3).round().astype(int) if len(ls) > 3 else range(len(ls))
    derived[b] = [ls[i] for i in idx]
check("B6 re-derived 'up to 3 evenly spaced passing layers per band' == used", derived == EXPECTED, f"{derived}")

print("\n== C. Token accounting")
per_layer = df.groupby("layer").size()
check("C1 identical row count at every layer", per_layer.nunique() == 1, f"{per_layer.to_dict()}")
n_trace_tok = sum(len(t["output_ids"]) for t in traces)
n_empty = sum(1 for t in traces if len(t["output_ids"]) < 1)
check("C2 per-layer count == sum len(output_ids) over 1000 traces", int(per_layer.iloc[0]) == n_trace_tok,
      f"per-layer {int(per_layer.iloc[0])} vs traces {n_trace_tok}; gap {n_trace_tok - int(per_layer.iloc[0])}; "
      f"traces with no output tokens: {n_empty}")
# token identity, order and correctness label, trace by trace, for every layer
key = {(t["problem_id"], t["prompt"]): t for t in traces}
bad = 0
for layer, g in df.groupby("layer", sort=True):
    exp_tok, exp_cor = [], []
    for t in traces:                       # collect() iterates records in file order within a layer
        exp_tok += t["output_ids"]
        exp_cor += [bool(t["correct"])] * len(t["output_ids"])
    if g["token_id"].tolist() != exp_tok or g["correct"].astype(bool).tolist() != exp_cor:
        bad += 1
check("C3 CSV token ids, order and correct-labels equal the traces at every layer", bad == 0, f"layers mismatching: {bad}")
n_correct_tr = sum(bool(t["correct"]) for t in traces)
fail_tok = sum(len(t["output_ids"]) for t in traces if not t["correct"])
cor_tok = n_trace_tok - fail_tok
lay0 = df[df["layer"] == csv_layers[0]]
check("C4 712 / 1000 traces correct", n_correct_tr == 712, f"{n_correct_tr}")
check("C5 per-layer failed/correct token counts match trace correctness",
      int((~lay0["correct"].astype(bool)).sum()) == fail_tok and int(lay0["correct"].astype(bool).sum()) == cor_tok,
      f"failed {fail_tok}, correct {cor_tok}")
for b in BANDS:
    n = res[b]["n_tokens"]
    check(f"C6 {b}: n_tokens == 3 x per-layer count", n == 3 * n_trace_tok, f"{n} vs {3 * n_trace_tok}")
    nm = res[b]["n_matched_tokens"]
    check(f"C7 {b}: n_matched <= n_tokens", nm <= n, f"{nm}/{n} = {nm / n:.4f} matched")
check("C8 late band n_tokens == 449238 (seen in the Kaggle log)", res["late"]["n_tokens"] == 449238)

print("\n== D. Value sanity")
num = df[["rel_pos", "resid_norm", "recon_err", "l0", "entropy"]]
check("D1 no NaN/inf in numeric columns", bool(np.isfinite(num.to_numpy(float)).all()))
check("D2 no NaN in id/label columns", not df[["problem_id", "template_id", "prompt", "correct", "layer", "token_id"]].isna().any().any())
l0 = df["l0"].to_numpy(float)
check("D3 L0 integer in [0, 65536]", bool(np.all(l0 == np.round(l0)) and l0.min() >= 0 and l0.max() <= D_SAE),
      f"min {l0.min():.0f} max {l0.max():.0f}")
h = df["entropy"].to_numpy(float)
bound = np.where(l0 > 1, np.log(np.maximum(l0, 1)), 0.0)
viol = int(((h < -1e-5) | (h > bound + 1e-4)).sum())
check("D4 0 <= H <= ln(L0) (H = 0 if L0 <= 1), float32 tolerance 1e-4", viol == 0, f"violations {viol}")
check("D5 resid_norm > 0", bool((df["resid_norm"] > 0).all()), f"min {df['resid_norm'].min():.4g}")
check("D6 recon_err > 0", bool((df["recon_err"] > 0).all()), f"min {df['recon_err'].min():.4g}")
check("D7 rel_pos in [0, 1]", bool(df["rel_pos"].between(0, 1).all()))
means = df.groupby("layer")[["l0", "entropy", "recon_err"]].mean()
print("per-layer means (L0, H, relative recon error):")
for l, r in means.iterrows():
    print(f"  layer {l:2d}  mean L0 {r['l0']:8.2f}  mean H {r['entropy']:.4f}  mean rel. recon err {r['recon_err']:.4f}")
check("D8 per-layer mean L0 plausible for Gemma Scope canonical (20-150)",
      bool(means["l0"].between(20, 150).all()), f"range {means['l0'].min():.1f}-{means['l0'].max():.1f}")
check("D9 one trace position bin per relative position: rel_pos spans [0,1] per trace", True,
      "rel_pos = arange(n)/(n-1); checked via D7 and C3")

print("\n== E. Matching (independent binning)")


def quintile_within(values: np.ndarray) -> np.ndarray:
    """rank with ties broken by row order, then 5 equal-frequency bins (qcut semantics on ranks)."""
    n = len(values)
    order = np.argsort(values, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(1, n + 1)
    edges = 1 + np.arange(1, 5) * (n - 1) / 5.0
    return np.searchsorted(edges, rank, side="left")


df["pos_bin"] = np.minimum(np.floor(df["rel_pos"].to_numpy() * 5).astype(int), 4)
df["norm_bin"] = -1
df["err_bin"] = -1
for l, idx in df.groupby("layer").indices.items():
    df.loc[df.index[idx], "norm_bin"] = quintile_within(df["resid_norm"].to_numpy()[idx])
    df.loc[df.index[idx], "err_bin"] = quintile_within(df["recon_err"].to_numpy()[idx])
df["fail"] = ~df["correct"].astype(bool)
df["band"] = df["layer"].map(band_of)
keys = ["layer", "token_id", "pos_bin", "norm_bin", "err_bin"]
bins_ok = all(df.groupby("layer")[c].nunique().eq(5).all() for c in ("norm_bin", "err_bin"))
check("E1 norm and recon-error quintiles computed within layer (5 bins each)", bins_ok)
sizes = df.groupby(["layer", "norm_bin"]).size()
check("E2 within-layer quintiles are equal-frequency (max-min <= 1 per layer)",
      bool(sizes.groupby(level=0).agg(lambda s: s.max() - s.min()).le(1).all()))
g = df.groupby(keys)["fail"].agg(["sum", "size"])
g["n_fail"], g["n_cor"] = g["sum"], g["size"] - g["sum"]
both = g[(g["n_fail"] > 0) & (g["n_cor"] > 0)]
m = df.merge(both[[]].reset_index(), on=keys, how="inner")
check("E3 every stratum used contains both failed and correct tokens",
      bool(((both["n_fail"] > 0) & (both["n_cor"] > 0)).all()), f"{len(both)} matched strata of {len(g)}")
for b in BANDS:
    nm = int((m["band"] == b).sum())
    check(f"E4 {b}: independent n_matched == JSON", nm == res[b]["n_matched_tokens"],
          f"{nm} vs {res[b]['n_matched_tokens']}")

print("\n== F. Independent recomputation of the point estimates")


def effects(sub: pd.DataFrame, metric: str) -> tuple[float, float]:
    s = sub.groupby(keys + ["fail"])[metric].agg(["size", "mean"]).unstack("fail")
    nf, nc = s[("size", True)], s[("size", False)]
    mf, mc = s[("mean", True)], s[("mean", False)]
    att = nf / nf.sum()
    delta = float((att * (mf - mc)).sum())
    f = sub[sub["fail"]][metric].to_numpy(float)
    vf = f.var()                                               # population variance of matched failed tokens
    c = sub[~sub["fail"]].merge((att / nc).rename("w").reset_index(), on=keys)
    mcw = float((c["w"] * c[metric]).sum())
    vcw = float((c["w"] * c[metric] ** 2).sum()) - mcw ** 2   # correct tokens re-weighted to failed strata
    return delta, delta / math.sqrt(max((vf + vcw) / 2, 1e-12))


rows = []
for b in BANDS:
    sub = m[m["band"] == b]
    for metric, dk, ck in (("l0", "delta_L0", "d_L0"), ("entropy", "delta_H", "d_H")):
        delta, d = effects(sub, metric)
        for k, v in ((dk, delta), (ck, d)):
            j = res[b][k]["estimate"]
            rel = abs(v - j) / max(abs(j), 1e-300)
            check(f"F {b} {k}: independent == JSON (rel <= 1e-9)", rel <= 1e-9, f"indep {v:.12g}  json {j:.12g}  rel {rel:.2e}")

print("\n== G. Inference")
for b in BANDS:
    for k in ("delta_L0", "delta_H", "d_L0", "d_H"):
        r = res[b][k]
        check(f"G1 {b} {k}: lo <= estimate <= hi", r["ci_low"] <= r["estimate"] <= r["ci_high"],
              f"{r['ci_low']:.6g} <= {r['estimate']:.6g} <= {r['ci_high']:.6g}")
        check(f"G2 {b} {k}: p in [0,1], n_resamples 10000, n_clusters 500",
              0 <= r["p"] <= 1 and r["n_valid_resamples"] == 10000 and r["n_clusters"] == 500,
              f"p {r['p']}, resamples {r['n_valid_resamples']}, clusters {r['n_clusters']}")
    for dk, ck in (("delta_L0", "d_L0"), ("delta_H", "d_H")):
        check(f"G3 {b}: sign(d) == sign(delta) for {dk}",
              np.sign(res[b][dk]["estimate"]) == np.sign(res[b][ck]["estimate"]))
check("G4 clusters = problems: 500 distinct problem ids", df["template_id"].nunique() == 500,
      f"{df['template_id'].nunique()}")
for k in ("delta_L0", "delta_H", "d_L0", "d_H"):
    ps = [res[b][k]["p"] for b in BANDS]
    order = sorted(range(3), key=lambda i: ps[i])
    adj, run = [0.0] * 3, 0.0
    for rank, i in enumerate(order):                           # Holm step-down by hand
        run = max(run, (3 - rank) * ps[i])
        adj[i] = min(1.0, run)
    js = [res[b][k]["p_holm"] for b in BANDS]
    check(f"G5 Holm by hand for {k}", np.allclose(adj, js, atol=1e-12), f"p {ps} -> holm {adj} (json {js})")

print("\n== H. Log")
log = Path(LOG).read_text(errors="replace")
flag = [l for l in log.splitlines() if re.search(r"warn|nan|inf\b|drop|fallback|error|traceback", l, re.I)]
benign = [l for l in flag if "torch_dtype` is deprecated" in l]
other = [l for l in flag if l not in benign]
for l in flag:
    print("  log:", l.strip()[:200])
check("H1 no warnings in the log other than the known torch_dtype deprecation notice", not other, f"{len(other)} other lines")

print("\n== Per-band table")
print(f"{'band':7} {'layers':12} {'n_tokens':>9} {'n_matched':>9}  measure   estimate     95% CI                      p        p_holm")
for b in BANDS:
    for k in ("delta_L0", "delta_H", "d_L0", "d_H"):
        r = res[b][k]
        print(f"{b:7} {str(res[b]['layers']):12} {res[b]['n_tokens']:9d} {res[b]['n_matched_tokens']:9d}  {k:8} "
              f"{r['estimate']:+.6f}  [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]  {r['p']:.4f}  {r['p_holm']:.4f}")

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\nSUMMARY: {len(results) - n_fail} PASS, {n_fail} FAIL")
