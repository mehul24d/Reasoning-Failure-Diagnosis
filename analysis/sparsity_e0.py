"""E0: prerequisite sparsity check (plan Appendix A).

Per token with SAE activations a (active set A = {j : a_j > 0}):
    L0 = |A|,   H = -sum_{j in A} p_j log p_j,   p_j = a_j / sum_{k in A} a_k        (eq. 9)
Failed-trace tokens are compared with correct-trace tokens after matching on token identity,
relative-position bin and residual-norm bin, controlling for per-token reconstruction error
(here: reconstruction-error bin is an extra matching key). For each depth band (early, middle,
late) we report the matched difference dL0, dH and Cohen's d (eq. 10) with template-clustered
bootstrap CIs, Holm-corrected across bands.

Two stages:
  collect  (GPU): token-level records -> CSV.gz, one row per (trace, layer, trace token)
  analyse  (CPU): matching + bootstrap -> JSON

The check is reported for completeness; it is not a research contribution (Appendix A).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .stats import cluster_bootstrap, holm

BANDS = ("early", "middle", "late")


# --------------------------------------------------------------------------- token metrics
def l0_and_entropy(acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """acts: [n_tokens, D] non-negative SAE activations. Returns (L0, H) per token."""
    a = acts.float().clamp_min(0)
    active = a > 0
    l0 = active.sum(-1)
    total = a.sum(-1, keepdim=True)
    p = torch.where(active, a / total.clamp_min(1e-30), torch.zeros_like(a))
    h = -(torch.where(active, p * torch.log(p.clamp_min(1e-30)), torch.zeros_like(p))).sum(-1)
    h = torch.where(l0 > 0, h, torch.zeros_like(h))
    return l0, h


def depth_band(layer: int, n_layers: int) -> str:
    """Thirds of the depth range."""
    return BANDS[min(2, (3 * layer) // n_layers)]


# --------------------------------------------------------------------------- collect (GPU)
@torch.no_grad()
def collect(model, load_sae, layers: list[int], records: list[dict]) -> pd.DataFrame:
    """One SAE in memory at a time (a 9B 131k SAE is ~3.7 GB in float32).

    `load_sae(layer)` returns an object with encode/decode. The forward pass is repeated per
    layer; that costs GPU time but keeps memory flat.
    """
    rows = []
    for layer in layers:
        sae = load_sae(layer)
        name = f"blocks.{layer}.hook_resid_post"
        sp = next(sae.parameters(), None) if hasattr(sae, "parameters") else None
        dt = sp.dtype if sp is not None else torch.float32
        sdev = sp.device if sp is not None else torch.device("cpu")
        for rec in records:
            ids = rec["prompt_ids"] + rec["output_ids"]
            start = len(rec["prompt_ids"])
            n_trace = len(ids) - start
            if n_trace < 1:
                continue
            toks = torch.tensor([ids], device=model.cfg.device)
            _, cache = model.run_with_cache(toks, names_filter=name)
            h = cache[name][0, start:]
            f = sae.encode(h.to(sdev, dt))
            h_hat = sae.decode(f).to(h.device, h.dtype)
            h = h.to(sdev)
            h_hat = h_hat.to(sdev)
            l0, ent = l0_and_entropy(f)
            norm = h.float().norm(dim=-1)
            rerr = ((h - h_hat).float().norm(dim=-1) ** 2) / (norm ** 2).clamp_min(1e-12)
            rows.append(pd.DataFrame({
                "problem_id": rec["problem_id"],
                "template_id": rec.get("template_id") or rec["problem_id"],
                "prompt": rec.get("prompt"), "correct": bool(rec["correct"]), "layer": layer,
                "token_id": ids[start:], "rel_pos": np.arange(n_trace) / max(n_trace - 1, 1),
                "resid_norm": norm.cpu().numpy(), "recon_err": rerr.cpu().numpy(),
                "l0": l0.cpu().numpy(), "entropy": ent.cpu().numpy(),
            }))
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pick_layers(passing: list[int], n_layers: int, per_band: int) -> list[int]:
    """Up to `per_band` evenly spaced passing layers from each depth band (pre-registered)."""
    chosen = []
    for band in BANDS:
        ls = sorted(l for l in passing if depth_band(l, n_layers) == band)
        if len(ls) <= per_band:
            chosen += ls
        else:
            idx = np.linspace(0, len(ls) - 1, per_band).round().astype(int)
            chosen += [ls[i] for i in idx]
    return chosen


# --------------------------------------------------------------------------- analyse (CPU)
def add_bins(df: pd.DataFrame, n_pos_bins: int = 5, n_norm_bins: int = 5, n_err_bins: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["pos_bin"] = np.minimum((df["rel_pos"] * n_pos_bins).astype(int), n_pos_bins - 1)
    # norm and recon-error bins are quantiles WITHIN each layer (scales differ by depth)
    df["norm_bin"] = df.groupby("layer")["resid_norm"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_norm_bins, labels=False))
    df["err_bin"] = df.groupby("layer")["recon_err"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_err_bins, labels=False))
    df["stratum"] = (df["layer"].astype(str) + "|" + df["token_id"].astype(str) + "|"
                     + df["pos_bin"].astype(str) + "|" + df["norm_bin"].astype(str) + "|"
                     + df["err_bin"].astype(str))
    return df


def matched_effect(df: pd.DataFrame, metric: str):
    """Returns statistic(cluster_weights) -> (delta, d) closure plus cell arrays.

    Matching = exact on stratum; strata need >= 1 failed and >= 1 correct token. The effect is
    the failed-weighted average over strata of (mean_fail - mean_correct) (ATT weighting), and
    Cohen's d divides it by the pooled SD of the matched, re-weighted samples.
    """
    both = df.groupby("stratum")["correct"].agg(lambda c: c.any() and (~c).any())
    m = df[df["stratum"].isin(both[both].index)]
    if m.empty:
        return None, 0
    clusters, cl_idx = np.unique(m["template_id"].astype(str), return_inverse=True)
    strata, st_idx = np.unique(m["stratum"], return_inverse=True)
    y = m[metric].to_numpy(float)
    fail = (~m["correct"].to_numpy(bool)).astype(int)
    n_s = len(strata)
    cell = st_idx * 2 + fail                                  # (stratum, group) cell id
    unit, unit_idx = np.unique(cl_idx.astype(np.int64) * (2 * n_s) + cell, return_inverse=True)
    u_cl = unit // (2 * n_s)
    u_cell = unit % (2 * n_s)
    u_n = np.bincount(unit_idx)
    u_s1 = np.bincount(unit_idx, weights=y)
    u_s2 = np.bincount(unit_idx, weights=y * y)

    def stat(w_cluster: np.ndarray, which: str = "delta") -> float:
        w = w_cluster[u_cl]
        cnt = np.bincount(u_cell, weights=w * u_n, minlength=2 * n_s).reshape(n_s, 2)
        s1 = np.bincount(u_cell, weights=w * u_s1, minlength=2 * n_s).reshape(n_s, 2)
        s2 = np.bincount(u_cell, weights=w * u_s2, minlength=2 * n_s).reshape(n_s, 2)
        ok = (cnt[:, 0] > 0) & (cnt[:, 1] > 0)
        if not ok.any():
            return np.nan
        cnt, s1, s2 = cnt[ok], s1[ok], s2[ok]
        mean = s1 / cnt
        att = cnt[:, 1] / cnt[:, 1].sum()                     # weight strata by failed count
        delta = float((att * (mean[:, 1] - mean[:, 0])).sum())
        if which == "delta":
            return delta
        # re-weight correct tokens to the failed stratum distribution, then pooled SD
        n_f = cnt[:, 1].sum()
        mf = (s1[:, 1]).sum() / n_f
        vf = (s2[:, 1]).sum() / n_f - mf ** 2
        wc = att / cnt[:, 0]                                  # per-token weight for correct
        mc = (wc * s1[:, 0]).sum()
        vc = (wc * s2[:, 0]).sum() - mc ** 2
        sp = np.sqrt(max((vf + vc) / 2, 1e-12))
        return delta / sp

    return (stat, clusters), int(len(m))


def analyse(df: pd.DataFrame, n_layers: int, n_resamples: int = 10_000, seed: int = 0) -> dict:
    df = add_bins(df)
    df["band"] = df["layer"].map(lambda l: depth_band(int(l), n_layers))
    out = {}
    for band in BANDS:
        sub = df[df["band"] == band]
        res = {"n_tokens": int(len(sub)), "layers": sorted(int(x) for x in sub["layer"].unique())}
        for metric, label in (("l0", "delta_L0"), ("entropy", "delta_H")):
            closure, n_matched = matched_effect(sub, metric) if len(sub) else (None, 0)
            res["n_matched_tokens"] = n_matched
            if closure is None:
                res[label] = None
                continue
            stat, clusters = closure
            res[label] = cluster_bootstrap(clusters, lambda w: stat(w, "delta"), n_resamples, seed)
            res[label.replace("delta", "d")] = cluster_bootstrap(clusters, lambda w: stat(w, "d"),
                                                                 n_resamples, seed)
        out[band] = res
    # Holm across the three bands, separately for each family
    for key in ("delta_L0", "delta_H", "d_L0", "d_H"):
        present = [b for b in BANDS if out[b].get(key)]
        adj = holm([out[b][key]["p"] for b in present])
        for b, p in zip(present, adj):
            out[b][key]["p_holm"] = p
    return out


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--traces", required=True, help="scored traces JSONL (needs a 'correct' field)")
    c.add_argument("--sae-report", required=True, help="sae_validation JSON; only passing layers used")
    c.add_argument("--out", required=True)
    c.add_argument("--max-traces", type=int, default=1000)
    c.add_argument("--layers-per-band", type=int, default=3)
    c.add_argument("--n-devices", type=int, default=1, help="split TL over this many GPUs")
    c.add_argument("--device", default=None)
    a_ = sub.add_parser("analyse")
    a_.add_argument("--tokens", required=True)
    a_.add_argument("--n-layers", type=int, required=True)
    a_.add_argument("--out", required=True)
    a_.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args()

    if a.cmd == "collect":
        from sae_lens import SAE

        from harness.config import model_config
        from harness.generate import read_jsonl
        from harness.models import load_hf, load_tl, load_tokenizer, pick_device
        from .sae_validation import sae_ids_for

        cfg = model_config(a.model)
        device = pick_device(a.device)
        passing = json.loads(Path(a.sae_report).read_text())["passing_layers"]
        layers = pick_layers(passing, cfg.n_layers, a.layers_per_band)
        tok = load_tokenizer(cfg)
        model = load_tl(cfg, hf_model=load_hf(cfg, device="cpu"), tokenizer=tok, device=device,
                        n_devices=a.n_devices)
        recs = read_jsonl(a.traces)[: a.max_traces]
        df = collect(model, lambda l: SAE.from_pretrained(*sae_ids_for(cfg, l), device=device),
                     layers, recs)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False, compression="gzip")
        print(f"wrote {len(df)} token rows for layers {layers}")
    else:
        df = pd.read_csv(a.tokens)
        res = analyse(df, a.n_layers, a.resamples)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
