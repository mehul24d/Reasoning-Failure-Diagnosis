"""The six pipeline-validation checks of plan Table 5 (milestone M0, step S1).

1. TransformerLens vs HuggingFace residual streams, worst layer (relative error) < 0.01
2. Top-1 agreement of TL and HF logits >= 0.99
3. Left-padded batch vs single input, last-token logits (relative error) < 0.01
4. Teacher-forced argmax reproduces the greedy output >= 0.99 of tokens
5. Generation config is pure greedy (no penalty, temperature, top-p or top-k)
6. Exactly one BOS token per prompt (models that use BOS)

Usage (GPU server):
    python -m harness.equivalence --model gemma-2-2b-it --out reports/m0/equivalence_gemma-2-2b-it.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import experiment_config, model_config
from .generate import generate_one, greedy_generation_config, is_pure_greedy
from .models import load_hf, load_tl, load_tokenizer, pick_device
from .prompts import PromptSpec, count_bos, encode_chat

# Fixed probe problems for the checks (illustrative GSM-style; not part of any dataset split).
CHECK_PROBLEMS = [
    "Liam buys 4 boxes of pencils. Each box holds 12 pencils. He gives 9 pencils to his sister. "
    "How many pencils does Liam have left?",
    "A baker makes 36 muffins and sells them in packs of 4. Each pack costs $3. "
    "How much money does the baker make if all packs are sold?",
    "Maya reads 15 pages a day for 6 days, then 20 pages on the seventh day. "
    "How many pages does she read in the week?",
    "A train travels 120 km in 2 hours and then 90 km in 1.5 hours. "
    "What is its average speed in km per hour over the whole trip?",
]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    # compared on CPU in float64 so HF and TL may live on different GPUs
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@torch.no_grad()
def hf_reference(hf, ids_list):
    device = next(hf.parameters()).device
    refs = []
    for ids in ids_list:
        out = hf(torch.tensor([ids], device=device), output_hidden_states=True)
        refs.append({"hidden": [h[0].float().cpu() for h in out.hidden_states], "logits": out.logits[0].float().cpu()})
    return refs


@torch.no_grad()
def compare_with_tl(tl, ids_list, refs):
    n_layers = tl.cfg.n_layers
    worst = {}
    agree = total = 0
    for ids, ref in zip(ids_list, refs):
        tl_logits, cache = tl.run_with_cache(torch.tensor([ids]))
        hs = ref["hidden"]
        assert len(hs) == n_layers + 1, (len(hs), n_layers)
        errs = {"embed": rel_err(cache["blocks.0.hook_resid_pre"][0], hs[0])}
        for l in range(n_layers - 1):
            errs[f"L{l}"] = rel_err(cache[f"blocks.{l}.hook_resid_post"][0], hs[l + 1])
        final = tl.ln_final(cache[f"blocks.{n_layers - 1}.hook_resid_post"])[0]
        errs[f"L{n_layers - 1}(ln_final)"] = rel_err(final, hs[-1])
        errs["logits"] = rel_err(tl_logits[0], ref["logits"])
        for k, v in errs.items():
            worst[k] = max(worst.get(k, 0.0), v)
        agree += int((tl_logits[0].argmax(-1).cpu() == ref["logits"].argmax(-1)).sum())
        total += tl_logits.shape[1]
        del cache
    layer_errs = {k: v for k, v in worst.items() if k != "logits"}
    worst_layer = max(layer_errs, key=layer_errs.get)
    return ({"worst_layer": worst_layer, "worst_rel_err": layer_errs[worst_layer], "logits_rel_err": worst["logits"], "per_layer": layer_errs}, {"top1_agreement": agree / total, "tokens": total})


def check_residuals_and_logits(hf, tl, ids_list):
    return compare_with_tl(tl, ids_list, hf_reference(hf, ids_list))


@torch.no_grad()
def check_padding(hf, tokenizer, ids_list: list[list[int]]) -> float:
    """Left-pad two prompts of different length into one batch; compare with single runs."""
    device = next(hf.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    a, b = ids_list[0], ids_list[1]
    n = max(len(a), len(b))
    batch = torch.tensor([[pad] * (n - len(s)) + s for s in (a, b)], device=device)
    mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in (a, b)], device=device)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    batched = hf(batch, attention_mask=mask, position_ids=pos).logits[:, -1]
    worst = 0.0
    for i, s in enumerate((a, b)):
        single = hf(torch.tensor([s], device=device)).logits[0, -1]
        worst = max(worst, rel_err(batched[i], single))
    return worst


@torch.no_grad()
def check_teacher_forcing(hf, tokenizer, ids_list: list[list[int]], max_new_tokens: int) -> dict:
    device = next(hf.parameters()).device
    gen_cfg = greedy_generation_config(tokenizer, max_new_tokens)
    match = total = 0
    for ids in ids_list:
        g = generate_one(hf, tokenizer, ids, gen_cfg)
        full = torch.tensor([ids + g["output_ids"]], device=device)
        logits = hf(full).logits[0]
        pred = logits[len(ids) - 1: len(ids) - 1 + len(g["output_ids"])].argmax(-1).tolist()
        match += sum(int(p == t) for p, t in zip(pred, g["output_ids"]))
        total += len(g["output_ids"])
    return {"reproduction": match / max(total, 1), "tokens": total,
            "pure_greedy": is_pure_greedy(gen_cfg)}


def run(model_key: str, out: str, device: str | None = None, dtype: str | None = None,
        max_new_tokens: int = 256, tl_n_devices: int = 1) -> dict:
    thr = experiment_config()["equivalence_checks"]
    cfg = model_config(model_key)
    device = pick_device(device)
    tok = load_tokenizer(cfg)
    spec = PromptSpec("marked", 0)
    ids_list = [encode_chat(tok, p, spec) for p in CHECK_PROBLEMS]
    bos_counts = [count_bos(tok, ids) for ids in ids_list]
    hf = load_hf(cfg, device=device, dtype=dtype)
    refs = hf_reference(hf, ids_list)
    padded = check_padding(hf, tok, ids_list)
    tf = check_teacher_forcing(hf, tok, ids_list, max_new_tokens)
    hf.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tl = load_tl(cfg, hf_model=hf, tokenizer=tok, device=device, dtype=dtype, n_devices=tl_n_devices)
    del hf
    resid, top1 = compare_with_tl(tl, ids_list, refs)
    soft_caps = {"output_logits_soft_cap": getattr(tl.cfg, "output_logits_soft_cap", None), "attn_scores_soft_cap": getattr(tl.cfg, "attn_scores_soft_cap", None)}

    uses_bos = tok.bos_token_id is not None and any(c > 0 for c in bos_counts)
    checks = {
        "resid_tl_vs_hf": {"value": resid["worst_rel_err"], "layer": resid["worst_layer"],
                           "threshold": f"< {thr['resid_max_rel_err']}",
                           "pass": resid["worst_rel_err"] < thr["resid_max_rel_err"]},
        "top1_agreement": {"value": top1["top1_agreement"], "tokens": top1["tokens"],
                           "threshold": f">= {thr['top1_agreement_min']}",
                           "pass": top1["top1_agreement"] >= thr["top1_agreement_min"]},
        "padded_vs_single": {"value": padded, "threshold": f"< {thr['padded_vs_single_max_rel_err']}",
                             "pass": padded < thr["padded_vs_single_max_rel_err"]},
        "teacher_forced_reproduction": {"value": tf["reproduction"], "tokens": tf["tokens"],
                                        "threshold": f">= {thr['teacher_forced_reproduction_min']}",
                                        "pass": tf["reproduction"] >= thr["teacher_forced_reproduction_min"]},
        "pure_greedy_config": {"value": tf["pure_greedy"], "pass": tf["pure_greedy"]},
        "one_bos_per_prompt": {"value": bos_counts, "uses_bos": uses_bos,
                               "pass": (all(c == 1 for c in bos_counts) if uses_bos
                                        else all(c == 0 for c in bos_counts))},
    }
    report = {
        "model": model_key, "hf_id": cfg.hf_id, "device": device, "tl_n_devices": tl_n_devices, "tl_soft_caps": soft_caps,
        "dtype": dtype or cfg.dtype, "torch": torch.__version__,
        "all_pass": all(c["pass"] for c in checks.values()),
        "checks": checks, "per_layer_rel_err": resid["per_layer"],
        "logits_rel_err": resid["logits_rel_err"],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--tl-n-devices", type=int, default=1, help="split the TL copy over this many GPUs")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    r = run(a.model, a.out, a.device, a.dtype, a.max_new_tokens, a.tl_n_devices)
    for name, c in r["checks"].items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {name}: {c['value']}")
    print("ALL PASS" if r["all_pass"] else "SOME CHECKS FAILED")
    raise SystemExit(0 if r["all_pass"] else 1)


if __name__ == "__main__":
    main()
