"""Phase 0: check the HF hook harness against TransformerLens before trusting it.

Checks (thresholds fixed in docs/prereg/phase0.md):
  1. TL `blocks.L.hook_resid_post` == our HF decoder-layer hook, same tokens:
     median relative error ||h_tl - h_hf|| / ||h_hf|| < 0.01 at every checked layer.
  2. TL logits vs HF logits: top-1 agreement >= 0.99 over all positions.
  3. Left-padded batch vs single-sequence forward (HF): median relative error < 0.01.
  4. Greedy reproducibility: teacher-forced argmax reproduces >= 0.99 of the
     tokens produced by batched KV-cached generation.

Usage: python -m rfd.experiments.phase0_equivalence --out results/phase0
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np

from rfd.harness.model import library_versions, load  # sets MPS env before torch use

import torch  # noqa: E402

from rfd.data.gsm_symbolic import load_gsm_symbolic  # noqa: E402
from rfd.harness.generate import forward, generate  # noqa: E402
from rfd.harness.prompts import arith_messages, chat_ids, cot_messages  # noqa: E402
from rfd.utils.config import load_yaml  # noqa: E402
from rfd.utils.seed import set_seed  # noqa: E402


def check_layers(n_layers: int) -> list[int]:
    return sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})


def _rel(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-6)).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/harness.yaml")
    ap.add_argument("--out", default="results/phase0")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = load(cfg["model"])
    model, tok = loaded.model, loaded.tokenizer
    LAYERS = check_layers(model.config.num_hidden_layers)
    probs, _ = load_gsm_symbolic(cfg["cache_dir"])
    rng = random.Random(cfg["seed"])
    sample = rng.sample(probs, 8)
    seqs = [
        chat_ids(tok, cot_messages(p.question)) + tok(p.answer_text)["input_ids"]
        for p in sample[:3]
    ]
    seqs += [chat_ids(tok, arith_messages(123, 456, "+", [(271, 318), (402, 197)]), "<<123+456=")]
    report: dict = {
        "model": cfg["model"]["model_name"],
        "dtype": str(model.dtype),
        "device": loaded.device,
        "layers_checked": LAYERS,
    }

    # ---- HF reference (bs=1, no padding)
    hf_resid, hf_logits = [], []
    for s in seqs:
        r = forward(model, tok, [s], LAYERS, [list(range(len(s)))])
        hf_resid.append(r["resid"][0])
        with torch.no_grad():
            hf_logits.append(model(torch.tensor([s], device=model.device)).logits[0].float().cpu())

    # ---- check 3: padding invariance
    batched = forward(model, tok, seqs, LAYERS, [list(range(len(s))) for s in seqs])["resid"]
    pad_err = np.concatenate(
        [_rel(batched[i, :, : len(s)], hf_resid[i]).ravel() for i, s in enumerate(seqs)]
    )
    report["padding_rel_err_median"] = float(np.median(pad_err))
    report["padding_rel_err_p99"] = float(np.quantile(pad_err, 0.99))

    # ---- check 4: greedy reproducibility under teacher forcing
    prefixes = [chat_ids(tok, cot_messages(p.question)) for p in sample]
    gens = generate(model, tok, prefixes, max_new_tokens=cfg["generation"]["max_new_tokens"])
    full = [p + g.gen_ids for p, g in zip(prefixes, gens)]
    tf = forward(model, tok, full, return_argmax=True)
    agree = total = 0
    first_div = []
    for p, g, am in zip(prefixes, gens, tf["argmax"]):
        pred = am[len(p) - 1 : len(p) - 1 + len(g.gen_ids)]
        hits = [a == b for a, b in zip(pred, g.gen_ids)]
        agree += sum(hits)
        total += len(hits)
        first_div.append(hits.index(False) if False in hits else None)
    report["greedy_tf_agreement"] = agree / max(total, 1)
    report["greedy_tf_first_divergence"] = first_div
    report["greedy_tf_n_tokens"] = total

    # ---- TransformerLens
    model.to("cpu")
    if loaded.device == "mps":
        torch.mps.empty_cache()
    elif loaded.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    from transformer_lens import HookedTransformer

    tl = HookedTransformer.from_pretrained_no_processing(
        cfg["model"]["model_name"],
        hf_model=model,
        tokenizer=tok,
        dtype=model.dtype,
        device="cpu",
    )
    del model
    gc.collect()
    tl.to(loaded.device)
    names = [f"blocks.{layer}.hook_resid_post" for layer in LAYERS]
    resid_err = {layer: [] for layer in LAYERS}
    top1 = n_pos = 0
    max_logit_diff = []
    for s, hr, hl in zip(seqs, hf_resid, hf_logits):
        with torch.no_grad():
            logits, cache = tl.run_with_cache(
                torch.tensor([s], device=loaded.device), names_filter=lambda n: n in names
            )
        tl_logits = logits[0].float().cpu()
        for j, layer in enumerate(LAYERS):
            resid_err[layer].append(_rel(cache[names[j]][0].float().cpu(), hr[j]))
        top1 += int((tl_logits.argmax(-1) == hl.argmax(-1)).sum())
        n_pos += len(s)
        max_logit_diff.append(float((tl_logits - hl).abs().max()))
    report["tl_resid_rel_err_median"] = {
        str(k): float(np.median(np.concatenate(v))) for k, v in resid_err.items()
    }
    report["tl_logits_top1_agreement"] = top1 / n_pos
    report["tl_logits_max_abs_diff"] = max(max_logit_diff)
    report["n_positions"] = n_pos

    report["pass"] = {
        "tl_resid": all(v < 0.01 for v in report["tl_resid_rel_err_median"].values()),
        "tl_logits_top1": report["tl_logits_top1_agreement"] >= 0.99,
        "padding": report["padding_rel_err_median"] < 0.01,
        "greedy_tf": report["greedy_tf_agreement"] >= 0.99,
    }
    report["versions"] = library_versions()
    (out / "equivalence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
