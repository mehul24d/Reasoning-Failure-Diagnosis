"""Phase 0: check the HF hook harness against TransformerLens before trusting it.

Checks (thresholds fixed in docs/prereg/phase0.md):
  1. TL `blocks.L.hook_resid_post` == our HF decoder-layer hook, same tokens:
     median relative error ||h_tl - h_hf|| / ||h_hf|| < 0.01 at every checked layer.
  2. TL logits vs HF logits: top-1 agreement >= 0.99 over all positions.
  3. Left-padded batch vs single-sequence forward (HF): median relative error < 0.01.
  4. Greedy reproducibility: teacher-forced argmax reproduces >= 0.99 of the
     tokens produced by batched KV-cached generation.
  5. Generation config (amendment 2026-10-22): the checkpoint's sampling
     defaults are recorded, and the effective config must be pure greedy
     (do_sample False, repetition_penalty 1, no temperature / top-p / top-k).
  6. Chat prompts carry exactly one BOS token (Gemma's template adds its own).

Memory (Colab T4 + 12 GB host RAM, fp32 Gemma-2-2B = 10.5 GB): HF references
are computed first; the HF weights are then moved to the CPU as a bfloat16
copy when that copy is lossless (the checkpoint is stored in bf16), and the
TL model is allocated directly on the GPU in float32 from it. TL's float32
conversion of the scaled Gemma embedding is recomputed in float32 on the GPU,
because scaling a bf16 tensor would round it. TL's logits are computed from its
own final norm / unembedding / softcap in position chunks (`tl_logits`), since
TL's Unembed makes a 2.2 GiB transposed copy of W_U per call.

Usage: python -m rfd.experiments.phase0_equivalence [--config CFG] [--out results/phase0]
"""

from __future__ import annotations

import argparse
import gc
import json
import random

import numpy as np

from rfd.harness.model import library_versions, load  # sets MPS env before torch use

import torch  # noqa: E402

from rfd.data.gsm_symbolic import load_gsm_symbolic  # noqa: E402
from rfd.harness.generate import forward, generate  # noqa: E402
from rfd.harness.prompts import arith_messages, chat_ids, cot_messages  # noqa: E402
from rfd.utils.config import load_yaml  # noqa: E402
from rfd.utils.resume import StageMarker, atomic_write_json, results_path  # noqa: E402
from rfd.utils.seed import set_seed  # noqa: E402


def check_layers(n_layers: int) -> list[int]:
    return sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})


def _rel(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-6)).numpy()


def _empty_cache(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _lossless_bf16(model: torch.nn.Module) -> bool:
    return all(
        bool((p.data.to(torch.bfloat16).to(p.dtype) == p.data).all()) for p in model.parameters()
    )


def tl_from_hf(model: torch.nn.Module, tok, model_name: str, device: str):
    """TransformerLens model with the same weights as `model`, built with at most
    one full-precision copy in memory (the HF model is moved to the CPU; it is
    unusable afterwards). Mirrors HookedTransformer.from_pretrained_no_processing
    except that the TL parameters are allocated on `device` directly."""
    from transformer_lens import HookedTransformer
    from transformer_lens import loading_from_pretrained as loading

    lossless = model.dtype == torch.float32 and _lossless_bf16(model)
    dtype = torch.float32 if lossless else model.dtype
    if lossless:
        model.to(torch.bfloat16)
    model.to("cpu")
    _empty_cache(device)
    name = loading.get_official_model_name(model_name)
    tl_cfg = loading.get_pretrained_model_config(
        name, hf_cfg=model.config.to_dict(), fold_ln=False, device=device, dtype=dtype
    )
    # TL moves inputs to cfg.device; it must be where the weights are allocated below
    tl_cfg.device = device
    state_dict = loading.get_pretrained_state_dict(name, tl_cfg, model, dtype=dtype)
    with torch.device(device):
        tl = HookedTransformer(tl_cfg, tok, move_to_device=False)
    tl.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    del state_dict
    if lossless and model.config.model_type.startswith("gemma"):
        # TL computes W_E = embed * sqrt(d_model) in the SOURCE dtype; with a bf16
        # source that rounds, so redo it in float32 as TL would from fp32 weights
        emb = model.model.embed_tokens.weight
        mult = float(model.config.hidden_size) ** 0.5
        with torch.no_grad():
            for r0 in range(0, emb.shape[0], 16384):
                tl.embed.W_E[r0 : r0 + 16384] = emb[r0 : r0 + 16384].to(device).float() * mult
    return tl, ("bf16 copy (lossless)" if lossless else str(dtype))


def tl_logits(tl, resid_final: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """TL logits from the last block's output, using TL's own final norm,
    unembedding and softcap, applied to `chunk` positions at a time (CPU float32
    result). TL's Unembed.forward builds `W_U.T.contiguous()`, a 2.2 GiB copy for
    Gemma-2's 256k vocabulary, which does not fit next to the fp32 TL model on a
    T4. TL does not tie W_E and W_U, so that model alone takes ~11.9 GiB."""
    from transformer_lens.utilities.activation_functions import apply_softcap

    x = tl.ln_final(resid_final)
    out = []
    with torch.no_grad():
        for s0 in range(0, x.shape[-2], chunk):
            lg = x[..., s0 : s0 + chunk, :] @ tl.unembed.W_U + tl.unembed.b_U
            out.append(apply_softcap(lg, tl.cfg.output_logits_soft_cap).float().cpu())
    return torch.cat(out, -2)


def generation_config_report(model: torch.nn.Module) -> dict:
    from rfd.harness.model import SAMPLING_FIELDS

    gc_ = model.generation_config
    eff = {k: getattr(gc_, k, None) for k in SAMPLING_FIELDS}
    return {
        "checkpoint": getattr(model, "rfd_checkpoint_sampling", None),
        "effective": eff,
        "eos_token_id": gc_.eos_token_id,
        "pad_token_id": gc_.pad_token_id,
        "cache_implementation": getattr(gc_, "cache_implementation", None),
        "greedy_neutral": (
            eff["do_sample"] is False
            and eff["repetition_penalty"] in (None, 1.0)
            and all(eff[k] is None for k in ("temperature", "top_p", "top_k"))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gemma2_2b.yaml")
    ap.add_argument("--out", default="results/phase0")
    ap.add_argument("--force", action="store_true", help="re-run even if already complete")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = results_path(args.out)
    m = cfg["model"]
    stage = StageMarker(
        out, "phase0", f"{m['model_name']}|{m.get('dtype')}|{m.get('attn_implementation')}"
    )
    if stage.skip(args.force):
        return
    set_seed(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)

    loaded = load(cfg["model"])
    model, tok = loaded.model, loaded.tokenizer
    LAYERS = check_layers(model.config.num_hidden_layers)
    probs, _ = load_gsm_symbolic(cfg["cache_dir"])
    rng = random.Random(cfg["seed"])
    sample = rng.sample(probs, 8)
    seqs = [
        chat_ids(tok, cot_messages(p.question))
        + tok(p.answer_text, add_special_tokens=False)["input_ids"]
        for p in sample[:3]
    ]
    seqs += [chat_ids(tok, arith_messages(123, 456, "+", [(271, 318), (402, 197)]), "<<123+456=")]
    report: dict = {
        "model": cfg["model"]["model_name"],
        "dtype": str(model.dtype),
        "device": loaded.device,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "layers_checked": LAYERS,
        "generation_config": generation_config_report(model),
    }
    if tok.bos_token_id is not None:
        report["prompt_bos_counts"] = [s.count(tok.bos_token_id) for s in seqs]

    # ---- HF reference (bs=1, no padding). Full-vocab logits are kept only for the
    # last TAIL positions (a 256k vocabulary would not fit host RAM otherwise)
    TAIL = 32
    hf_resid, hf_top1, hf_tail = [], [], []
    for s in seqs:
        r = forward(model, tok, [s], LAYERS, [list(range(len(s)))])
        hf_resid.append(r["resid"][0])
        with torch.no_grad():
            lg = model(torch.tensor([s], device=model.device)).logits[0].float()
        hf_top1.append(lg.argmax(-1).cpu())
        hf_tail.append(lg[-TAIL:].cpu())
        del lg

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

    # ---- TransformerLens (see the module docstring for the memory plan)
    device = loaded.device
    if device == "cuda":
        report["hf_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    tl, report["tl_weight_source"] = tl_from_hf(model, tok, cfg["model"]["model_name"], device)
    del model
    loaded.model = None  # drop the CPU bf16 copy as well
    _empty_cache(device)
    if device == "cuda":
        report["tl_weights_gpu_gb"] = torch.cuda.memory_allocated() / 2**30
    names = [f"blocks.{layer}.hook_resid_post" for layer in LAYERS]
    last = f"blocks.{tl.cfg.n_layers - 1}.hook_resid_post"
    wanted = set(names) | {last}
    resid_err = {layer: [] for layer in LAYERS}
    top1 = n_pos = 0
    max_logit_diff = []
    for s, hr, ht1, htail in zip(seqs, hf_resid, hf_top1, hf_tail):
        with torch.no_grad():
            # return_type=None: TL stops after ln_final; logits via tl_logits (memory)
            _, cache = tl.run_with_cache(
                torch.tensor([s], device=device),
                names_filter=lambda n: n in wanted,
                return_type=None,
            )
        logits = tl_logits(tl, cache[last])[0]
        for j, layer in enumerate(LAYERS):
            resid_err[layer].append(_rel(cache[names[j]][0].float().cpu(), hr[j]))
        top1 += int((logits.argmax(-1) == ht1).sum())
        n_pos += len(s)
        max_logit_diff.append(float((logits[-TAIL:] - htail).abs().max()))
        del logits, cache
    report["tl_resid_rel_err_median"] = {
        str(k): float(np.median(np.concatenate(v))) for k, v in resid_err.items()
    }
    report["tl_logits_top1_agreement"] = top1 / n_pos
    report["tl_logits_max_abs_diff_last32"] = max(max_logit_diff)
    report["n_positions"] = n_pos

    report["pass"] = {
        "tl_resid": all(v < 0.01 for v in report["tl_resid_rel_err_median"].values()),
        "tl_logits_top1": report["tl_logits_top1_agreement"] >= 0.99,
        "padding": report["padding_rel_err_median"] < 0.01,
        "greedy_tf": report["greedy_tf_agreement"] >= 0.99,
        "generation_config_greedy": report["generation_config"]["greedy_neutral"],
        "single_bos": all(c == 1 for c in report.get("prompt_bos_counts", [1])),
    }
    if device == "cuda":
        report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    report["versions"] = library_versions()
    atomic_write_json(out / "equivalence.json", report)
    print(json.dumps(report, indent=2, default=str))
    stage.mark({"all_pass": all(report["pass"].values())})


if __name__ == "__main__":
    main()
