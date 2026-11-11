"""SAE validation per layer (plan Section 3.12.2) — milestone M0, step S1c.

For every layer with an SAE, over the tokens of generated reasoning traces:
    FVU   = sum_t ||h_t - h^_t||^2 / sum_t ||h_t - mean(h)||^2              (eq. 2)
    dCE   = mean_t [ -log p_spliced(y_{t+1}) + log p(y_{t+1}) ]             (eq. 3)
A layer passes if FVU <= 0.40 and dCE <= 0.20 nats/token (configs/experiment.yaml). Both values
are reported for every layer, pass or fail.

Conventions (to be stated in prereg/E0.md):
  * hook site: blocks.{l}.hook_resid_post (Gemma Scope residual SAEs are trained there);
  * the BOS position is never spliced or scored (Gemma Scope excluded BOS in training);
  * T = generated-trace tokens only (not the prompt); the splice is applied at every non-BOS
    position, and CE is scored on the trace tokens.

Usage (GPU server):
    python -m analysis.sae_validation --model gemma-2-2b-it \
        --traces data/generated/gemma-2-2b-it/s1_validation.jsonl --out reports/m0/sae_validation_gemma-2-2b-it.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def trace_tokens(rec: dict) -> tuple[list[int], int]:
    """Full token sequence and the index where the generated trace starts."""
    return rec["prompt_ids"] + rec["output_ids"], len(rec["prompt_ids"])


@dataclass
class LayerAccumulator:
    """Streaming sums so FVU uses the global mean over all of T without storing activations."""
    sq_err: float = 0.0
    sum_h: torch.Tensor | None = None
    sum_sq_norm: float = 0.0
    n_tokens: int = 0
    dce_sum: float = 0.0
    n_ce: int = 0
    per_trace_dce: list = field(default_factory=list)

    def add_recon(self, h: torch.Tensor, h_hat: torch.Tensor) -> None:
        h, h_hat = h.double(), h_hat.double()
        self.sq_err += float(((h - h_hat) ** 2).sum())
        s = h.sum(0)
        self.sum_h = s if self.sum_h is None else self.sum_h + s
        self.sum_sq_norm += float((h ** 2).sum())
        self.n_tokens += h.shape[0]

    def add_ce(self, dce: torch.Tensor) -> None:
        self.dce_sum += float(dce.double().sum())
        self.n_ce += dce.numel()
        self.per_trace_dce.append(float(dce.double().mean()))

    def fvu(self) -> float:
        # sum ||h - mean||^2 = sum ||h||^2 - n ||mean||^2
        mean = self.sum_h / self.n_tokens
        total_var = self.sum_sq_norm - self.n_tokens * float((mean ** 2).sum())
        return self.sq_err / total_var

    def dce(self) -> float:
        return self.dce_sum / self.n_ce


@torch.no_grad()
def validate_layer(model, sae, records: list[dict], layer: int, bos_id: int | None) -> dict:
    """`model`: HookedTransformer. `sae`: object with encode(x) and decode(f) (SAELens SAE)."""
    name = hook_name(layer)
    acc = LayerAccumulator()
    sae_param = next(sae.parameters(), None) if hasattr(sae, "parameters") else None
    sae_dtype = sae_param.dtype if sae_param is not None else torch.float32
    sae_device = sae_param.device if sae_param is not None else torch.device("cpu")
    for rec in records:
        ids, start = trace_tokens(rec)
        if len(ids) - start < 1:
            continue
        assert start >= 1, "trace must follow a non-empty prompt"
        toks = torch.tensor([ids], device=model.cfg.device)
        # kept on CPU; moved to whichever GPU holds this layer when TL is split (n_devices > 1)
        splice_mask = torch.ones(len(ids), dtype=torch.bool)
        if bos_id is not None and ids[0] == bos_id:
            splice_mask[0] = False

        clean_logits, cache = model.run_with_cache(toks, names_filter=name)
        h = cache[name][0]                                   # [seq, d]
        h_hat = sae.decode(sae.encode(h.to(sae_device, sae_dtype))).to(h.device, h.dtype)
        acc.add_recon(h[start:], h_hat[start:])

        def splice(act, hook):
            out = act.clone()
            m = splice_mask.to(act.device)
            out[0, m] = h_hat.to(act.device, act.dtype)[m]
            return out

        spliced_logits = model.run_with_hooks(toks, fwd_hooks=[(name, splice)])
        # positions start-1 .. len-2 predict trace tokens start .. len-1
        tgt = toks[0, start:].to(clean_logits.device)
        lp_clean = F.log_softmax(clean_logits[0, start - 1:-1].float(), -1).gather(1, tgt[:, None])[:, 0]
        lp_splice = F.log_softmax(spliced_logits[0, start - 1:-1].float(), -1).gather(1, tgt[:, None])[:, 0]
        acc.add_ce(lp_clean - lp_splice)                     # = -log p_spliced + log p
    return {"layer": layer, "fvu": acc.fvu(), "delta_ce": acc.dce(), "n_tokens": acc.n_tokens,
            "n_traces": len(acc.per_trace_dce)}


def judge(row: dict, fvu_max: float, dce_max: float) -> dict:
    row = dict(row)
    row["pass_fvu"] = row["fvu"] <= fvu_max
    row["pass_dce"] = row["delta_ce"] <= dce_max
    row["pass"] = row["pass_fvu"] and row["pass_dce"]
    return row


def sae_ids_for(model_cfg, layer: int) -> tuple[str, str]:
    """(release, sae_id): instruction-tuned SAE where available, pre-trained elsewhere."""
    s = model_cfg.sae
    if s.get("it_release") and layer in s.get("it_layers", []):
        return s["it_release"], s["it_id_pattern"].format(layer=layer)
    return s["release"], s["id_pattern"].format(layer=layer)


def run(model_key: str, traces: str, out: str, layers: list[int] | None = None,
        device: str | None = None, max_traces: int | None = None, n_devices: int = 1) -> dict:
    from sae_lens import SAE

    from harness.config import experiment_config, model_config
    from harness.generate import read_jsonl
    from harness.models import load_hf, load_tl, load_tokenizer, pick_device

    cfg = model_config(model_key)
    if not cfg.sae:
        raise ValueError(f"{model_key} has no SAEs (plan Table 12)")
    thr = experiment_config()["thresholds"]
    device = pick_device(device)
    tok = load_tokenizer(cfg)
    hf = load_hf(cfg, device="cpu")  # only the TL copy goes on the GPU
    model = load_tl(cfg, hf_model=hf, tokenizer=tok, device=device, n_devices=n_devices)
    del hf
    records = read_jsonl(traces)[:max_traces]
    layers = layers if layers is not None else list(range(cfg.n_layers))
    rows = []
    for layer in layers:
        release, sae_id = sae_ids_for(cfg, layer)
        sae = SAE.from_pretrained(release, sae_id, device=device)
        row = validate_layer(model, sae, records, layer, tok.bos_token_id)
        row.update({"release": release, "sae_id": sae_id})
        rows.append(judge(row, thr["sae_fvu_max"], thr["sae_delta_ce_max"]))
        print(f"layer {layer:2d}  FVU {row['fvu']:.3f}  dCE {row['delta_ce']:.3f}  "
              f"{'PASS' if rows[-1]['pass'] else 'FAIL'}", flush=True)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {"model": model_key, "traces": traces, "n_traces": len(records),
              "thresholds": {"fvu_max": thr["sae_fvu_max"], "delta_ce_max": thr["sae_delta_ce_max"]},
              "layers": rows, "passing_layers": [r["layer"] for r in rows if r["pass"]]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-traces", type=int, default=None)
    ap.add_argument("--n-devices", type=int, default=1, help="split TL over this many GPUs")
    a = ap.parse_args()
    run(a.model, a.traces, a.out, a.layers, a.device, a.max_traces, a.n_devices)


if __name__ == "__main__":
    main()
