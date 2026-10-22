"""Encode pilot traces' residual activations into SAE feature space, and
compute per-token L0 sparsity and feature entropy.

Reads traces_file (from generate_traces.py) and sae_checkpoint.pt (from
train_sae.py); writes one .npz per trace under sae_features_cache/ plus a
summary JSONL with per-trace sparsity/entropy arrays.

Usage: python -m rfd.legacy.sae.build_sae_features --experiment configs/legacy/experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rfd.legacy.metrics.sparsity import feature_entropy, l0_sparsity
from rfd.legacy.sae.activation_extraction import extract_trace_activations, load_hooked_model, resolve_layer
from rfd.legacy.sae.jumprelu_sae import JumpReLUSAE
from rfd.utils.config import load_experiment
from rfd.utils.io import read_jsonl, write_jsonl


def load_sae_checkpoint(checkpoint_path: str | Path) -> JumpReLUSAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae = JumpReLUSAE(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"], bandwidth=ckpt["bandwidth"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def run_feature_extraction(experiment_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = experiment_cfg["model"]
    sae_cfg = experiment_cfg["sae"]

    hooked_model = load_hooked_model(model_cfg)
    layer = resolve_layer(sae_cfg["layers"][0], hooked_model.cfg.n_layers)
    hook_template = sae_cfg["hook_point_template"]

    checkpoint_path = f"{experiment_cfg['output_dir']}/sae_checkpoint.pt"
    sae = load_sae_checkpoint(checkpoint_path)

    features_dir = Path(experiment_cfg["sae_features_cache"])
    features_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    traces = list(read_jsonl(experiment_cfg["traces_file"]))
    for record in tqdm(traces, desc="encoding SAE features"):
        resid = extract_trace_activations(
            hooked_model, record["prompt"], record["generated_trace"], layer, hook_template
        )
        if resid.shape[0] == 0:
            continue

        with torch.no_grad():
            out = sae.forward(torch.tensor(resid, dtype=torch.float32))
        feature_acts = out["features"].numpy()  # [n_tokens, d_sae]

        l0 = l0_sparsity(feature_acts)
        entropy = feature_entropy(feature_acts)

        npz_path = features_dir / f"{record['id']}.npz"
        np.savez_compressed(npz_path, feature_acts=feature_acts, l0=l0, entropy=entropy)

        summaries.append(
            {
                "id": record["id"],
                "n_tokens": int(feature_acts.shape[0]),
                "mean_l0": float(l0.mean()),
                "mean_entropy": float(entropy.mean()),
                "feature_file": str(npz_path),
            }
        )

    write_jsonl(f"{experiment_cfg['output_dir']}/sae_feature_summary.jsonl", summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/legacy/experiment.yaml")
    args = parser.parse_args()

    experiment_cfg = load_experiment(args.experiment)
    summaries = run_feature_extraction(experiment_cfg)
    mean_l0 = sum(s["mean_l0"] for s in summaries) / max(1, len(summaries))
    print(f"Encoded {len(summaries)} traces. Mean L0 across traces: {mean_l0:.2f}")


if __name__ == "__main__":
    main()
