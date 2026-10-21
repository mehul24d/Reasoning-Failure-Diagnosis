"""Train a JumpReLU SAE on residual-stream activations.

Two configs live in configs/sae.yaml:
  - `full_scale_training` (D=65536, L0~=64, trained on the full GSM8K train
    split): the Semester-7 target. NOT executed by this script yet --
    TODO, gated on more compute/time than a single pilot session has.
  - `pilot_training` (small D, few steps): what this script actually runs,
    on a small slice of the GSM8K *train* split (kept disjoint from the
    pilot *test* traces used for segmentation, to avoid even pilot-level
    train/test leakage in the learned dictionary).

Usage: python -m rfd.legacy.sae.train_sae --experiment configs/legacy/experiment.yaml
"""

from __future__ import annotations

import argparse
import gc
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rfd.data.loader import load_split
from rfd.generation.generate_traces import generate_one
from rfd.generation.model import load_model
from rfd.legacy.sae.activation_extraction import extract_trace_activations, load_hooked_model, resolve_layer
from rfd.legacy.sae.jumprelu_sae import JumpReLUSAE
from rfd.utils.config import load_experiment
from rfd.utils.io import ensure_parent, write_jsonl
from rfd.utils.seed import set_seed


def _generate_train_traces(experiment_cfg: dict[str, Any], n_problems: int) -> list[dict[str, Any]]:
    model_cfg = experiment_cfg["model"]
    dataset_cfg = experiment_cfg["dataset"]
    pilot_dataset = dataset_cfg["pilot"]["dataset"]

    problems = load_split(dataset_cfg, pilot_dataset, "train")[:n_problems]

    loaded = load_model(model_cfg)
    records = [generate_one(loaded, model_cfg, p) for p in tqdm(problems, desc="generating SAE-train traces")]

    # Free the HF model before loading the TransformerLens copy -- avoids
    # holding two copies of the model in memory at once on constrained RAM.
    del loaded
    gc.collect()
    return records


def _collect_training_matrix(
    experiment_cfg: dict[str, Any], train_records: list[dict[str, Any]]
) -> np.ndarray:
    model_cfg = experiment_cfg["model"]
    sae_cfg = experiment_cfg["sae"]
    hooked_model = load_hooked_model(model_cfg)
    layer = resolve_layer(sae_cfg["layers"][0], hooked_model.cfg.n_layers)
    hook_template = sae_cfg["hook_point_template"]

    chunks = []
    for record in tqdm(train_records, desc="extracting SAE-train activations"):
        acts = extract_trace_activations(
            hooked_model, record["prompt"], record["generated_trace"], layer, hook_template
        )
        if acts.shape[0] > 0:
            chunks.append(acts)

    del hooked_model
    gc.collect()
    return np.concatenate(chunks, axis=0)


def train_jumprelu_sae(activations: np.ndarray, pilot_training_cfg: dict[str, Any]) -> JumpReLUSAE:
    set_seed(pilot_training_cfg["seed"])
    device = "cpu"  # SAE training itself is cheap; keep on CPU for determinism
    x = torch.tensor(activations, dtype=torch.float32, device=device)

    sae = JumpReLUSAE(
        d_in=x.shape[1],
        d_sae=pilot_training_cfg["d_sae"],
        bandwidth=pilot_training_cfg["bandwidth"],
    ).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=pilot_training_cfg["learning_rate"])

    n_steps = pilot_training_cfg["n_training_steps"]
    batch_size = min(pilot_training_cfg["batch_size"], x.shape[0])
    n = x.shape[0]

    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,))
        batch = x[idx]
        out = sae.loss(batch, pilot_training_cfg["l1_coefficient"])
        optimizer.zero_grad()
        out["total"].backward()
        optimizer.step()
        if step % max(1, n_steps // 10) == 0:
            print(f"step {step}: mse={out['mse'].item():.4f} l0={out['l0'].item():.1f}")

    return sae


def run_sae_training(experiment_cfg: dict[str, Any], n_train_problems: int = 20) -> dict[str, Any]:
    train_records = _generate_train_traces(experiment_cfg, n_train_problems)
    write_jsonl(f"{experiment_cfg['output_dir']}/sae_train_traces.jsonl", train_records)

    activations = _collect_training_matrix(experiment_cfg, train_records)
    sae = train_jumprelu_sae(activations, experiment_cfg["sae"]["pilot_training"])

    checkpoint_path = ensure_parent(f"{experiment_cfg['output_dir']}/sae_checkpoint.pt")
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "bandwidth": sae.bandwidth,
        },
        checkpoint_path,
    )
    return {"checkpoint_path": str(checkpoint_path), "n_training_tokens": activations.shape[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/legacy/experiment.yaml")
    parser.add_argument("--n-train-problems", type=int, default=20)
    args = parser.parse_args()

    experiment_cfg = load_experiment(args.experiment)
    info = run_sae_training(experiment_cfg, args.n_train_problems)
    print(f"Saved pilot SAE checkpoint: {info}")


if __name__ == "__main__":
    main()
