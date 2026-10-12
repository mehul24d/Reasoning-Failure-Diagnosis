"""Trace collection harness: greedy-decodes each pilot problem and saves
{id, prompt, generated trace, predicted answer, gold answer, correct flag,
token list} to JSONL. Also usable as a script: `python -m
rfd.generation.generate_traces --experiment configs/experiment.yaml`.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from tqdm import tqdm

from rfd.data.loader import load_pilot
from rfd.data.schema import Problem
from rfd.generation.answer_extraction import extract_predicted_answer, is_correct
from rfd.generation.model import LoadedModel, build_prompt, load_model
from rfd.utils.config import load_experiment
from rfd.utils.io import write_jsonl
from rfd.utils.seed import set_seed


@torch.no_grad()
def generate_one(loaded: LoadedModel, model_cfg: dict[str, Any], problem: Problem) -> dict[str, Any]:
    prompt = build_prompt(loaded.tokenizer, model_cfg["prompt_template"], problem.question)
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    gen_cfg = model_cfg["generation"]

    output_ids = loaded.model.generate(
        **inputs,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        temperature=None if not gen_cfg["do_sample"] else gen_cfg["temperature"],
        pad_token_id=loaded.tokenizer.eos_token_id,
    )
    new_token_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    generated_text = loaded.tokenizer.decode(new_token_ids, skip_special_tokens=True)
    tokens = [loaded.tokenizer.decode([tid]) for tid in new_token_ids.tolist()]

    predicted = extract_predicted_answer(generated_text)
    correct = is_correct(predicted, problem.gold_answer)

    return {
        "id": problem.id,
        "dataset": problem.dataset,
        "prompt": prompt,
        "question": problem.question,
        "generated_trace": generated_text,
        "token_ids": new_token_ids.tolist(),
        "tokens": tokens,
        "predicted_answer": predicted,
        "gold_answer": problem.gold_answer,
        "correct": correct,
    }


def run_generation(experiment_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    set_seed(experiment_cfg["seed"])
    model_cfg = experiment_cfg["model"]
    dataset_cfg = experiment_cfg["dataset"]

    loaded = load_model(model_cfg)
    problems = load_pilot(dataset_cfg)

    records = [generate_one(loaded, model_cfg, p) for p in tqdm(problems, desc="generating traces")]
    write_jsonl(experiment_cfg["traces_file"], records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    args = parser.parse_args()

    experiment_cfg = load_experiment(args.experiment)
    records = run_generation(experiment_cfg)

    n_correct = sum(r["correct"] for r in records)
    print(f"Generated {len(records)} traces. Exact-match accuracy: {n_correct}/{len(records)}")


if __name__ == "__main__":
    main()
