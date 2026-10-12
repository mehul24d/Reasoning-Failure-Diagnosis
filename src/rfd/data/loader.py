"""Config-driven dataset loading for GSM8K (primary), MATH, and BBH Logic.

Splits are fixed slices (by index) of the underlying HuggingFace split, as
specified in configs/dataset.yaml, so runs are reproducible across machines
without needing to re-shuffle with a seed at load time.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from rfd.data.schema import Problem

_GSM8K_GOLD_RE = re.compile(r"####\s*(.+)")
_MATH_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_gold_gsm8k(raw_answer: str) -> str:
    match = _GSM8K_GOLD_RE.search(raw_answer)
    if not match:
        raise ValueError(f"Could not find '####' gold answer in: {raw_answer!r}")
    return match.group(1).strip().replace(",", "")


def _extract_gold_math(raw_solution: str) -> str:
    matches = _MATH_BOXED_RE.findall(raw_solution)
    if not matches:
        return raw_solution.strip()
    return matches[-1].strip()


def _extract_gold_bbh(raw_target: str) -> str:
    return raw_target.strip()


_GOLD_EXTRACTORS: dict[str, Callable[[str], str]] = {
    "gsm8k": _extract_gold_gsm8k,
    "math": _extract_gold_math,
    "bbh_logic": _extract_gold_bbh,
}


def load_split(dataset_cfg: dict[str, Any], dataset_name: str, split: str) -> list[Problem]:
    """Load one fixed split of `dataset_name` as a list of `Problem`.

    Requires network access on first call (HF Hub download + local cache).
    """
    from datasets import load_dataset

    ds_cfg = dataset_cfg["datasets"][dataset_name]
    split_cfg = ds_cfg["splits"][split]

    hf_dataset = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_config"))[
        split_cfg["source_split"]
    ]
    start, end = split_cfg["start"], split_cfg["end"]
    subset = hf_dataset.select(range(start, min(end, len(hf_dataset))))

    extractor = _GOLD_EXTRACTORS[dataset_name]
    question_field = ds_cfg["question_field"]
    answer_field = ds_cfg["answer_field"]

    problems = []
    for i, row in enumerate(subset):
        problems.append(
            Problem(
                id=f"{dataset_name}-{split}-{i}",
                dataset=dataset_name,
                split=split,
                question=row[question_field],
                gold_answer=extractor(row[answer_field]),
            )
        )
    return problems


def load_pilot(dataset_cfg: dict[str, Any]) -> list[Problem]:
    """Load the pilot slice specified under `dataset_cfg["pilot"]`."""
    pilot_cfg = dataset_cfg["pilot"]
    problems = load_split(dataset_cfg, pilot_cfg["dataset"], pilot_cfg["split"])
    return problems[: pilot_cfg["n_problems"]]
