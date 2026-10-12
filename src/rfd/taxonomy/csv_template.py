"""Generate a CSV with a `selected_for_manual_check` column flagging a
random `fraction` of rows (seeded) for human verification of the
heuristic `error_type` label, plus blank columns for the human to fill in."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from rfd.utils.io import ensure_parent


def write_taxonomy_csv(
    rows: list[dict[str, Any]], output_path: str | Path, fraction: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    n_select = max(1, int(round(len(df) * fraction))) if len(df) > 0 else 0

    rng = random.Random(seed)
    selected_ids = set(rng.sample(list(df["id"]), n_select)) if n_select else set()

    df["selected_for_manual_check"] = df["id"].isin(selected_ids)
    df["human_error_type"] = ""
    df["human_notes"] = ""

    path = ensure_parent(output_path)
    df.to_csv(path, index=False)
    return df
