"""Evaluate a residual edit on recorded greedy traces, with the divergence shortcut.

Greedy decoding is deterministic, so a steered run equals the recorded run up
to the first position where the steered argmax differs from the unsteered
one. A steered and an unsteered teacher-forced pass over the same batch
(identical padding, so fp16 near-ties cancel) find that position for every
trace; only traces that diverge are re-generated, from the divergence point
on, with the edit still active. Exactness against brute-force steered
generation is checked in tests/test_steering.py.

Edit placement:
  where="gen": every position from the last prompt token on (steering
               during generation; the first generated token is affected)
  where="at":  only the token at generated index `at_gen_index[i]` of trace i
               (Phase 3: first token of the first deviating step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rfd.generation.answer_extraction import extract_predicted_answer, is_correct
from rfd.harness.generate import forward, generate, padded_offsets
from rfd.harness.hooks import ResidEdit


@dataclass
class SteerSpec:
    layer: int
    mode: str  # add | project | set
    vec: torch.Tensor  # [d] shared, or [n_traces, d] per trace
    strength: float = 1.0
    where: str = "gen"  # gen | at
    at_gen_index: list[int] | None = None
    active: list[bool] | None = None  # gate per trace (None = all active)
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


_BASE_CACHE: dict[tuple, list[list[int]]] = {}


def _base_argmax(
    model: Any, tok: Any, batch: list[dict[str, Any]], seqs: list[list[int]]
) -> list[list[int]]:
    """Unsteered teacher-forced argmax for this exact batch (cached by trace uids)."""
    key = tuple(t.get("uid") for t in batch) if all("uid" in t for t in batch) else None
    if key is not None and key in _BASE_CACHE:
        return _BASE_CACHE[key]
    am = forward(model, tok, seqs, return_argmax=True)["argmax"]
    if key is not None:
        _BASE_CACHE[key] = am
    return am


@dataclass
class SteerResult:
    text: str
    pred: str | None
    correct: bool
    diverged: bool
    div_gen_index: int | None  # generated-token index of the first changed token


def _edit_for(
    spec: SteerSpec, rows: list[int], seqs: list[list[int]], prompt_lens: list[int]
) -> ResidEdit:
    off = padded_offsets(seqs)
    vec = spec.vec if spec.vec.dim() == 1 else spec.vec[rows]
    active = None if spec.active is None else torch.tensor([spec.active[r] for r in rows])
    if spec.where == "gen":
        start = torch.tensor([prompt_lens[k] - 1 for k in range(len(rows))]) + off
        return ResidEdit(spec.mode, vec, spec.strength, start=start, active=active)
    at = torch.tensor([prompt_lens[k] + spec.at_gen_index[r] for k, r in enumerate(rows)]) + off
    return ResidEdit(spec.mode, vec, spec.strength, at=at, active=active)


def _first_edit_index(spec: SteerSpec, r: int) -> int:
    """Generated index from which outputs can change (logits at position P-1+k predict gen token k)."""
    return 0 if spec.where == "gen" else spec.at_gen_index[r] + 1  # type: ignore[index]


@torch.no_grad()
def run_condition(
    model: Any,
    tok: Any,
    traces: list[dict[str, Any]],
    spec: SteerSpec,
    max_new_tokens: int,
    batch_size: int = 8,
    eos_ids: set[int] | None = None,
) -> list[SteerResult]:
    """Steered greedy outputs for every trace (dicts need prompt_ids, gen_ids, text,
    gold_answer, correct, pred)."""
    eos_ids = eos_ids or {tok.eos_token_id}
    results: list[SteerResult | None] = [None] * len(traces)
    diverged: list[tuple[int, int, int]] = []  # (row, gen index k, new token)
    for b0 in range(0, len(traces), batch_size):
        rows = list(range(b0, min(b0 + batch_size, len(traces))))
        if spec.active is not None and not any(spec.active[r] for r in rows):
            for r in rows:
                t = traces[r]
                results[r] = SteerResult(t["text"], t["pred"], t["correct"], False, None)
            continue
        seqs = [traces[r]["prompt_ids"] + traces[r]["gen_ids"] for r in rows]
        plens = [len(traces[r]["prompt_ids"]) for r in rows]
        edit = _edit_for(spec, rows, seqs, plens)
        am = forward(model, tok, seqs, hooks={spec.layer: edit}, return_argmax=True)["argmax"]
        # unsteered reference with IDENTICAL batch layout: fp16 near-ties that differ
        # between batched generation and teacher forcing cancel out, so only
        # steering-induced changes count as divergence
        base = _base_argmax(model, tok, [traces[r] for r in rows], seqs)
        for k, r in enumerate(rows):
            t = traces[r]
            gen, P = t["gen_ids"], plens[k]
            first_k = None
            for g in range(_first_edit_index(spec, r), len(gen)):
                if am[k][P - 1 + g] != base[k][P - 1 + g]:
                    first_k = g
                    break
            # past the recorded end (only matters if generation did not stop on a
            # complete answer line)
            if first_k is None and not t.get("has_final_marker", False):
                if (
                    am[k][P - 1 + len(gen)] != base[k][P - 1 + len(gen)]
                    and len(gen) < max_new_tokens
                ):
                    first_k = len(gen)
            if first_k is None:
                results[r] = SteerResult(t["text"], t["pred"], t["correct"], False, None)
            else:
                diverged.append((r, first_k, am[k][P - 1 + first_k]))
    # regenerate diverged traces from the divergence point, edit still active
    for b0 in range(0, len(diverged), batch_size):
        chunk = diverged[b0 : b0 + batch_size]
        rows = [r for r, _, _ in chunk]
        prefixes = [
            traces[r]["prompt_ids"] + traces[r]["gen_ids"][:g] + [new] for r, g, new in chunk
        ]
        plens = [len(traces[r]["prompt_ids"]) for r in rows]
        edit = _edit_for(spec, rows, prefixes, plens)
        budgets = [max_new_tokens - (g + 1) for _, g, _ in chunk]
        outs = generate(
            model,
            tok,
            prefixes,
            max_new_tokens=max(1, max(budgets)),
            hooks={spec.layer: edit},
            n_prompt_tokens=plens,
        )
        for (r, g, _), o, budget in zip(chunk, outs, budgets):
            gen_ids = o.gen_ids[: g + 1 + max(budget, 0)]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            pred = extract_predicted_answer(text)
            results[r] = SteerResult(
                text, pred, is_correct(pred, str(traces[r]["gold_answer"])), True, g
            )
    return results  # type: ignore[return-value]


def random_directions(d: int, n: int, norm: float, seed: int) -> torch.Tensor:
    """n Gaussian directions in raw residual space, each rescaled to `norm`."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    return v / v.norm(dim=1, keepdim=True) * norm


def outcome_rates(orig_correct: np.ndarray, new_correct: np.ndarray) -> dict[str, float]:
    """Repair = wrong->right among originally wrong; damage = right->wrong among
    originally right; net = accuracy change on this set."""
    o, n = np.asarray(orig_correct, bool), np.asarray(new_correct, bool)
    rep = float((~o & n).sum() / max((~o).sum(), 1))
    dmg = float((o & ~n).sum() / max(o.sum(), 1))
    return {"repair": rep, "damage": dmg, "net": float(n.mean() - o.mean())}
