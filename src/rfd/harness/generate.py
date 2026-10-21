"""Batched greedy generation and teacher-forced forward passes.

Left padding throughout; `position_ids` are derived from the attention mask in
BOTH generation and teacher-forced forwards, so a token's activations do not
depend on how much padding its batch needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria

from rfd.harness.hooks import ResidCapture, hooked

_FINAL_LINE_RE = re.compile(r"Final Answer:[^\n]*\d[^\n]*\n")


@dataclass
class GenOutput:
    prompt_ids: list[int]
    gen_ids: list[int]
    text: str
    token_logprobs: list[float] = field(default_factory=list)  # chosen (=argmax) token
    token_entropies: list[float] = field(default_factory=list)


class _RecordStats(LogitsProcessor):
    """Log-prob of the greedy token and the predictive entropy at each step.
    Stores O(steps x bs) floats instead of full score tensors."""

    def __init__(self) -> None:
        self.logp: list[torch.Tensor] = []
        self.ent: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(scores.float(), dim=-1)
        self.logp.append(lp.max(-1).values.cpu())
        self.ent.append(-(lp.exp() * lp).sum(-1).cpu())
        return scores


class _StopAfterFinalAnswer(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        self.tok, self.prompt_len = tokenizer, prompt_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kw: Any) -> torch.Tensor:
        done = []
        for row in input_ids:
            tail = self.tok.decode(row[max(self.prompt_len, len(row) - 24) :])
            done.append(bool(_FINAL_LINE_RE.search(tail)))
        return torch.tensor(done, device=input_ids.device)


def _left_pad(seqs: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s) :] = torch.tensor(s, dtype=torch.long)
        mask[i, n - len(s) :] = 1
    return ids.to(device), mask.to(device)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    prefixes: list[list[int]],
    max_new_tokens: int,
    hooks: dict[int, Any] | None = None,
    stop_at_final_answer: bool = True,
    n_prompt_tokens: list[int] | None = None,
) -> list[GenOutput]:
    """Greedy-continue each token prefix. `hooks` (layer -> ResidEdit) use padded
    absolute positions of THIS batch: build them with `padded_offsets`.

    `n_prompt_tokens[i]` marks where the chat prompt ends inside prefix i (for
    continuations of teacher-forced partial traces); defaults to the full prefix.
    """
    device = model.device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    ids, mask = _left_pad(prefixes, pad_id, device)
    stats = _RecordStats()
    stopping = [_StopAfterFinalAnswer(tokenizer, ids.shape[1])] if stop_at_final_answer else []
    with hooked(model, hooks or {}):
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,  # checkpoint defaults must not alter greedy argmax
            pad_token_id=pad_id,
            logits_processor=LogitsProcessorList([stats]),
            stopping_criteria=stopping,
            use_cache=True,
        )
    new = out[:, ids.shape[1] :].cpu()
    logp = torch.stack(stats.logp, 1) if stats.logp else torch.zeros(len(prefixes), 0)
    ent = torch.stack(stats.ent, 1) if stats.ent else torch.zeros(len(prefixes), 0)
    eos = {tokenizer.eos_token_id, pad_id}
    if getattr(model.generation_config, "eos_token_id", None) is not None:
        e = model.generation_config.eos_token_id
        eos |= set(e if isinstance(e, list) else [e])
    results = []
    for i, prefix in enumerate(prefixes):
        row = new[i].tolist()
        n = len(row)
        for j, t in enumerate(row):
            if t in eos:
                n = j
                break
        gen = row[:n]
        npt = n_prompt_tokens[i] if n_prompt_tokens else len(prefix)
        full_gen = prefix[npt:] + gen  # generated part relative to the chat prompt
        results.append(
            GenOutput(
                prompt_ids=prefix[:npt],
                gen_ids=full_gen,
                text=tokenizer.decode(full_gen, skip_special_tokens=True),
                token_logprobs=logp[i, :n].tolist(),
                token_entropies=ent[i, :n].tolist(),
            )
        )
    return results


def padded_offsets(prefixes: list[list[int]]) -> torch.Tensor:
    """Left-pad amount per row for a batch built from `prefixes`."""
    n = max(len(s) for s in prefixes)
    return torch.tensor([n - len(s) for s in prefixes], dtype=torch.long)


@torch.no_grad()
def forward(
    model: Any,
    tokenizer: Any,
    seqs: list[list[int]],
    capture_layers: list[int] | None = None,
    capture_positions: list[list[int]] | None = None,
    hooks: dict[int, Any] | None = None,
    return_argmax: bool = False,
    chunk: int = 64,
) -> dict[str, Any]:
    """Teacher-forced forward of full token sequences.

    capture_positions[i] are UNPADDED indices into seqs[i]; returns `resid`
    [bs, n_layers_captured, k, d] (float32, CPU). With `return_argmax`, also
    `argmax` [bs, seq] (greedy next token at every UNPADDED position, -1 on
    padding) and `logp_next` [bs, seq] (log-prob of the actual next token),
    computed by applying the LM head in chunks so full-vocab logits for the
    whole batch never sit in memory at once.
    """
    device = model.device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    ids, mask = _left_pad(seqs, pad_id, device)
    off = padded_offsets(seqs)
    all_hooks: dict[int, list[Any]] = {k: [v] for k, v in (hooks or {}).items()}
    captures: dict[int, ResidCapture] = {}
    if capture_layers:
        k = max(len(p) for p in capture_positions)  # type: ignore[union-attr]
        pos = torch.zeros((len(seqs), k), dtype=torch.long)
        for i, p in enumerate(capture_positions):  # type: ignore[arg-type]
            pos[i, : len(p)] = torch.tensor(p, dtype=torch.long) + off[i]
        for layer in capture_layers:
            captures[layer] = ResidCapture(pos)
            all_hooks.setdefault(layer, []).append(captures[layer])
    with hooked(model, all_hooks):
        out = model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids_from_mask(mask),
            use_cache=False,
        )
    res: dict[str, Any] = {"pad_offsets": off}
    if captures:
        res["resid"] = torch.stack([captures[layer].result() for layer in capture_layers], 1)
    if return_argmax:
        hidden = out.last_hidden_state
        bs, n, _ = hidden.shape
        am = torch.full((bs, n), -1, dtype=torch.long)
        lpn = torch.zeros((bs, n))
        nxt = torch.cat([ids[:, 1:], ids[:, -1:]], 1)
        for s0 in range(0, n, chunk):
            logits = model.lm_head(hidden[:, s0 : s0 + chunk]).float()
            am[:, s0 : s0 + chunk] = logits.argmax(-1).cpu()
            lp = torch.log_softmax(logits, -1)
            lpn[:, s0 : s0 + chunk] = lp.gather(-1, nxt[:, s0 : s0 + chunk, None])[..., 0].cpu()
            del logits, lp
        # back to unpadded coordinates
        res["argmax"] = [am[i, off[i] :].tolist() for i in range(bs)]
        res["logp_next"] = [lpn[i, off[i] :].tolist() for i in range(bs)]
    return res
