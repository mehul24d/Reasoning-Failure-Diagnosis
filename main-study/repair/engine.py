"""Edited greedy continuations of recorded traces, with the exact divergence shortcut
(prereg/R1.md; ported from the October engine `rfd.steering.engine`, DECISIONS D11).

The edit: add vec[r] to the output of decoder block L (TL `blocks.L.hook_resid_post`) at every
ABSOLUTE position p with start <= p < end (the tokens of the first deviating step). Positions
before `start` are untouched, so a greedy run under the edit equals the recorded run up to the
first position where the edited argmax differs from the unedited one.

  scan()        one teacher-forced pass per edit over the recorded tokens from `start` on, with
                the KV cache of the unedited prefix (tokens < start) computed once and shared.
                Every batch carries one UNEDITED reference row, so a divergence is counted only
                when it is caused by the edit, never by batch-shape numerics (D11).
  regenerate()  greedy continuation from each divergence point, with the edit still active on
                its window, the S2 stop rule (end of the answer line, end of turn) and the S2 cap
                of 512 generated tokens counted from the end of the prompt.

Exactness against brute-force edited generation is tested in tests/test_repair_engine.py.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from harness.batched import answer_line_end
from parsing.answers import extract_final_answer, is_correct


class WindowAdd:
    """Forward hook: h[r, p] += vec[r] for start[r] <= p < end[r] (absolute, padded positions).

    `offset` is the absolute position of the first token of the next forward call; it advances
    by the call's sequence length, so one object serves a prefill and the cached decode steps."""

    def __init__(self, vecs: torch.Tensor, start: torch.Tensor, end: torch.Tensor, offset: int = 0):
        self.vecs, self.start, self.end = vecs, start, end
        self.offset0 = offset
        self.offset = offset

    def reset(self) -> None:
        self.offset = self.offset0

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        bs, seq, _ = h.shape
        pos = self.offset + torch.arange(seq, device=h.device)
        self.offset += seq
        mask = (pos[None, :] >= self.start.to(h.device)[:, None]) & \
               (pos[None, :] < self.end.to(h.device)[:, None])
        if not bool(mask.any()):
            return output
        new = h + mask[..., None].to(h.dtype) * self.vecs.to(h.device, h.dtype)[:, None, :]
        return (new, *output[1:]) if isinstance(output, tuple) else new


class _hooked:
    def __init__(self, model, layer: int, hook):
        self.layer, self.hook, self.model = layer, hook, model

    def __enter__(self):
        self.hook.reset()
        self.h = self.model.model.layers[self.layer].register_forward_hook(self.hook)

    def __exit__(self, *a):
        self.h.remove()


def lm_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """LM head in float32 with Gemma-2's final logit soft-cap (which the bare lm_head omits)."""
    logits = model.lm_head(hidden).float()
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _argmax(model, hidden: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    out = []
    for s in range(0, hidden.shape[1], chunk):
        out.append(lm_logits(model, hidden[:, s:s + chunk]).argmax(-1))
    return torch.cat(out, 1).cpu()


@dataclass
class Divergence:
    pos: int | None          # absolute position whose argmax first differs (it predicts pos + 1)
    token: int | None        # the edited argmax there


@torch.no_grad()
def scan(model, seq: list[int], start: int, end: int, layer: int, vecs: torch.Tensor,
         batch: int = 16) -> tuple[list[Divergence], dict]:
    """First divergence of each edit (rows of `vecs`) from the unedited reference.

    Only positions start .. len(seq) - 2 can diverge: the logits at p predict token p + 1, the
    edit starts at `start`, and the last recorded token ends the run."""
    dev = next(model.parameters()).device
    prefix = torch.tensor([seq[:start]], device=dev)
    suffix = seq[start:]
    out = model.model(input_ids=prefix, use_cache=True)
    cache = out.past_key_values
    res: list[Divergence] = []
    ref_match = None
    for b0 in range(0, len(vecs), batch - 1):
        V = vecs[b0:b0 + batch - 1]
        V = torch.cat([torch.zeros_like(V[:1]), V])                  # row 0 = unedited
        n = len(V)
        c = copy.deepcopy(cache)
        c.batch_repeat_interleave(n)
        ids = torch.tensor([suffix] * n, device=dev)
        hook = WindowAdd(V, torch.full((n,), start), torch.full((n,), end), offset=start)
        with _hooked(model, layer, hook):
            h = model.model(input_ids=ids, past_key_values=c, use_cache=True).last_hidden_state
        am = _argmax(model, h)                                        # [n, len(suffix)]
        ref = am[0]
        if ref_match is None:
            rec = torch.tensor(suffix[1:])
            ref_match = float((ref[:-1] == rec).float().mean()) if len(rec) else 1.0
        last = len(suffix) - 1                                        # positions j < last count
        for r in range(1, n):
            diff = (am[r, :last] != ref[:last]).nonzero()
            if len(diff):
                j = int(diff[0])
                res.append(Divergence(start + j, int(am[r, j])))
            else:
                res.append(Divergence(None, None))
        del c, h
    return res, {"reference_reproduces_recorded": ref_match}


class AnswerStop(StoppingCriteria):
    """Per-row stop at the end of a complete answer line, reading the text from the end of the
    chat prompt (so an answer line that began before the divergence point still counts)."""

    def __init__(self, tokenizer, text_start: list[int]):
        self.tok, self.text_start = tokenizer, text_start
        self.done = torch.zeros(len(text_start), dtype=torch.bool)
        self.stop_len: dict[int, int] = {}
        self._nl: dict[int, bool] = {}

    def _has_nl(self, t: int) -> bool:
        if t not in self._nl:
            self._nl[t] = "\n" in self.tok.decode([t], skip_special_tokens=True)
        return self._nl[t]

    def __call__(self, input_ids, scores, **kw):
        for i, t in enumerate(input_ids[:, -1].tolist()):
            if self.done[i] or not self._has_nl(t):
                continue
            text = self.tok.decode(input_ids[i, self.text_start[i]:].tolist(), skip_special_tokens=True)
            if answer_line_end(text) is not None:
                self.done[i] = True
                self.stop_len[i] = input_ids.shape[1]
        return self.done.to(input_ids.device).clone()


@dataclass
class Job:
    prompt_ids: list[int]
    prefix_out: list[int]     # recorded output tokens before the divergent token, + that token
    vec: torch.Tensor
    start: int                # absolute window, unpadded
    end: int


@torch.no_grad()
def regenerate(model, tokenizer, jobs: list[Job], layer: int, gen_cfg_fn, max_out: int = 512,
               eos: list[int] | None = None) -> list[dict]:
    """Greedy continuation of each job under its edit. `gen_cfg_fn(max_new)` builds the S2 greedy
    GenerationConfig. Returns output_ids (from the end of the prompt), stop reason, text."""
    dev = next(model.parameters()).device
    seqs = [j.prompt_ids + j.prefix_out for j in jobs]
    width = max(len(s) for s in seqs)
    pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    pads = [width - len(s) for s in seqs]
    ids = torch.tensor([[pad_tok] * p + s for p, s in zip(pads, seqs)], device=dev)
    mask = torch.tensor([[0] * p + [1] * len(s) for p, s in zip(pads, seqs)], device=dev)
    budgets = [max_out - len(j.prefix_out) for j in jobs]
    out_ids = [list(j.prefix_out) for j in jobs]
    live = [i for i, b in enumerate(budgets) if b > 0]
    results = [None] * len(jobs)
    if live:
        V = torch.stack([j.vec for j in jobs])
        start = torch.tensor([p + j.start for p, j in zip(pads, jobs)])
        end = torch.tensor([p + j.end for p, j in zip(pads, jobs)])
        stop = AnswerStop(tokenizer, [p + len(j.prompt_ids) for p, j in zip(pads, jobs)])
        gen_cfg = gen_cfg_fn(max(budgets))
        hook = WindowAdd(V, start, end, offset=0)
        with _hooked(model, layer, hook):
            g = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_cfg,
                               use_model_defaults=False,
                               stopping_criteria=StoppingCriteriaList([stop]))
        eos_set = set(gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list)
                      else [gen_cfg.eos_token_id])
        for i in range(len(jobs)):
            new = g[i, width:].tolist()
            cut = len(new)
            why = "max_new_tokens"
            e = next((k for k, t in enumerate(new) if t in eos_set), None)
            s = stop.stop_len.get(i)
            s = None if s is None else s - width - 1          # index of the stopping token
            cands = [(k, w) for k, w in ((e, "eos"), (s, "answer_line")) if k is not None]
            if cands:
                k, why = min(cands)
                cut = k + 1
            new = new[:cut]
            if len(new) > budgets[i]:                     # a shorter budget than the batch's
                new, why = new[:budgets[i]], "max_new_tokens"
            out_ids[i] = out_ids[i] + new
            results[i] = why
    rows = []
    for i, j in enumerate(jobs):
        why = results[i] or "max_new_tokens"
        text = tokenizer.decode(out_ids[i], skip_special_tokens=True)
        rows.append({"output_ids": out_ids[i], "stop_reason": why, "text": text})
    return rows


def outcome(text: str, stop_reason: str, gold, orig_pred) -> dict:
    pred = extract_final_answer(text)
    p = None if pred is None else str(pred)
    return {"pred": p, "correct": bool(is_correct(pred, gold)),
            "answer_changed": p != orig_pred,
            "broken": pred is None or stop_reason == "max_new_tokens"}
