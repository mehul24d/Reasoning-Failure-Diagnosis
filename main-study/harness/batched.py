"""Batched generation for the minor project (prereg/S2.md amendments A2, A3, A5; DECISIONS D22-D24).

Hugging Face `generate` with a KV cache, float32, eager attention, LEFT padding and pure greedy
decoding (or the T = 0.7 sampled subset). TransformerLens is not used here: the verified TL
forward pass runs afterwards on the finished traces only (cache/verify_cache.py), and its
teacher-forced logits must reproduce these tokens (the pre-registered equivalence check).

Stop rule (amendment A3): a row stops at its end-of-turn token, at max_new_tokens (512), or at
the end of its answer line, whichever comes first. The answer line is a line that starts with
the answer marker (`parsing.steps._ANSWER_LINE`: "Answer:", "Final Answer:", "**Answer:**") and
has a number after the marker (`parsing.answers.NUMBER`); it ends at its newline. A trace that
reaches max_new_tokens without either is truncated and scored as a failure, as before.

For every greedy step the HF top-2 logits are recorded (a LogitsProcessor that reads the scores
and returns them unchanged), so the TL check can report the exact logit difference at the HF
token and the runner-up, and the margin of any near-tie.

Fallback (amendment A2): if the equivalence check fails, `engine="tl"` generates greedily with
the TransformerLens model itself (batch 1, TL key-value cache) and the same stop rule.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor, StoppingCriteria

from parsing.answers import NUMBER
from parsing.steps import _ANSWER_LINE

STOP_EOS, STOP_ANSWER, STOP_MAX = "eos", "answer_line", "max_new_tokens"


def answer_line_end(text: str) -> int | None:
    """Index just past the newline that ends the first complete answer line, or None.

    "Complete" means the line has its newline; a line still being written does not count, and
    neither does an answer line with no number after the marker (the scoring rule would find no
    answer on it, so the model is allowed to continue).
    """
    for m in _ANSWER_LINE.finditer(text):
        nl = text.find("\n", m.end())
        if nl < 0:
            return None
        if NUMBER.search(text[m.end():nl]):
            return nl + 1
    return None


class AnswerLineStop(StoppingCriteria):
    """Per-row stopping criterion (transformers >= 4.39 accepts a bool tensor per row).

    Only rows whose newest token contains a newline are decoded, so the cost is small.
    `stop_step[i]` is the 0-based index of the generated token at which row i stopped.
    """

    def __init__(self, tokenizer, prompt_width: int, batch: int):
        self.tok = tokenizer
        self.width = prompt_width
        self.stop_step: dict[int, int] = {}
        self.done = torch.zeros(batch, dtype=torch.bool)
        self._nl_ids: dict[int, bool] = {}

    def _has_newline(self, tid: int) -> bool:
        if tid not in self._nl_ids:
            self._nl_ids[tid] = "\n" in self.tok.decode([tid], skip_special_tokens=True)
        return self._nl_ids[tid]

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        step = input_ids.shape[1] - self.width - 1
        last = input_ids[:, -1].tolist()
        for i, tid in enumerate(last):
            if self.done[i] or not self._has_newline(tid):
                continue
            text = self.tok.decode(input_ids[i, self.width:].tolist(), skip_special_tokens=True)
            if answer_line_end(text) is not None:
                self.done[i] = True
                self.stop_step[i] = step
        return self.done.to(input_ids.device).clone()


class Top2Recorder(LogitsProcessor):
    """LogitsProcessor that records the top-2 logits per row and step, unchanged scores."""

    def __init__(self):
        self.vals: list[torch.Tensor] = []
        self.ids: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        v, i = scores.float().topk(2, dim=-1)
        self.vals.append(v)
        self.ids.append(i)
        return scores

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.vals:
            return None
        return torch.stack(self.vals, 1).cpu(), torch.stack(self.ids, 1).cpu()   # [B, steps, 2]


def eos_ids(tokenizer) -> list[int]:
    from .generate import _eos_ids

    return _eos_ids(tokenizer)


@dataclass
class RowResult:
    output_ids: list[int]
    stop_reason: str
    top2: dict | None = None            # {"l1": [...], "id2": [...], "l2": [...], "id1": [...]}
    extra: dict = field(default_factory=dict)


def _trim(gen: list[int], eos: set[int], stop_step: int | None, max_new: int) -> tuple[list[int], str]:
    """Cut a generated row after its first end-of-turn token or its answer-line stop."""
    cut_eos = next((j for j, t in enumerate(gen) if t in eos), None)
    cands = [(j, STOP_EOS) for j in [cut_eos] if j is not None]
    if stop_step is not None:
        cands.append((stop_step, STOP_ANSWER))
    if cands:
        j, why = min(cands)
        return gen[: j + 1], why
    return gen[:max_new], STOP_MAX


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[list[int]], gen_cfg, seed: int | None = None,
                   record_top2: bool = False, stop_on_answer: bool = True) -> list[RowResult]:
    """Left-padded batched generation. Returns one RowResult per prompt, in input order."""
    from transformers import LogitsProcessorList, StoppingCriteriaList

    device = next(model.parameters()).device
    pad = gen_cfg.pad_token_id
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    stop = AnswerLineStop(tokenizer, width, len(prompts))
    rec = Top2Recorder() if record_top2 else None
    if seed is not None:
        torch.manual_seed(seed)
    # use_model_defaults=False: see harness.generate.generate_one (Qwen repetition penalty).
    out = model.generate(
        input_ids=ids, attention_mask=mask, generation_config=gen_cfg, use_model_defaults=False,
        stopping_criteria=StoppingCriteriaList([stop]) if stop_on_answer else None,
        logits_processor=LogitsProcessorList([rec]) if rec else None)
    eos = set(gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id])
    top = rec.stacked() if rec else None
    results = []
    for i in range(len(prompts)):
        gen = out[i, width:].tolist()
        kept, why = _trim(gen, eos, stop.stop_step.get(i), gen_cfg.max_new_tokens)
        r = RowResult(kept, why)
        if top is not None:
            v, t = top[0][i, : len(kept)], top[1][i, : len(kept)]
            r.top2 = {"id1": t[:, 0].tolist(), "l1": v[:, 0].tolist(),
                      "id2": t[:, 1].tolist(), "l2": v[:, 1].tolist()}
        results.append(r)
    return results


@torch.no_grad()
def tl_generate_one(tl, tokenizer, prompt: list[int], max_new_tokens: int, eos: list[int],
                    stop_on_answer: bool = True) -> RowResult:
    """Fallback engine: greedy with the TransformerLens model (batch 1, no padding, KV cache)."""
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    cache = HookedTransformerKeyValueCache.init_cache(tl.cfg, tl.cfg.device, 1)
    toks = torch.tensor([prompt], device=tl.cfg.device)
    logits = tl(toks, past_kv_cache=cache)[0, -1]
    gen: list[int] = []
    l1, l2, id2 = [], [], []
    eos_set = set(eos)
    why = STOP_MAX
    for step in range(max_new_tokens):
        v, i = logits.float().topk(2)
        nxt = int(logits.argmax())
        gen.append(nxt)
        l1.append(float(v[0])), l2.append(float(v[1])), id2.append(int(i[1]))
        if nxt in eos_set:
            why = STOP_EOS
            break
        if stop_on_answer and "\n" in tokenizer.decode([nxt], skip_special_tokens=True) and \
                answer_line_end(tokenizer.decode(gen, skip_special_tokens=True)) is not None:
            why = STOP_ANSWER
            break
        logits = tl(torch.tensor([[nxt]], device=tl.cfg.device), past_kv_cache=cache)[0, -1]
    return RowResult(gen, why, {"id1": list(gen), "l1": l1, "id2": id2, "l2": l2})


def batch_seed(keys: list[tuple]) -> int:
    """Seed of a sampled batch: stable across processes (Python's hash() is salted per run)."""
    return zlib.crc32("||".join("|".join(map(str, k)) for k in keys).encode())


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def s(self) -> float:
        return time.time() - self.t0
