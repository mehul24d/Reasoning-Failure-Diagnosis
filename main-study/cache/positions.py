"""Pre-registered read-out positions of a trace (configs/experiment.yaml `probing.positions`).

* step_first   (primary): the token holding the first non-blank character of a step
* step_last:              the token holding the last non-blank character of a step
* final_answer:           the token holding the first character of the extracted final answer
                          (its first digit, or its minus sign)

Steps are the character spans of the frozen parser (`parsing.steps.segment`, v4), which needs
no gold program, so positions exist for correct and failed traces alike. The final answer is
found with the scoring rule (`parsing.answers`): the last number on the line of the last answer
marker. A token's position is its absolute index in prompt + output, i.e. the residual stream
AFTER that token has been read.

Characters are mapped to tokens by decoding growing prefixes of the output ids. That is valid
only if every decoded prefix is a prefix of the decoded whole; if not, the trace gets no
positions and is counted (`char_map_ok = False`), never guessed.
"""
from __future__ import annotations

from parsing.answers import ANSWER_MARKER, NUMBER, extract_final_answer, normalise_number
from parsing.steps import segment

KINDS = ("step_first", "step_last", "final_answer")


def token_char_ends(tokenizer, ids: list[int]) -> list[int] | None:
    """ends[k] = length of the decoded text of ids[:k+1]; None if decoding is not prefix-stable."""
    full = tokenizer.decode(ids, skip_special_tokens=True)
    ends, prev = [], 0
    for k in range(len(ids)):
        t = tokenizer.decode(ids[: k + 1], skip_special_tokens=True)
        if len(t) < prev or not full.startswith(t):
            return None
        ends.append(len(t))
        prev = len(t)
    if prev != len(full):
        return None
    return ends


def char_to_token(ends: list[int], c: int) -> int:
    """Index of the token whose decoded text contains character c."""
    lo, hi = 0, len(ends) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ends[mid] > c:
            hi = mid
        else:
            lo = mid + 1
    return lo


def answer_char(text: str) -> int | None:
    """Character index of the final answer's first digit or sign (scoring rule of answers.py)."""
    if extract_final_answer(text) is None:
        return None
    m = list(ANSWER_MARKER.finditer(text))[-1]
    tail_start = m.end()
    tail = text[tail_start:]
    lead = len(tail) - len(tail.lstrip())
    line = tail.strip().split("\n", 1)[0]
    nums = list(NUMBER.finditer(line))
    last = nums[-1]
    assert normalise_number(last.group()) == extract_final_answer(text)
    off = next(i for i, ch in enumerate(last.group()) if ch.isdigit() or ch == "-")
    return tail_start + lead + last.start() + off


def trace_positions(tokenizer, rec: dict) -> dict:
    """Positions of one trace record (needs prompt_ids, output_ids, text, prompt)."""
    style = rec["prompt"].split("_")[0]
    text = rec["text"]
    n_prompt = len(rec["prompt_ids"])
    ends = token_char_ends(tokenizer, rec["output_ids"])
    out = {"trace_id": rec["trace_id"], "n_prompt": n_prompt, "n_output": len(rec["output_ids"]),
           "char_map_ok": ends is not None, "positions": []}
    if ends is None or not text:
        return out
    pos = out["positions"]
    for i, (s, e) in enumerate(segment(text, style)):
        seg = text[s:e]
        if not seg.strip():
            continue
        fs = s + (len(seg) - len(seg.lstrip()))
        le = s + len(seg.rstrip()) - 1
        pos.append({"kind": "step_first", "step": i, "char": fs, "tok": n_prompt + char_to_token(ends, fs)})
        pos.append({"kind": "step_last", "step": i, "char": le, "tok": n_prompt + char_to_token(ends, le)})
    a = answer_char(text)
    if a is not None:
        pos.append({"kind": "final_answer", "step": None, "char": a, "tok": n_prompt + char_to_token(ends, a)})
    ids = rec["prompt_ids"] + rec["output_ids"]
    for p in pos:
        p["token_id"] = ids[p["tok"]]
    return out
