"""The repair engine's divergence shortcut equals brute-force edited generation (tiny models).

Run on a tiny random Qwen2 and a tiny random Gemma-2 (the study architecture: soft-capping,
sliding-window layers, its cache class), both offline."""
from __future__ import annotations

import pytest
import torch

from repair.engine import Job, WindowAdd, _hooked, regenerate, scan
from tests.tiny_model import build


def _gemma(tok):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       max_position_embeddings=512, sliding_window=64, attn_implementation="eager",
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                       bos_token_id=tok.eos_token_id)
    torch.manual_seed(1)
    return Gemma2ForCausalLM(cfg).eval()


def _cfg(tok):
    from transformers import GenerationConfig

    def f(n):
        return GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                repetition_penalty=1.0, max_new_tokens=n,
                                eos_token_id=[tok.eos_token_id], pad_token_id=tok.pad_token_id)
    return f


def _brute(model, tok, prompt, layer, vec, start, end, max_out):
    hook = WindowAdd(vec[None], torch.tensor([start]), torch.tensor([end]), offset=0)
    from repair.engine import AnswerStop
    from transformers import StoppingCriteriaList

    stop = AnswerStop(tok, [len(prompt)])
    with _hooked(model, layer, hook):
        g = model.generate(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                           generation_config=_cfg(tok)(max_out), use_model_defaults=False,
                           stopping_criteria=StoppingCriteriaList([stop]))
    new = g[0, len(prompt):].tolist()
    if tok.eos_token_id in new:
        new = new[: new.index(tok.eos_token_id) + 1]
    return new


@pytest.mark.parametrize("arch", ["qwen2", "gemma2"])
def test_shortcut_equals_brute_force(arch):
    hf, tok = build(0)
    model = hf if arch == "qwen2" else _gemma(tok)
    layer, max_out = 1, 40
    words = "Solve the following problem step by step . Tom has 3 apples".split()
    prompts = [[tok.convert_tokens_to_ids(w) for w in words[: 6 + k]] for k in range(4)]
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    n_div = 0
    for prompt in prompts:
        # the "recorded" trace: unedited greedy output
        rec = regenerate(model, tok, [Job(prompt, [], torch.zeros(d), 0, 0)], layer, _cfg(tok), max_out)[0]
        out = rec["output_ids"]
        if len(out) < 8:
            continue
        seq = prompt + out
        start, end = len(prompt) + 3, len(prompt) + 7
        vecs = torch.randn(6, d, generator=g) * torch.tensor([0.0, 0.01, 1, 3, 10, 30])[:, None]
        divs, info = scan(model, seq, start, end, layer, vecs, batch=4)
        assert info["reference_reproduces_recorded"] == 1.0
        for v, dv in zip(vecs, divs):
            brute = _brute(model, tok, prompt, layer, v, start, end, max_out)
            if dv.pos is None:
                assert brute == out
                continue
            n_div += 1
            k = dv.pos + 1 - len(prompt)            # output index of the divergent token
            job = Job(prompt, out[:k] + [dv.token], v, start, end)
            got = regenerate(model, tok, [job], layer, _cfg(tok), max_out)[0]["output_ids"]
            assert got == brute
    assert n_div > 0


def test_batched_regenerate_matches_single():
    hf, tok = build(0)
    layer, max_out, d = 1, 30, hf.config.hidden_size
    words = "Solve the following problem step by step . Tom has 3 apples".split()
    g = torch.Generator().manual_seed(3)
    jobs = []
    for k in range(4):
        prompt = [tok.convert_tokens_to_ids(w) for w in words[: 5 + k]]
        jobs.append(Job(prompt, [], torch.randn(d, generator=g) * 5, len(prompt) - 1, len(prompt) + 4))
    single = [regenerate(hf, tok, [j], layer, _cfg(tok), max_out)[0]["output_ids"] for j in jobs]
    batched = [r["output_ids"] for r in regenerate(hf, tok, jobs, layer, _cfg(tok), max_out)]
    assert single == batched


def test_worker_edit_stage_end_to_end_and_resume(tmp_path):
    import argparse
    import json

    import numpy as np

    from repair import run as R

    hf, tok = build(0)
    d = hf.config.hidden_size
    words = "Solve the following problem step by step . Tom has 3 apples".split()
    items, vecs = [], []
    for k in range(3):
        prompt = [tok.convert_tokens_to_ids(w) for w in words[: 6 + k]]
        out = regenerate(hf, tok, [Job(prompt, [], torch.zeros(d), 0, 0)], 1, _cfg(tok), 30)[0]["output_ids"]
        role = "failure" if k < 2 else "solved"
        items.append({"item": k, "role": role, "split": "dev", "template_id": "t", "layer": 1, "r_v": 5.0,
                      "start": len(prompt) + 2, "end": len(prompt) + 6, "prompt_ids": prompt,
                      "output_ids": out, "gold_answer": "7", "orig_pred": None,
                      "orig_correct": role == "solved", "stop_reason": "max_new_tokens",
                      "gate_open": k == 0, "trace_id": f"tr{k}"})
        v = np.zeros((3, d), np.float32)
        if role == "failure":
            v[0] = np.eye(d)[k]
        if k == 0:
            v[1] = np.eye(d)[5]
        v[2] = np.eye(d)[7]
        vecs.append(v)
    vecs = np.stack(vecs)
    old = R.D_MODEL
    R.D_MODEL = d
    try:
        a = argparse.Namespace(out=str(tmp_path), tag="main", worker=0, n_workers=1, mults="1,4",
                               conds="oracle,self,eval,rand0-2", roles="failure,solved", splits="dev",
                               batch=4, gen_batch=4, deadline=0.0)
        t = R.stage_edit(a, items, vecs, hf, tok)
        rows = [json.loads(x) for x in (tmp_path / "main_w0.jsonl").read_text().splitlines()]
        assert len(rows) == 3 * 2 * 6
        # zero edits are the recorded trace: oracle on the solved item, self with a closed gate
        z = [r for r in rows if not r["edited"]]
        assert {(r["item"], r["cond"]) for r in z} == {(2, "oracle"), (1, "self"), (2, "self")}
        assert t["scan_rows"] == len(rows) - len(z)
        t2 = R.stage_edit(a, items, vecs, hf, tok)            # resume: nothing left to do
        assert t2["scan_rows"] == 0 and t2["trivial_rows"] == 0
    finally:
        R.D_MODEL = old
