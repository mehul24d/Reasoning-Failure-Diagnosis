"""Harness logic on a tiny random model (no downloads)."""
import json

import pytest
import torch

from harness import equivalence, generate
from harness.prompts import PromptSpec, all_prompt_specs, encode_chat

from .tiny_model import build, build_tl


@pytest.fixture(scope="module")
def tiny():
    hf, tok = build()
    return hf, tok, build_tl(hf, tok)


def _ids(tok):
    return [encode_chat(tok, p, PromptSpec("marked", 0)) for p in equivalence.CHECK_PROBLEMS[:2]]


def test_greedy_config_is_pure_and_ignores_model_defaults(tiny):
    hf, tok, _ = tiny
    hf.generation_config.repetition_penalty = 1.3   # mimic Qwen's shipped default
    hf.generation_config.temperature = 0.7
    g = generate.greedy_generation_config(tok, 16)
    assert generate.is_pure_greedy(g)
    assert g.repetition_penalty == 1.0 and not g.do_sample
    s = generate.sampled_generation_config(tok, 16, 0.7)
    assert not generate.is_pure_greedy(s)


def test_tl_hf_residuals_and_logits_match(tiny):
    hf, tok, tl = tiny
    resid, top1 = equivalence.check_residuals_and_logits(hf, tl, _ids(tok))
    assert resid["worst_rel_err"] < 1e-4, resid
    assert top1["top1_agreement"] == 1.0
    # embed + every layer is compared (last one through ln_final)
    assert len(resid["per_layer"]) == hf.config.num_hidden_layers + 1


def test_residual_check_detects_a_broken_model(tiny):
    hf, tok, _ = tiny
    bad_hf, _ = build(seed=1)  # different weights
    tl_bad = build_tl(bad_hf, tok)
    resid, _ = equivalence.check_residuals_and_logits(hf, tl_bad, _ids(tok))
    assert resid["worst_rel_err"] > 0.1


def test_padding_and_teacher_forcing(tiny):
    hf, tok, _ = tiny
    assert equivalence.check_padding(hf, tok, _ids(tok)) < 1e-4
    tf = equivalence.check_teacher_forcing(hf, tok, _ids(tok), max_new_tokens=20)
    assert tf["reproduction"] == 1.0 and tf["tokens"] > 0 and tf["pure_greedy"]


def test_no_bos_model_has_zero_bos(tiny):
    _, tok, _ = tiny
    for ids in _ids(tok):
        assert tok.bos_token_id is None or tok.bos_token_id not in ids


def test_generate_run_writes_resumes_and_is_deterministic(tiny, tmp_path, monkeypatch):
    hf, tok, _ = tiny
    monkeypatch.setattr(generate, "load_tokenizer", lambda cfg: tok)
    monkeypatch.setattr(generate, "load_hf", lambda cfg, device=None: hf)
    probs = tmp_path / "p.jsonl"
    probs.write_text("\n".join(json.dumps({"problem_id": f"t{i}", "template_id": "T", "variant": "base",
                                           "question": "Tom has 3 apples", "answer": 3}) for i in range(2)))
    out = tmp_path / "g.jsonl"
    generate.run("gemma-2-2b-it", str(probs), str(out), device="cpu")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2 * len(all_prompt_specs())
    generate.run("gemma-2-2b-it", str(probs), str(out), device="cpu")   # resume: nothing new
    assert len(out.read_text().splitlines()) == len(rows)
    # same prompt, greedy -> identical outputs across the two problems
    a = [r for r in rows if r["prompt"] == "marked_p0"]
    assert a[0]["output_ids"] == a[1]["output_ids"]
    # sampled mode: k samples per (problem, prompt), seeds stable
    out2 = tmp_path / "s.jsonl"
    generate.run("gemma-2-2b-it", str(probs), str(out2), mode="sampled",
                 specs=[PromptSpec("terse", 0)], device="cpu", limit=1)
    srows = [json.loads(l) for l in out2.read_text().splitlines()]
    assert [r["sample"] for r in srows] == [0, 1, 2, 3, 4]
    assert len({r["seed"] for r in srows}) == 5


def test_shipped_repetition_penalty_cannot_leak(tiny):
    """Regression: transformers merges model defaults into an explicit config unless told not to."""
    hf, tok, _ = tiny
    ids = _ids(tok)[0]
    g = generate.greedy_generation_config(tok, 20)
    hf.generation_config.repetition_penalty = 1.0
    clean = generate.generate_one(hf, tok, ids, g)["output_ids"]
    hf.generation_config.repetition_penalty = 1.3
    leaked = generate.generate_one(hf, tok, ids, g)["output_ids"]
    assert clean == leaked


def test_neutralize_generation_defaults(tiny):
    from harness.models import neutralize_generation_defaults

    hf, _, _ = tiny
    hf.generation_config.repetition_penalty = 1.3
    hf.generation_config.temperature = 0.7
    neutralize_generation_defaults(hf)
    assert hf.generation_config.repetition_penalty == 1.0
    assert hf.generation_config.temperature == 1.0
