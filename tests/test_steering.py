"""The divergence shortcut must equal brute-force steered greedy generation."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from rfd.harness.generate import generate, padded_offsets  # noqa: E402
from rfd.harness.hooks import ResidEdit  # noqa: E402
from rfd.steering.engine import SteerSpec, outcome_rates, random_directions, run_condition  # noqa: E402


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(1)
    cfg = transformers.Qwen2Config(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model, _Tok()


PROMPTS = [[5, 6, 7, 8, 9, 10], [11, 12, 13], [20, 21, 22, 23]]
N_NEW = 10


def _recorded(model, tok):
    outs = generate(model, tok, PROMPTS, N_NEW, stop_at_final_answer=False)
    return [
        {
            "prompt_ids": p,
            "gen_ids": o.gen_ids,
            "text": o.text,
            "pred": None,
            "correct": False,
            "gold_answer": 0,
            "has_final_marker": True,  # fixed-length runs: nothing past the record matters
        }
        for p, o in zip(PROMPTS, outs)
    ]


def _brute_gen(model, tok, vec, strength):
    off = padded_offsets(PROMPTS)
    start = torch.tensor([len(p) - 1 for p in PROMPTS]) + off
    edit = ResidEdit("add", vec, strength, start=start)
    outs = generate(model, tok, PROMPTS, N_NEW, hooks={1: edit}, stop_at_final_answer=False)
    return [o.gen_ids for o in outs]


@pytest.mark.parametrize("strength", [0.0, 0.5, 4.0])
def test_shortcut_equals_brute_force_gen_mode(tiny, strength):
    model, tok = tiny
    traces = _recorded(model, tok)
    vec = torch.randn(32, generator=torch.Generator().manual_seed(3))
    spec = SteerSpec(layer=1, mode="add", vec=vec, strength=strength, where="gen")
    res = run_condition(model, tok, traces, spec, N_NEW, batch_size=2)
    brute = _brute_gen(model, tok, vec, strength)
    for r, b in zip(res, brute):
        assert r.text == tok.decode(b)
    if strength == 0.0:
        assert not any(r.diverged for r in res)


def test_shortcut_equals_brute_force_single_position(tiny):
    model, tok = tiny
    traces = _recorded(model, tok)
    at = [2, 0, 4]
    vec = torch.randn(32, generator=torch.Generator().manual_seed(4)) * 5
    spec = SteerSpec(layer=0, mode="add", vec=vec, where="at", at_gen_index=at)
    res = run_condition(model, tok, traces, spec, N_NEW, batch_size=3)
    # brute force: teacher-force the recorded prefix through the edited token, then generate
    for i, (t, r) in enumerate(zip(traces, res)):
        prefix = t["prompt_ids"] + t["gen_ids"][: at[i] + 1]
        edit = ResidEdit("add", vec, at=torch.tensor([len(t["prompt_ids"]) + at[i]]))
        out = generate(
            model,
            tok,
            [t["prompt_ids"] + t["gen_ids"][: at[i]]],
            N_NEW - at[i],
            hooks={0: edit},
            stop_at_final_answer=False,
        )[0]
        # the edited token is the input at position P+at, so the edit changes
        # predictions from gen index at+1 on; the brute-force run re-derives it
        brute = t["gen_ids"][: at[i]] + out.gen_ids
        assert (
            prefix[: len(t["prompt_ids"]) + at[i]]
            == (t["prompt_ids"] + brute)[: len(t["prompt_ids"]) + at[i]]
        )
        assert r.text == tok.decode(brute[:N_NEW])


def test_gating_leaves_inactive_rows_untouched(tiny):
    model, tok = tiny
    traces = _recorded(model, tok)
    vec = torch.randn(32) * 10
    spec = SteerSpec(1, "add", vec, where="gen", active=[False, False, False])
    res = run_condition(model, tok, traces, spec, N_NEW)
    assert [r.text for r in res] == [t["text"] for t in traces]


def test_random_directions_and_rates():
    v = random_directions(16, 5, norm=3.0, seed=0)
    assert torch.allclose(v.norm(dim=1), torch.full((5,), 3.0))
    r = outcome_rates([True, True, False, False], [True, False, True, False])
    assert r == {"repair": 0.5, "damage": 0.5, "net": 0.0}
