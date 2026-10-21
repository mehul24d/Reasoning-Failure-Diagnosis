"""Hook/generation consistency on a tiny randomly initialised Qwen2 (no download, CPU)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from rfd.harness.generate import forward, generate, padded_offsets  # noqa: E402
from rfd.harness.hooks import ResidCapture, ResidEdit, hooked  # noqa: E402


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = transformers.Qwen2ForCausalLM(cfg).eval().to(torch.float32)
    model.generation_config.eos_token_id = None  # never stop early in tests
    model.generation_config.pad_token_id = 0
    return model, _Tok()


PREFIXES = [[5, 6, 7, 8, 9, 10], [11, 12, 13]]


def test_capture_matches_output_hidden_states(tiny):
    model, tok = tiny
    seq = [PREFIXES[0]]
    res = forward(model, tok, seq, capture_layers=[0, 1], capture_positions=[list(range(6))])
    with torch.no_grad():
        hs = model(torch.tensor(seq), output_hidden_states=True).hidden_states
    assert torch.allclose(res["resid"][0, 0], hs[1][0], atol=1e-5)
    assert torch.allclose(res["resid"][0, 1], hs[2][0], atol=1e-5)


def test_padding_does_not_change_activations(tiny):
    model, tok = tiny
    alone = forward(model, tok, [PREFIXES[1]], [1], [[0, 1, 2]])["resid"][0]
    batched = forward(model, tok, PREFIXES, [1], [[0, 1, 2], [0, 1, 2]])["resid"][1]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_generation_matches_teacher_forced_argmax(tiny):
    model, tok = tiny
    outs = generate(model, tok, PREFIXES, max_new_tokens=6, stop_at_final_answer=False)
    seqs = [p + o.gen_ids for p, o in zip(PREFIXES, outs)]
    res = forward(model, tok, seqs, return_argmax=True)
    for p, o, am in zip(PREFIXES, outs, res["argmax"]):
        assert am[len(p) - 1 : len(p) - 1 + len(o.gen_ids)] == o.gen_ids


def test_steered_generation_matches_steered_teacher_forcing(tiny):
    """The divergence shortcut relies on this: steering every position >= the
    last prompt token gives the same greedy tokens in both code paths."""
    model, tok = tiny
    vec = torch.randn(32) * 3
    off = padded_offsets(PREFIXES)
    start = torch.tensor([len(p) - 1 for p in PREFIXES]) + off
    edit = ResidEdit("add", vec, start=start)
    outs = generate(model, tok, PREFIXES, 6, hooks={1: edit}, stop_at_final_answer=False)
    seqs = [p + o.gen_ids for p, o in zip(PREFIXES, outs)]
    off2 = padded_offsets(seqs)
    edit2 = ResidEdit("add", vec, start=torch.tensor([len(p) - 1 for p in PREFIXES]) + off2)
    res = forward(model, tok, seqs, hooks={1: edit2}, return_argmax=True)
    for p, o, am in zip(PREFIXES, outs, res["argmax"]):
        assert am[len(p) - 1 : len(p) - 1 + len(o.gen_ids)] == o.gen_ids


def test_project_mode_removes_component(tiny):
    model, tok = tiny
    u = torch.randn(32)
    cap = ResidCapture(None)
    edit = ResidEdit("project", u, strength=1.0, start=torch.tensor([0]))
    ids = torch.tensor([PREFIXES[0]])
    with hooked(model, {1: [edit, cap]}), torch.no_grad():
        model(ids)
    h = cap.result()[0]
    assert torch.allclose(h @ (u / u.norm()), torch.zeros(6), atol=1e-4)


def test_single_position_set_and_gating(tiny):
    model, tok = tiny
    target = torch.ones(2, 32)
    cap = ResidCapture(None)
    edit = ResidEdit("set", target, at=torch.tensor([4, 4]), active=torch.tensor([True, False]))
    ids = torch.tensor([PREFIXES[0][:5], PREFIXES[0][:5]])
    with hooked(model, {0: [edit, cap]}), torch.no_grad():
        model(ids)
    h = cap.result()
    assert torch.allclose(h[0, 4], target[0]) and not torch.allclose(h[1, 4], target[1])
    assert torch.allclose(h[0, :4], h[1, :4])
