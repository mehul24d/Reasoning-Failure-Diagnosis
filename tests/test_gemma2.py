"""Gemma-2 specifics on a tiny randomly initialised Gemma2 (eager attention, no
download, CPU): logit softcapping in teacher-forced log-probs, greedy
reproducibility with left padding, the phase-0 TransformerLens build (bf16
weight source + float32 embedding), and system-role folding in chat prompts."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
tl_lib = pytest.importorskip("transformer_lens")

from rfd.harness.generate import forward, generate  # noqa: E402
from rfd.harness.prompts import chat_ids  # noqa: E402

D = 36  # sqrt(36) = 6: embedding scaling is not exact in bf16


class _Tok:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _hf_config():
    return transformers.Gemma2Config(
        vocab_size=64,
        hidden_size=D,
        intermediate_size=72,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=4096,
        query_pre_attn_scalar=8,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=3.0,  # small cap so softcapping visibly matters
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = transformers.Gemma2ForCausalLM(_hf_config()).eval().to(torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.3)
            if "embed" in name:
                p.mul_(40.0)  # logits of order 10, well inside the softcap's nonlinear range
            p.copy_(p.to(torch.bfloat16).float())  # like a bf16 checkpoint loaded in fp32
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model, _Tok()


SEQS = [[2, 5, 6, 7, 8, 9, 10, 11], [2, 12, 13, 14]]


def test_logp_next_includes_final_softcap(tiny):
    model, tok = tiny
    res = forward(model, tok, [SEQS[0]], return_argmax=True)
    with torch.no_grad():
        logits = model(torch.tensor([SEQS[0]])).logits[0]
    assert float(logits.abs().max()) > 2.0  # the cap (3.0) is active
    lp = torch.log_softmax(logits.float(), -1)
    nxt = torch.tensor(SEQS[0][1:] + SEQS[0][-1:])
    want = lp.gather(-1, nxt[:, None])[:, 0]
    assert torch.allclose(torch.tensor(res["logp_next"][0]), want, atol=1e-4)
    assert res["argmax"][0] == logits.argmax(-1).tolist()


def test_eager_generation_matches_teacher_forcing_with_left_padding(tiny):
    model, tok = tiny
    outs = generate(model, tok, SEQS, max_new_tokens=6, stop_at_final_answer=False)
    seqs = [p + o.gen_ids for p, o in zip(SEQS, outs)]
    res = forward(model, tok, seqs, return_argmax=True)
    for p, o, am in zip(SEQS, outs, res["argmax"]):
        assert am[len(p) - 1 : len(p) - 1 + len(o.gen_ids)] == o.gen_ids


def test_phase0_tl_build_matches_hf_hooks(tiny, monkeypatch):
    from transformer_lens import HookedTransformerConfig
    from transformer_lens import loading_from_pretrained as loading

    from rfd.experiments.phase0_equivalence import tl_from_hf

    model, tok = tiny
    seq = SEQS[0]
    ref = forward(model, tok, [seq], [0, 1, 2, 3], [list(range(len(seq)))])["resid"][0]
    with torch.no_grad():
        ref_logits = model(torch.tensor([seq])).logits[0]
    cfg = HookedTransformerConfig(
        d_model=D,
        d_head=8,
        n_heads=4,
        d_mlp=72,
        n_layers=4,
        n_ctx=128,
        eps=1e-6,
        d_vocab=64,
        act_fn="gelu_pytorch_tanh",
        normalization_type="RMS",
        rotary_base=10000.0,
        rotary_dim=8,
        positional_embedding_type="rotary",
        use_attn_scale=True,
        attn_scale=8**0.5,
        n_key_value_heads=2,
        window_size=4096,
        use_local_attn=True,
        attn_types=["local", "global"] * 2,
        attn_scores_soft_cap=50.0,
        output_logits_soft_cap=3.0,
        gated_mlp=True,
        final_rms=True,
        use_normalization_before_and_after=True,
        original_architecture="Gemma2ForCausalLM",
        dtype=torch.float32,
    )
    monkeypatch.setattr(loading, "get_pretrained_model_config", lambda *a, **k: cfg)
    clone = transformers.Gemma2ForCausalLM(_hf_config()).eval()
    clone.load_state_dict(model.state_dict())
    tl, source = tl_from_hf(clone, None, "google/gemma-2-2b-it", "cpu")
    assert source == "bf16 copy (lossless)"
    assert tl.embed.W_E.dtype == torch.float32
    logits, cache = tl.run_with_cache(torch.tensor([seq]))
    for layer in range(4):
        got = cache[f"blocks.{layer}.hook_resid_post"][0]
        rel = ((got - ref[layer]).norm(dim=-1) / ref[layer].norm(dim=-1)).max()
        assert float(rel) < 1e-5, (layer, float(rel))
    assert torch.allclose(logits[0], ref_logits, atol=1e-4)


class _NoSystemTemplate:
    """A tokenizer whose chat template, like Gemma-2's, rejects the system role."""

    rfd_system_role = False

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "<bos>" + "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<model>"

    def __call__(self, text, add_special_tokens=True):
        self.last_text = text
        return {"input_ids": [ord(c) for c in text]}


def test_chat_ids_folds_system_text_into_first_user_turn():
    tok = _NoSystemTemplate()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    chat_ids(tok, msgs, assistant_prefix="Step")
    assert tok.last_text == "<bos><user>SYS\n\nQ1<assistant>A1<user>Q2<model>Step"
    tok.rfd_system_role = True  # declared as supported: the template error surfaces
    with pytest.raises(ValueError):
        chat_ids(tok, msgs)
