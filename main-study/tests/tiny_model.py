"""A tiny random Qwen2-architecture model + word-level tokenizer, built offline.

Lets the harness logic (greedy config, padding, teacher forcing, TL-HF residual alignment,
generation resume) be tested with no downloads. It says nothing about real-model numerics;
those are checked by scripts/s1_equivalence.sh on the real models.
"""
from __future__ import annotations

import torch

WORDS = (["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>", "user", "assistant", "\n"]
         + [str(i) for i in range(100)]
         + ("Solve the following problem step by step . Write each on a new line in form Step k : "
            "calculation = result End with Answer number Problem What is + ? Tom has apples "
            "Liam buys boxes of pencils").split())


def build(seed: int = 0):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {w: i for i, w in enumerate(dict.fromkeys(WORDS))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", unk_token="<unk>")
    tok.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} \n {{ m['content'] }} <|im_end|> \n "
        "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant \n {% endif %}")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=False)
    # Round-trip through disk so the tokenizer looks like a pretrained one (TL reads name_or_path).
    import tempfile

    from transformers import AutoTokenizer

    d = tempfile.mkdtemp(prefix="tiny_tok_")
    tok.save_pretrained(d)
    tok = AutoTokenizer.from_pretrained(d)
    torch.manual_seed(seed)
    hf = Qwen2ForCausalLM(cfg).eval()
    return hf, tok


def build_tl(hf, tok):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights

    c = hf.config
    d_head = c.hidden_size // c.num_attention_heads
    cfg = HookedTransformerConfig(
        d_model=c.hidden_size, d_head=d_head, n_heads=c.num_attention_heads,
        n_key_value_heads=c.num_key_value_heads, d_mlp=c.intermediate_size,
        n_layers=c.num_hidden_layers, n_ctx=512, eps=c.rms_norm_eps, d_vocab=c.vocab_size,
        act_fn="silu", normalization_type="RMS", positional_embedding_type="rotary",
        rotary_base=c.rope_theta, rotary_adjacent_pairs=False, rotary_dim=d_head,
        gated_mlp=True, final_rms=True, original_architecture="Qwen2ForCausalLM",
        tokenizer_prepends_bos=False, default_prepend_bos=False, dtype=torch.float32, device="cpu")
    tl = HookedTransformer(cfg, tokenizer=tok)
    tl.load_state_dict(convert_qwen2_weights(hf, cfg), strict=False)
    return tl.eval()
