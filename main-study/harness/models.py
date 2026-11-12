"""Model loading through HuggingFace and TransformerLens (plan Section 3.7, component 1)."""
from __future__ import annotations

import torch

from .config import ModelConfig

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype(cfg: ModelConfig, override: str | None = None) -> torch.dtype:
    name = override or cfg.dtype
    if name == "float16":
        # A float16 near-tie effect made random directions appear to alter every trace in the
        # preliminary run (Section 3.9.1). Refuse it rather than silently reintroduce it.
        raise ValueError("float16 is not allowed; use float32 or bfloat16")
    return DTYPES[name]


def load_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.hf_id)


def load_hf(cfg: ModelConfig, device: str | None = None, dtype: str | None = None):
    from transformers import AutoModelForCausalLM

    device = pick_device(device)
    kwargs = {"torch_dtype": torch_dtype(cfg, dtype)}
    if cfg.hf_id.startswith("google/gemma-2"):
        # Eager attention is the reference implementation for Gemma-2 soft-capping.
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **kwargs).to(device)
    model.eval()
    neutralize_generation_defaults(model)
    return model


def neutralize_generation_defaults(model) -> None:
    """Replace the model's shipped generation_config with a neutral one (keeping token ids).

    Second line of defence against defaults such as Qwen's repetition penalty (Section 3.9.1);
    the first is use_model_defaults=False in harness.generate.generate_one.
    """
    from transformers import GenerationConfig

    old = model.generation_config
    model.generation_config = GenerationConfig(
        bos_token_id=old.bos_token_id, eos_token_id=old.eos_token_id, pad_token_id=old.pad_token_id)


def load_tl(cfg: ModelConfig, hf_model=None, tokenizer=None, device: str | None = None,
            dtype: str | None = None, n_devices: int = 1):
    """HookedTransformer with NO weight processing, so residual states match HuggingFace.

    fold_ln / centering would change the residual stream and break the equivalence check.
    """
    from transformer_lens import HookedTransformer

    device = pick_device(device)
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.tl_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch_dtype(cfg, dtype),
        default_prepend_bos=False,  # the chat template supplies BOS (prompts.py)
        n_devices=n_devices,        # >1 splits the blocks over cuda:0..n-1
    )
    model.eval()
    return model
