"""Model loading for the harness (thin wrapper over rfd.generation.model)."""

from __future__ import annotations

import os
from typing import Any

# The MPS allocator reads these once, at first use; 8 GB machines swap badly
# without a cap (see docs/DECISIONS.md D7). Must be set before torch touches MPS.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

from rfd.generation.model import LoadedModel, load_model  # noqa: E402


def resolve_dtype(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """dtype "auto": bfloat16 on CUDA, float16 on MPS, float32 on CPU."""
    if model_cfg.get("dtype", "auto") != "auto":
        return model_cfg
    from rfd.utils.io import device_for

    device = device_for(model_cfg.get("device", "auto"))
    dtype = {"cuda": "bfloat16", "mps": "float16"}.get(device, "float32")
    return {**model_cfg, "dtype": dtype}


def load(model_cfg: dict[str, Any]) -> LoadedModel:
    loaded = load_model(resolve_dtype(model_cfg))
    tok = loaded.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    neutralize_generation_config(loaded.model)
    loaded.model.generation_config.pad_token_id = tok.pad_token_id
    return loaded


def neutralize_generation_config(model: Any) -> None:
    """Drop checkpoint sampling defaults. Qwen2.5-Instruct ships
    repetition_penalty=1.1 (and top_k/top_p/temperature), which HF applies even
    with do_sample=False, so "greedy" would not be the raw-logit argmax
    (DECISIONS D12)."""
    gc = model.generation_config
    gc.do_sample = False
    gc.repetition_penalty = 1.0
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None


def library_versions() -> dict[str, str]:
    import platform

    import numpy
    import sklearn
    import torch
    import transformers

    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }
    try:
        from importlib.metadata import version

        out["transformer_lens"] = version("transformer_lens")
        out["datasets"] = version("datasets")
    except Exception:  # noqa: BLE001
        pass
    return out
