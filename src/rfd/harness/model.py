"""Model loading for the harness (thin wrapper over rfd.generation.model)."""

from __future__ import annotations

import os
from typing import Any

# The MPS allocator reads these once, at first use; 8 GB machines swap badly
# without a cap (see docs/DECISIONS.md D7). Must be set before torch touches MPS.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

from rfd.generation.model import LoadedModel, load_model  # noqa: E402


def load(model_cfg: dict[str, Any]) -> LoadedModel:
    loaded = load_model(model_cfg)
    tok = loaded.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    loaded.model.generation_config.pad_token_id = tok.pad_token_id
    return loaded


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
