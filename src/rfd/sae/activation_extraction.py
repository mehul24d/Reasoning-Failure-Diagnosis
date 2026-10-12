"""Residual-stream activation extraction via TransformerLens hooks.

For each trace we re-tokenize `prompt + generated_trace` with the
TransformerLens model's own tokenizer (guaranteed consistent with its
hook points) and cache `hook_resid_post` at the configured layer(s),
then slice out the positions corresponding to the generated tokens only
-- those are what phase segmentation operates over.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer

from rfd.utils.io import device_for


def resolve_layer(layer_spec: int | str, n_layers: int) -> int:
    if layer_spec == "middle":
        return n_layers // 2
    return int(layer_spec)


def load_hooked_model(model_cfg: dict[str, Any]) -> HookedTransformer:
    device = device_for(model_cfg.get("device", "auto"))
    dtype = torch.float32 if model_cfg.get("dtype", "float32") == "float32" else torch.bfloat16
    model = HookedTransformer.from_pretrained(model_cfg["model_name"], device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def extract_trace_activations(
    hooked_model: HookedTransformer,
    prompt: str,
    generated_text: str,
    layer: int,
    hook_point_template: str,
) -> np.ndarray:
    """Returns activations of shape [n_generated_tokens, d_model]."""
    prompt_tokens = hooked_model.to_tokens(prompt)
    full_tokens = hooked_model.to_tokens(prompt + generated_text)
    hook_name = hook_point_template.format(layer=layer)

    _, cache = hooked_model.run_with_cache(full_tokens, names_filter=hook_name)
    resid = cache[hook_name][0]  # [seq_len, d_model]

    n_prompt_tokens = prompt_tokens.shape[1]
    gen_activations = resid[n_prompt_tokens:]
    return gen_activations.float().cpu().numpy()
