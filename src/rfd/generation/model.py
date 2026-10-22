"""Model + tokenizer loading shared by generation and activation extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfd.utils.io import device_for


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(model_cfg: dict[str, Any]) -> LoadedModel:
    device = device_for(model_cfg.get("device", "auto"))
    dtype = _DTYPE_MAP[model_cfg.get("dtype", "float32")]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    extra: dict[str, Any] = {}
    if model_cfg.get("attn_implementation"):
        extra["attn_implementation"] = model_cfg["attn_implementation"]
    if device == "cuda":
        # load shards straight onto the GPU: an fp32 2B model does not fit in
        # host RAM next to its own upcast on a 12 GB Colab VM
        extra["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        **extra,
    )
    model.to(device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def build_prompt(tokenizer: Any, prompt_template: str, question: str) -> str:
    """Fill the math-solving template and wrap it in the model's chat
    template (Qwen2.5-Instruct expects chat-formatted input)."""
    filled = prompt_template.format(question=question)
    messages = [{"role": "user", "content": filled}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
