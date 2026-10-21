"""Residual-stream read/write hooks on HuggingFace decoder layers.

Convention: "layer L" = the output of decoder block L (0-indexed), i.e.
TransformerLens `blocks.{L}.hook_resid_post` and HF `hidden_states[L + 1]`
(except the very last HF hidden state, which HF returns AFTER the final norm;
our hook reads the pre-norm block output, which is what TL exposes). The
TL/HF equivalence of this convention is checked in `rfd.harness.equivalence`.

All hooks are position-aware so the same object works for a teacher-forced
full forward pass AND for incremental KV-cached generation: the hook tracks
the absolute (padded) position of the first token in each forward call.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


def decoder_layers(model: Any) -> Any:
    """The list of decoder blocks of a HF causal LM (Qwen2/Llama/Gemma layout)."""
    return model.model.layers


class ResidEdit:
    """Edit the output of one decoder block at selected positions.

    mode:
      "add"     h += strength * vec                       (vec: [d] or [bs, d])
      "project" h -= strength * (h . u) u, u = vec/|vec|  (Yuan et al. 2605.09502)
      "set"     h  = vec                                  (patching; vec: [bs, d])
    positions:
      start: LongTensor [bs]; edit every position >= start[i]   (steer during generation)
      at:    LongTensor [bs]; edit only position == at[i]       (single-token edit)
    `active`: optional BoolTensor [bs]; rows with False are left untouched
    (used for gated self-sourced edits).
    """

    def __init__(
        self,
        mode: str,
        vec: torch.Tensor,
        strength: float = 1.0,
        start: torch.Tensor | None = None,
        at: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> None:
        if mode not in ("add", "project", "set"):
            raise ValueError(mode)
        if (start is None) == (at is None):
            raise ValueError("give exactly one of start= or at=")
        self.mode, self.vec, self.strength = mode, vec, strength
        self.start, self.at, self.active = start, at, active
        self.offset = 0

    def reset(self) -> None:
        self.offset = 0

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        bs, seq, _ = h.shape
        pos = self.offset + torch.arange(seq, device=h.device)
        self.offset += seq
        if self.start is not None:
            mask = pos[None, :] >= self.start.to(h.device)[:, None]
        else:
            mask = pos[None, :] == self.at.to(h.device)[:, None]
        if self.active is not None:
            mask = mask & self.active.to(h.device)[:, None]
        if not bool(mask.any()):
            return output
        vec = self.vec.to(h.device, h.dtype)
        vec = vec.expand(bs, -1) if vec.dim() == 1 else vec
        m = mask[..., None].to(h.dtype)
        if self.mode == "add":
            new = h + m * self.strength * vec[:, None, :]
        elif self.mode == "project":
            u = vec / vec.norm(dim=-1, keepdim=True)
            coef = (h.float() * u[:, None, :].float()).sum(-1, keepdim=True).to(h.dtype)
            new = h - m * self.strength * coef * u[:, None, :]
        else:
            new = h * (1 - m) + m * vec[:, None, :]
        if isinstance(output, tuple):
            return (new, *output[1:])
        return new


class ResidCapture:
    """Record the output of one decoder block at selected absolute positions.

    positions: LongTensor [bs, k] (padded coordinates; -1 = skip), or None to
    keep every position. Captured values are stored on CPU in float32.
    """

    def __init__(self, positions: torch.Tensor | None = None) -> None:
        self.positions = positions
        self.offset = 0
        self.chunks: list[torch.Tensor] = []

    def __call__(self, module: Any, inputs: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        seq = h.shape[1]
        if self.positions is None:
            self.chunks.append(h.detach().float().cpu())
        else:
            idx = (self.positions - self.offset).to(h.device)
            valid = (idx >= 0) & (idx < seq)
            safe = idx.clamp(0, seq - 1)
            got = torch.gather(h, 1, safe[..., None].expand(-1, -1, h.shape[-1]))
            got = got * valid[..., None].to(h.dtype)
            self.chunks.append(got.detach().float().cpu())
        self.offset += seq

    def result(self) -> torch.Tensor:
        if self.positions is None:
            return torch.cat(self.chunks, dim=1)
        return torch.stack(self.chunks).sum(0)  # each position is valid in exactly one chunk


@contextmanager
def hooked(model: Any, hooks: dict[int, Any]) -> Iterator[None]:
    """Register {layer: hook_or_list_of_hooks} forward hooks for the duration of the block."""
    layers = decoder_layers(model)
    handles = []
    try:
        for layer, hs in hooks.items():
            for h in hs if isinstance(hs, list) else [hs]:
                if hasattr(h, "reset"):
                    h.reset()
                handles.append(layers[layer].register_forward_hook(h))
        yield
    finally:
        for handle in handles:
            handle.remove()
