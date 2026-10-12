"""A minimal JumpReLU sparse autoencoder.

Simplified relative to the full DeepMind JumpReLU-SAE recipe (Rajamanoharan
et al., 2024): no decoder unit-norm constraint and no resampling of dead
features. Those are TODO for the Semester 7 full-scale run -- see
docs/PROGRESS.md. This version is a real, working implementation, just not
a publication-grade one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _RectangleSTE(torch.autograd.Function):
    """Straight-through estimator for the Heaviside step, with a rectangular
    pseudo-derivative of width `bandwidth` centered on the threshold."""

    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        return (pre_acts > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        in_window = (pre_acts - threshold).abs() < (bandwidth / 2)
        pseudo_grad = in_window.float() / bandwidth
        grad_pre_acts = grad_output * pseudo_grad
        grad_threshold = -grad_output * pseudo_grad
        return grad_pre_acts, grad_threshold, None


def jumprelu(pre_acts: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
    mask = _RectangleSTE.apply(pre_acts, threshold, bandwidth)
    return pre_acts * mask, mask


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, bandwidth: float = 0.001):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = bandwidth

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * (1.0 / d_in**0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * (1.0 / d_sae**0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(torch.full((d_sae,), -3.0))  # exp(-3) ~= 0.05

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_acts = x @ self.W_enc + self.b_enc
        threshold = self.log_threshold.exp()
        acts, mask = jumprelu(pre_acts, threshold, self.bandwidth)
        return acts, mask

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        acts, mask = self.encode(x)
        recon = self.decode(acts)
        return {"reconstruction": recon, "features": acts, "mask": mask}

    def loss(self, x: torch.Tensor, l1_coefficient: float) -> dict[str, torch.Tensor]:
        out = self.forward(x)
        mse = ((out["reconstruction"] - x) ** 2).mean()
        l0_penalty = out["mask"].sum(dim=-1).mean()
        total = mse + l1_coefficient * l0_penalty
        return {"total": total, "mse": mse, "l0": l0_penalty, **out}
