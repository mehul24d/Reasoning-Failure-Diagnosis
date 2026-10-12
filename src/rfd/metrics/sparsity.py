"""Per-token SAE sparsity and feature-entropy metrics."""

from __future__ import annotations

import numpy as np


def l0_sparsity(feature_acts: np.ndarray) -> np.ndarray:
    """Number of active (nonzero) features per token.

    feature_acts: [n_tokens, d_sae] -> returns [n_tokens]
    """
    return (feature_acts > 0).sum(axis=-1)


def feature_entropy(feature_acts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy (nats) of each token's normalized feature-activation
    distribution. Feature activations from JumpReLU are always >= 0.

    feature_acts: [n_tokens, d_sae] -> returns [n_tokens]
    """
    totals = feature_acts.sum(axis=-1, keepdims=True)
    probs = np.divide(feature_acts, totals, out=np.zeros_like(feature_acts), where=totals > eps)
    entropy = -(probs * np.log(probs + eps)).sum(axis=-1)
    entropy[totals.squeeze(-1) <= eps] = 0.0
    return np.clip(entropy, 0.0, None)
