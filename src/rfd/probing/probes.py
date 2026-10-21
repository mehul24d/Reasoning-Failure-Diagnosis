"""Linear probes with template-grouped cross-validation and shuffled-label nulls.

Two readouts share one interface so every variable gets the same procedure:
  "logistic"  StandardScaler + L2 logistic regression (C=0.1, as Yuan et al.)
  "diffmeans" class means of standardized activations; binary score =
              projection onto (mu_1 - mu_0), multi-class = nearest class mean
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from rfd.stats import safe_auroc, safe_balanced_accuracy


@dataclass
class FittedProbe:
    kind: str
    classes: np.ndarray
    scaler: StandardScaler
    model: LogisticRegression | None = None
    means: np.ndarray | None = None  # [n_classes, d] in standardized space

    def scores(self, X: np.ndarray) -> np.ndarray:
        """[n, n_classes] class scores (probabilities for logistic)."""
        Z = self.scaler.transform(X)
        if self.kind == "logistic":
            return self.model.predict_proba(Z)  # type: ignore[union-attr]
        d = -((Z[:, None, :] - self.means[None]) ** 2).sum(-1)  # type: ignore[index]
        if len(self.classes) == 2:  # projection onto mu1 - mu0, as a 2-column score
            w = self.means[1] - self.means[0]  # type: ignore[index]
            s = Z @ w - 0.5 * (self.means[1] @ w + self.means[0] @ w)  # type: ignore[index]
            p1 = 1 / (1 + np.exp(-s / max(np.std(s), 1e-6)))
            return np.stack([1 - p1, p1], 1)
        return d

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[self.scores(X).argmax(1)]

    def direction(self, raw_space: bool = True) -> np.ndarray:
        """Binary direction (class 1 minus class 0). In raw activation space by
        default: for logistic, w / scale; for diffmeans, the raw mean difference."""
        if self.kind == "logistic":
            w = self.model.coef_[0]  # type: ignore[union-attr]
            return w / self.scaler.scale_ if raw_space else w
        diff = self.means[1] - self.means[0]  # type: ignore[index]
        return diff * self.scaler.scale_ if raw_space else diff


def fit_probe(X: np.ndarray, y: np.ndarray, kind: str = "logistic", C: float = 0.1) -> FittedProbe:
    classes = np.unique(y)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    if kind == "logistic":
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(Z, y)
        return FittedProbe(kind, model.classes_, scaler, model=model)
    if kind == "diffmeans":
        means = np.stack([Z[y == c].mean(0) for c in classes])
        return FittedProbe(kind, classes, scaler, means=means)
    raise ValueError(kind)


def metric(y: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> float:
    """AUROC for binary variables, balanced accuracy for multi-class."""
    if len(classes) == 2:
        return safe_auroc(y == classes[1], scores[:, 1])
    return safe_balanced_accuracy(y, classes[scores.argmax(1)])


def grouped_oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logistic",
    n_splits: int = 5,
    C: float = 0.1,
    classes: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-fold class scores with template-grouped folds ([n, n_classes]).
    Folds whose training part lacks a class get NaN rows."""
    classes = np.unique(y) if classes is None else classes
    out = np.full((len(y), len(classes)), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < len(classes):
            continue
        p = fit_probe(X[tr], y[tr], kind, C)
        s = p.scores(X[te])
        col = {c: i for i, c in enumerate(p.classes)}
        for j, c in enumerate(classes):
            if c in col:
                out[te, j] = s[:, col[c]]
    return out


def shuffled_labels(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Labels permuted across items (control task: same marginals, no signal)."""
    return y[rng.permutation(len(y))]
