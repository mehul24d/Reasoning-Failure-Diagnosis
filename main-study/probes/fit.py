"""The shared construction procedure for every variable (prereg/E1.md; also used by the repair
pilot, prereg/R1.md).

Same steps for each variable, layer and position:
  * readouts: a difference-in-means direction (raw residual space) and an L2 logistic probe on
    standardised activations (StandardScaler fit on the training rows only);
  * lambda from experiment.yaml probing.regularisation (lambda = 10^k, C = 1/lambda);
  * template-grouped 5-fold CV on train templates (folds assigned to whole templates by a seeded
    permutation);
  * equal training size: every variable is subsampled to the smallest variable's number of
    train rows (seeded; stratified by class so rare classes are kept in proportion);
  * shuffled-label null: labels permuted WITHIN template groups, then the same fit.
Selection is two-stage, fixed in advance to bound the cost: (1) layer = argmax CV score at
lambda = 1; (2) lambda = argmax CV score on the grid at that layer.

Metric: AUROC for binary variables, balanced accuracy for 4-class ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from harness.config import experiment_config

LAMBDAS = [10.0 ** k for k in experiment_config()["probing"]["regularisation"]["lambda_grid_exponents"]]
N_FOLDS = experiment_config()["probing"]["cv_folds"]


def template_folds(templates: np.ndarray, k: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold index per row; every row of a template is in the same fold."""
    uniq = np.array(sorted(set(templates)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {t: int(i % k) for i, t in zip(range(len(uniq)), uniq[perm])}
    return np.array([fold_of[t] for t in templates])


def subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a class-stratified subsample of size n (all rows if n >= len(y))."""
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    out = []
    classes, counts = np.unique(y, return_counts=True)
    quota = np.floor(counts / len(y) * n).astype(int)
    for i in np.argsort(-(counts / len(y) * n - quota))[: n - quota.sum()]:
        quota[i] += 1
    for c, q in zip(classes, quota):
        idx = np.flatnonzero(y == c)
        out.append(rng.choice(idx, size=q, replace=False))
    return np.sort(np.concatenate(out))


def shuffle_within(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def metric(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], proba[:, 1]))
    return float(balanced_accuracy_score(y, classes[proba.argmax(1)]))


@dataclass
class Probe:
    classes: np.ndarray
    scaler: StandardScaler
    model: LogisticRegression
    means: np.ndarray            # [n_classes, d] class means in RAW residual space
    lam: float

    def proba(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(self.scaler.transform(X))
        # align columns with self.classes (a fold may miss a class)
        out = np.zeros((len(X), len(self.classes)))
        for j, c in enumerate(self.model.classes_):
            out[:, int(np.flatnonzero(self.classes == c)[0])] = p[:, j]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[self.proba(X).argmax(1)]

    def dim_scores(self, X: np.ndarray) -> np.ndarray:
        """Difference-in-means readout: binary = projection on mu1 - mu0 (as [1-s, s] after a
        logistic squash, for the metric); 4-class = minus squared distance to the class means."""
        if len(self.classes) == 2:
            w = self.means[1] - self.means[0]
            s = X @ w - 0.5 * (self.means[1] + self.means[0]) @ w
            s = s / max(float(np.std(s)), 1e-6)
            p1 = 1 / (1 + np.exp(-np.clip(s, -50, 50)))
            return np.stack([1 - p1, p1], 1)
        return -((X[:, None, :] - self.means[None]) ** 2).sum(-1)


def fit(X: np.ndarray, y: np.ndarray, lam: float, classes: np.ndarray | None = None) -> Probe:
    classes = np.unique(y) if classes is None else classes
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1.0 / lam, max_iter=1000, tol=1e-4)
    m.fit(sc.transform(X), y)
    means = np.stack([X[y == c].mean(0) if (y == c).any() else np.full(X.shape[1], np.nan)
                      for c in classes])
    return Probe(classes, sc, m, means, lam)


def cv_score(X: np.ndarray, y: np.ndarray, folds: np.ndarray, lam: float,
             readout: str = "logistic") -> float:
    """Pooled out-of-fold metric over the template-grouped folds."""
    classes = np.unique(y)
    oof = np.zeros((len(y), len(classes)))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2:
            return float("nan")
        p = fit(X[tr], y[tr], lam, classes)
        oof[te] = p.proba(X[te]) if readout == "logistic" else p.dim_scores(X[te])
    return metric(y, oof, classes)


def select(load_layer, layers: list[int], y: np.ndarray, folds: np.ndarray, rows: np.ndarray,
           log=print) -> dict:
    """Two-stage selection on train rows: layer at lambda = 1, then lambda at that layer."""
    by_layer = {}
    for L in layers:
        by_layer[L] = cv_score(load_layer(L, rows), y, folds, 1.0)
        log(f"  layer {L}: cv {by_layer[L]:.4f}")
    best_L = max(by_layer, key=lambda L: (np.nan_to_num(by_layer[L], nan=-1), -L))
    X = load_layer(best_L, rows)
    by_lam = {lam: cv_score(X, y, folds, lam) for lam in LAMBDAS}
    best_lam = max(by_lam, key=lambda lam: (np.nan_to_num(by_lam[lam], nan=-1), lam))
    return {"layer": int(best_L), "lambda": float(best_lam),
            "cv_by_layer": {int(k): v for k, v in by_layer.items()},
            "cv_by_lambda": {str(k): v for k, v in by_lam.items()}}
