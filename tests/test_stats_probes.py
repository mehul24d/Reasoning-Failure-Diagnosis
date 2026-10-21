import numpy as np

from rfd.probing.probes import fit_probe, grouped_oof_scores, metric, shuffled_labels
from rfd.stats import cluster_bootstrap, expected_calibration_error, safe_auroc


def _data(n_groups=20, per=10, d=16, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = rng.integers(0, 2, len(groups))
    X = rng.standard_normal((len(groups), d))
    X[:, 0] += signal * y
    return X, y, groups


def test_cluster_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(200) + 1.0
    clusters = np.repeat(np.arange(20), 10)
    ci = cluster_bootstrap(lambda idx: vals[idx].mean(), clusters, n_boot=500)
    assert ci["lo"] < ci["estimate"] < ci["hi"]
    assert ci["n_clusters"] == 20


def test_grouped_oof_detects_signal_and_shuffle_kills_it():
    X, y, g = _data()
    for kind in ("logistic", "diffmeans"):
        s = grouped_oof_scores(X, y, g, kind=kind)
        assert metric(y, s, np.array([0, 1])) > 0.85
    ys = shuffled_labels(y, g, np.random.default_rng(1))
    s = grouped_oof_scores(X, ys, g)
    assert metric(ys, s, np.array([0, 1])) < 0.7


def test_multiclass_balanced_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    X = rng.standard_normal((400, 8))
    X[np.arange(400), y] += 3.0
    s = grouped_oof_scores(X, y, np.repeat(np.arange(40), 10), kind="logistic")
    assert metric(y, s, np.arange(4)) > 0.8


def test_direction_points_to_positive_class():
    X, y, _ = _data(signal=3.0)
    for kind in ("logistic", "diffmeans"):
        w = fit_probe(X, y, kind).direction()
        assert np.argmax(np.abs(w)) == 0 and w[0] > 0


def test_ece_and_auroc_edge_cases():
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert np.isnan(safe_auroc(np.array([1, 1]), np.array([0.2, 0.3])))
