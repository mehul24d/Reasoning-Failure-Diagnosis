import numpy as np

from rfd.legacy.metrics.sparsity import feature_entropy, l0_sparsity


def test_l0_sparsity_counts_nonzero():
    acts = np.array([[0, 1, 2], [0, 0, 0], [3, 3, 3]])
    assert list(l0_sparsity(acts)) == [2, 0, 3]


def test_feature_entropy_zero_for_single_active_feature():
    acts = np.array([[5.0, 0.0, 0.0]])
    entropy = feature_entropy(acts)
    assert entropy[0] == 0.0


def test_feature_entropy_zero_row_is_zero():
    acts = np.array([[0.0, 0.0, 0.0]])
    entropy = feature_entropy(acts)
    assert entropy[0] == 0.0


def test_feature_entropy_higher_for_uniform_distribution():
    peaked = feature_entropy(np.array([[10.0, 0.1, 0.1]]))[0]
    uniform = feature_entropy(np.array([[1.0, 1.0, 1.0]]))[0]
    assert uniform > peaked
