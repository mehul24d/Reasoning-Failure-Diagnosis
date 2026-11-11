from fractions import Fraction

import numpy as np
import pytest

from analysis.stats import benjamini_hochberg, cluster_bootstrap, cohens_d, holm
from parsing.answers import extract_final_answer, gsm8k_gold, is_correct


@pytest.mark.parametrize("text,expected", [
    ("Step 1: 4 * 12 = 48\nStep 2: 48 - 9 = 39\nAnswer: 39", 39),
    ("... so the answer is $1,250.", 1250),
    ("Answer: 39 pencils.", 39),
    ("Answer: **42**", 42),
    ("Answer: -7", -7),
    ("Answer: 2.50", Fraction(5, 2)),
    ("Answer: 3/4", Fraction(3, 4)),
    ("Answer: 10\nWait, recheck. Answer: 12", 12),         # last marker wins
    ("Answer: 39\nThis uses 5 fewer than 44.", 39),         # chatter on later lines ignored
])
def test_extract(text, expected):
    assert extract_final_answer(text) == Fraction(expected)


def test_no_marker_means_no_answer():
    assert extract_final_answer("4 * 12 = 48, minus 9 is 39") is None
    assert extract_final_answer("Answer: unknown") is None


def test_gold_and_correct():
    assert gsm8k_gold("He has 3+4=<<3+4=7>>7\n#### 1,007") == 1007
    assert is_correct(Fraction(39), "39") and not is_correct(Fraction(43), "39")
    assert not is_correct(None, "39")
    assert is_correct(Fraction(5, 2), "2.5")


def test_holm_and_bh_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_cluster_bootstrap_mean_ci_covers_truth():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 8)
    x = rng.normal(1.0, 1.0, size=40)[clusters] + rng.normal(0, 0.5, size=len(clusters))
    uniq, inv = np.unique(clusters, return_inverse=True)

    def stat(w):
        ww = w[inv]
        return float((ww * x).sum() / ww.sum())
    r = cluster_bootstrap(clusters, stat, n_resamples=2000)
    assert r["ci_low"] < 1.0 < r["ci_high"] and r["p"] < 0.01 and r["n_clusters"] == 40


def test_cohens_d():
    x, y = np.array([2.0, 3, 4]), np.array([0.0, 1, 2])
    assert cohens_d(x, y) == pytest.approx(2.0)
