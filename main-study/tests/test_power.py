"""Design effect, power floor, the 89-template split and weight provenance (amendments A8, A11-A13)."""
import json

import pytest

from harness import power
from harness.config import REPO_ROOT, experiment_config


def test_kish_equal_sizes_reduces_to_textbook():
    k = power.kish_deff([10, 10, 10, 10], 0.2)
    assert k["cv"] == 0 and k["m_eff"] == 10 and k["deff"] == pytest.approx(1 + 9 * 0.2)


def test_kish_unequal_adjustment_and_zero_clusters():
    sizes = [2, 4, 10, 20]
    k = power.kish_deff(sizes, 0.3)
    assert k["m_eff"] == pytest.approx(sum(s * s for s in sizes) / sum(sizes))
    assert k["deff"] > 1 + (k["m_bar"] - 1) * 0.3           # unequal sizes raise it
    assert power.kish_deff(sizes + [0, 0], 0.3)["deff"] == pytest.approx(k["deff"])


def test_icc_anova_extremes():
    assert power.icc_anova({"a": [0, 0, 0], "b": [1, 1, 1]}) == pytest.approx(1.0)
    assert power.icc_anova({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}) == 0.0
    assert power.icc_anova({"a": [1]}) is None


def test_required_and_floor():
    pw = experiment_config()["data"]["power"]
    assert power.required_failures(1.2, pw) == 320
    assert power.required_failures(19.76, pw) == 4209
    assert power.min_instances_for(320, 40, 6, 0.29, 0.81) == 6
    assert power.min_instances_for(4209, 40, 6, 0.29, 0.81) is None     # > 50 instances


def test_pilot_values_pinned_in_config():
    pw = experiment_config()["data"]["power"]
    rep = json.loads((REPO_ROOT / "reports" / "m1" / "pilot_design_effect.json").read_text())["primary"]
    assert pw["deff"]["icc_failure"] == pytest.approx(rep["icc_failure"], abs=5e-4)
    assert pw["deff"]["value"] == pytest.approx(rep["deff"], abs=5e-3)
    assert pw["deff"]["m_eff"] == pytest.approx(rep["m_eff"], abs=5e-3)
    assert pw["deff"]["required_test_failures_with_s0_if_it_were_the_target"] == rep["required_test_failures_with_s0"]
    assert pw["deff"]["role"] == "descriptive"                        # A15
    assert pw["primary_target_before_outcome_deff"] == 320
    assert pw["pilot_accuracy"] == pytest.approx(rep["pilot_accuracy"], abs=5e-4)


def test_split_file_is_the_89_templates():
    d = experiment_config()["data"]
    assert d["n_templates"] == len(d["study_templates"]) == 89
    assert not set(d["study_templates"]) & set(d["excluded_templates"])
    assert sorted(d["study_templates"] + d["excluded_templates"]) == list(range(100))
    sp = json.loads((REPO_ROOT / "data" / "splits" / "gsm_symbolic_main_seed0.json").read_text())
    ids = sp["train"] + sp["dev"] + sp["test"]
    assert len(ids) == len(set(ids)) == 89
    assert {f"gsms-{t}" for t in d["study_templates"]} == set(ids)
    assert (len(sp["train"]), len(sp["dev"]), len(sp["test"])) == (33, 16, 40)
    from harness.splits import split_templates
    t = d["template_split"]
    assert split_templates(ids, t["train"], t["dev"], t["test"], 0, 40) == \
        {k: sp[k] for k in ("train", "dev", "test")}


def test_weights_provenance_compare(tmp_path):
    from harness import weights_provenance as wp

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 100)
    files = {p.name: wp.file_hashes(p) for p in tmp_path.iterdir()}
    hf = {"files": {"config.json": {"git_oid": files["config.json"]["git_blob_sha1"]},
                    "model-00001-of-00001.safetensors": {"sha256": files["model-00001-of-00001.safetensors"]["sha256"]}}}
    c = wp.compare_with_hf(files, hf)
    assert c["weights_identical_to_hf"] and c["all_required_identical"]
    hf["files"]["model-00001-of-00001.safetensors"]["sha256"] = "0" * 64
    assert not wp.compare_with_hf(files, hf)["weights_identical_to_hf"]
    # gated repo, anonymous request: the hash is masked, so the size is compared and it says so
    hf["files"]["model-00001-of-00001.safetensors"].update(sha256="*" * 64, size=100)
    c = wp.compare_with_hf(files, hf)
    assert c["weights_identical_to_hf"] is None and not c["weights_hash_published"]
    assert c["all_required_identical"] and c["per_file"]["model-00001-of-00001.safetensors"].startswith("same size")
    # git blob sha1 of "{}" (as `git hash-object` gives it)
    assert files["config.json"]["git_blob_sha1"] == "9e26dfeeb6e641a33dae4961196235bdb965b21b"
    h = wp.kaggle_handle("/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2")
    assert h["version"] == 2 and h["handle"] == "google/gemma-2/transformers/gemma-2-2b-it/2"


def test_prompts_frozen_by_s2_signoff():
    """A9: the six prompt templates are frozen as drafted (2026-11-13)."""
    import hashlib

    from harness.prompts import TEMPLATES
    h = hashlib.sha256(json.dumps(TEMPLATES, sort_keys=True).encode()).hexdigest()
    assert h == "5ebffca76922ce476170fac8480ed3579875e4f5cc0e18c47de1df44a3dd51e5"


# --------------------------------------------------------------------- A15, A16 (D36, D37)
def test_clustering_reports_icc_deff_and_effective_n():
    vals = [0] * 10 + [1] * 10 + [0, 1] * 5
    cl = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    c = power.clustering(vals, cl)
    assert c["n_units"] == 30 and c["n_templates"] == 3 and c["icc"] > 0.5
    assert c["deff"] == pytest.approx(1 + 9 * c["icc"]) and c["n_eff"] == pytest.approx(30 / c["deff"])
    # no clustering: deff 1, n_eff n
    flat = power.clustering([0, 1] * 15, [i % 3 for i in range(30)])
    assert flat["icc"] == 0 and flat["deff"] == 1 and flat["n_eff"] == 30


def test_primary_outcome_deff_takes_the_largest():
    clustered = [(t, float(t)) for t in range(5) for _ in range(6)]
    flat = [(t, float(i % 2)) for t in range(5) for i in range(6)]
    r = power.primary_outcome_deff({"e3_decodability": flat, "repair_difference": clustered}, [8] * 40)
    assert r["per_outcome"]["e3_decodability"]["deff"] == 1
    assert r["deff"] == r["per_outcome"]["repair_difference"]["deff"] > 1.5


def test_top_up_rule():
    pw = {"min_test_failures": 320, "n_after_holm": 213, "max_absorbed_deff": 1.5}
    # deff above 1.5 and budget allows -> top-up
    d = power.top_up_decision(2.0, 500, pw, top_up_t4_hours=11, remaining_t4_hours=12)
    assert d["trigger"] and d["action"] == "top_up_instances_9_16" and d["required"] == 426
    # same, budget does not allow -> none, reported underpowered (500 >= 426 so not underpowered)
    d = power.top_up_decision(2.5, 500, pw, top_up_t4_hours=11, remaining_t4_hours=5)
    assert d["action"] == "none" and d["underpowered"] and d["required"] == 533
    # deff <= 1.5 but fewer than 320 failures -> trigger (the proposal's count rule)
    d = power.top_up_decision(1.2, 290, pw, 11, 12)
    assert d["trigger"] and d["action"] == "top_up_instances_9_16"
    # nothing to do
    d = power.top_up_decision(1.2, 400, pw, 11, 12)
    assert not d["trigger"] and d["action"] == "none" and not d["underpowered"]


def test_cohens_kappa_and_bootstrap_clustering():
    import numpy as np

    from analysis.stats import cluster_bootstrap, cohens_kappa

    assert cohens_kappa(["OK", "OK", "FIX", "FIX"], ["OK", "FIX", "FIX", "FIX"]) == pytest.approx(0.5)
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1
    assert cohens_kappa([1, 1], [1, 1]) is None
    ids = np.repeat(np.arange(6), 5)
    vals = ids.astype(float)
    r = cluster_bootstrap(ids, lambda w: float(np.average(np.arange(6), weights=w)), n_resamples=50,
                          unit_values=vals)
    assert r["clustering"]["n_templates"] == 6 and r["clustering"]["n_eff"] < 30


def test_authenticated_hf_hashes_compare_byte_for_byte():
    from harness import weights_provenance as wp

    files = {"model-00001-of-00002.safetensors": {"size": 5, "sha256": "a" * 64, "git_blob_sha1": "x"},
             "config.json": {"size": 3, "sha256": "c" * 64, "git_blob_sha1": "g"}}
    hf = {"revision": "r", "files": {"model-00001-of-00002.safetensors": {"size": 5, "sha256": "a" * 64},
                                     "config.json": {"size": 3, "sha256": None, "git_oid": "g"}}}
    c = wp.compare_with_hf(files, hf)
    assert c["weights_identical_to_hf"] is True and c["weights_hash_published"]
    hf["files"]["model-00001-of-00002.safetensors"]["sha256"] = "b" * 64
    assert wp.compare_with_hf(files, hf)["weights_identical_to_hf"] is False
