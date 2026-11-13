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
    assert pw["deff"]["required_test_failures_with_s0"] == rep["required_test_failures_with_s0"]
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
