"""Fast tests: no model weights needed."""
from harness.config import experiment_config, model_config
from harness.prompts import TEMPLATES, PromptSpec, all_prompt_specs, render_user_message


def test_models_match_plan_table_12():
    for key, layers, width in [("gemma-2-2b-it", 26, 2304), ("gemma-2-9b-it", 42, 3584),
                               ("qwen2.5-7b-instruct", 28, 3584)]:
        m = model_config(key)
        assert (m.n_layers, m.d_model) == (layers, width)
    assert model_config("qwen2.5-7b-instruct").sae is None


def test_experiment_config_matches_plan_table_25():
    e = experiment_config()
    g = e["generation"]
    assert g["greedy"]["temperature"] == 0.0 and g["greedy"]["repetition_penalty"] == 1.0
    assert g["greedy"]["batch_size"] == 1
    assert {k: g["sampled_subset"][k] for k in ("temperature", "k", "do_sample")} == \
        {"temperature": 0.7, "k": 5, "do_sample": True}
    assert e["data"]["instances_per_template"] == 8
    assert e["thresholds"]["sae_fvu_max"] == 0.40
    assert e["thresholds"]["sae_delta_ce_max"] == 0.20
    assert e["thresholds"]["rq1_min_score"] == 0.75
    assert e["thresholds"]["rq3_min_top1_gold"] == 0.20
    assert e["probing"]["shuffled_label_permutations"] == 100
    assert len(e["probing"]["seeds"]) == 5
    assert e["statistics"]["bootstrap_resamples"] == 10000


def test_two_styles_three_paraphrases():
    assert set(TEMPLATES) == {"marked", "terse"}
    assert all(len(v) == 3 for v in TEMPLATES.values())
    assert len(all_prompt_specs()) == 6


def test_marked_prompt_is_plan_wording():
    msg = render_user_message("  What is 2+2?  ", PromptSpec("marked", 0))
    assert msg.startswith("Solve the following problem step by step.")
    assert '"Step k: <calculation> = <result>"' in msg
    assert msg.endswith("Problem: What is 2+2?")


def test_every_template_has_answer_marker_and_problem():
    for spec in all_prompt_specs():
        msg = render_user_message("X", spec)
        assert "Answer:" in msg and msg.endswith("Problem: X")


def test_rescope_2026_11_13_changes_no_threshold_or_procedure():
    """prereg/S2.md amendments A1-A6: compute changes only. Every threshold, the number of
    random directions / permutations, the bootstrap/CI procedure and the annotation rule stay."""
    e = experiment_config()
    assert e["scope"]["minor_project_models"] == ["gemma-2-2b-it"]
    assert e["generation"]["greedy"]["max_new_tokens"] == 512
    assert e["generation"]["sampled_subset"]["population"] == "test_failures"
    assert e["generation"]["batched"]["padding"] == "left"
    assert e["thresholds"] == {
        "sae_fvu_max": 0.40, "sae_delta_ce_max": 0.20, "rq1_min_score": 0.75, "rq2_alarm_fpr": 0.10,
        "rq3_min_top1_gold": 0.20, "stable_failure": "no sampled run correct", "kappa_min": 0.6}
    assert e["statistics"] == {"bootstrap_resamples": 10000, "ci": 0.95, "fdr_q": 0.05,
                               "e0_correction": "holm"}
    assert e["annotation"] == {"fraction": 0.20, "annotators": 2}
    assert e["probing"]["shuffled_label_permutations"] == 100
    assert e["equivalence_checks"]["teacher_forced_reproduction_min"] == 0.99
    assert e["verification"]["threshold_key"] == "teacher_forced_reproduction_min"
    assert e["cache"]["storage_dtype"] == "float16"
    assert e["data"]["min_test_failures"] == 320 and e["data"]["min_test_templates"] == 40
