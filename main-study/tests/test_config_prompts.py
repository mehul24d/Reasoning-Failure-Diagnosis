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
    assert g["sampled_subset"] == {"temperature": 0.7, "k": 5, "do_sample": True}
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
