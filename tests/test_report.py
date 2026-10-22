import json

from rfd.experiments.report import baseline_b, phase0, pilot, scale_lines, scale_rows


def test_missing_results_render_not_run(tmp_path):
    text = "\n".join(phase0(tmp_path) + baseline_b(tmp_path) + pilot(tmp_path))
    assert text.count("NOT RUN") >= 4
    assert "PASS" not in text and "FAIL" not in text


def test_baseline_b_verdict_follows_criteria(tmp_path):
    ci = lambda e, lo, hi: {"estimate": e, "lo": lo, "hi": hi}  # noqa: E731
    probe = {
        "n_traces": 10,
        "n_train_templates": 5,
        "n_test_templates": 5,
        "failure_rate_test": 0.5,
        "accuracy_test_by_variant": {"main": 0.6},
        "positions": {
            "last": {
                "layer": 3,
                "logistic": {
                    "test_auroc": ci(0.9, 0.8, 0.95),
                    "test_auprc": ci(0.8, 0.7, 0.9),
                    "auroc_minus_shuffled": ci(0.4, 0.3, 0.5),
                    "ece": 0.1,
                    "brier": 0.2,
                    "brier_base_rate": 0.25,
                },
                "diffmeans": {
                    "test_auroc": ci(0.85, 0.8, 0.9),
                    "test_auprc": ci(0.8, 0.7, 0.9),
                    "auroc_minus_shuffled": ci(0.3, 0.2, 0.4),
                },
            }
        },
        "baselines": {},
    }
    null = {"n": 100, "net_mean": 0.0, "net_q025": -0.02, "net_q975": 0.02, "p_value_net_ge": 0.5}
    steer = {
        "layer": 3,
        "norm_mu_diff": 1.0,
        "n_eval_train": 8,
        "n_eval_test": 8,
        "zero_strength_n_diverged": 0,
        "test": {
            "add_diffmeans": {
                "strength": 1,
                "repair_ci": ci(0.1, 0, 0.2),
                "damage_ci": ci(0.1, 0, 0.2),
                "net_natural_ci": ci(0.01, -0.02, 0.04),
                "net_natural": 0.01,
                "null": null,
            }
        },
    }
    (tmp_path / "baseline_b").mkdir()
    (tmp_path / "baseline_b" / "probe.json").write_text(json.dumps(probe))
    (tmp_path / "baseline_b" / "steer.json").write_text(json.dumps(steer))
    assert "**Verdict: PASS**" in "\n".join(baseline_b(tmp_path))
    steer["test"]["add_diffmeans"]["net_natural"] = 0.08
    (tmp_path / "baseline_b" / "steer.json").write_text(json.dumps(steer))
    assert "**Verdict: PARTIAL**" in "\n".join(baseline_b(tmp_path))


def _steer_with_scale(n_train, n_test, n_null, grid_len):
    ci = {"estimate": 0.0, "lo": -0.1, "hi": 0.1}
    return {
        "layer": 1,
        "norm_mu_diff": 1.0,
        "n_eval_train": n_train,
        "n_eval_test": n_test,
        "zero_strength_n_diverged": 0,
        "selection_train": {
            "yuan_project": {"grid": [{}] * 5, "selected": 1},
            "add_diffmeans": {"grid": [{}] * grid_len, "selected": 1},
        },
        "test": {
            k: {
                "strength": 1,
                "repair_ci": ci,
                "damage_ci": ci,
                "net_natural_ci": ci,
                "net_natural": 0.0,
                "null": {
                    "n": n_null,
                    "net_mean": 0.0,
                    "net_q025": 0.0,
                    "net_q975": 0.0,
                    "p_value_net_ge": 0.5,
                },
            }
            for k in ("yuan_project", "add_diffmeans")
        },
    }


def test_scale_rows_flag_reduced_runs(tmp_path):
    (tmp_path / "baseline_b").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "config_used.json").write_text('{"instances_per_template": 4}')
    full = _steer_with_scale(128, 160, 100, 7)
    (tmp_path / "baseline_b" / "steer.json").write_text(json.dumps(full))
    rows = {n: (want, got) for n, want, got in scale_rows(tmp_path)}
    assert rows["Baseline B steering: random directions per condition"] == (100, 100)
    assert rows["instances per template (traces)"] == (8, 4)  # traces alone are reduced
    small = _steer_with_scale(64, 80, 20, 4)
    (tmp_path / "baseline_b" / "steer.json").write_text(json.dumps(small))
    rows = {n: (want, got) for n, want, got in scale_rows(tmp_path)}
    assert rows["Baseline B steering: random directions per condition"] == (100, 20)
    assert rows["Baseline B strength grid: add_diffmeans"] == (7, 4)
    text = "\n".join(scale_lines(scale_rows(tmp_path)))
    assert "| Baseline B steering: test traces | 160 | 80 | False |" in text


def test_reduced_steering_run_gets_a_provisional_verdict(tmp_path):
    ci = lambda e, lo, hi: {"estimate": e, "lo": lo, "hi": hi}  # noqa: E731
    lg = {
        "test_auroc": ci(0.9, 0.8, 0.95),
        "test_auprc": ci(0.8, 0.7, 0.9),
        "auroc_minus_shuffled": ci(0.4, 0.3, 0.5),
        "ece": 0.1,
        "brier": 0.2,
        "brier_base_rate": 0.25,
    }
    probe = {
        "n_traces": 10,
        "n_train_templates": 5,
        "n_test_templates": 5,
        "failure_rate_test": 0.5,
        "accuracy_test_by_variant": {"main": 0.6},
        "positions": {"last": {"layer": 3, "logistic": lg, "diffmeans": lg}},
        "baselines": {},
    }
    (tmp_path / "baseline_b").mkdir()
    (tmp_path / "baseline_b" / "probe.json").write_text(json.dumps(probe))
    steer = _steer_with_scale(128, 160, 100, 7)
    (tmp_path / "baseline_b" / "steer.json").write_text(json.dumps(steer))
    assert "Provisional" not in "\n".join(baseline_b(tmp_path))
    steer = _steer_with_scale(64, 80, 20, 4)
    (tmp_path / "baseline_b" / "steer.json").write_text(json.dumps(steer))
    text = "\n".join(baseline_b(tmp_path))
    assert "**Verdict: PASS** " in text and "Provisional: reduced scale" in text
    assert "random directions per condition" in text
