"""Render docs/BASELINE_REPORT_TABLES.md from results/*.json.

Every number and every PASS / PARTIAL / FAIL verdict in the tables is computed
here from script outputs, using the criteria in docs/prereg/*.md. Missing
results render as NOT RUN, never as a guess.

Usage: python -m rfd.experiments.report [--results results] [--out docs/BASELINE_REPORT_TABLES.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MIN_METRIC = 0.75  # held-out AUROC / balanced accuracy threshold
MIN_RECOVERY = 0.20  # gold-value top-1 in failures
MIN_EFFECT = 0.05  # 5 points of net accuracy gain


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _ci(d: dict[str, Any] | None, pct: bool = False, digits: int = 3) -> str:
    if not d:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.{digits}f}")
    return f"{f(d['estimate'])} [{f(d['lo'])}, {f(d['hi'])}]"


def _verdict(flags: list[bool]) -> str:
    if all(flags):
        return "PASS"
    if any(flags):
        return "PARTIAL"
    return "FAIL"


def phase0(res: Path) -> list[str]:
    lines = ["## Phase 0: harness checks and data", ""]
    eq = _load(res / "phase0" / "equivalence.json")
    if eq is None:
        lines += ["TL/HF equivalence: **NOT RUN**", ""]
    else:
        lines += [
            f"Model `{eq['model']}`, {eq['dtype']} on {eq['device']}.",
            "",
            "| check | value | threshold | pass |",
            "|---|---|---|---|",
            f"| TL vs HF resid_post, median rel. error (worst layer) | {max(eq['tl_resid_rel_err_median'].values()):.2e} | < 0.01 | {eq['pass']['tl_resid']} |",
            f"| TL vs HF logits top-1 agreement | {eq['tl_logits_top1_agreement']:.4f} | ≥ 0.99 | {eq['pass']['tl_logits_top1']} |",
            f"| padded batch vs single, median rel. error | {eq['padding_rel_err_median']:.2e} | < 0.01 | {eq['pass']['padding']} |",
            f"| teacher-forced argmax reproduces greedy tokens | {eq['greedy_tf_agreement']:.4f} ({eq['greedy_tf_n_tokens']} tokens) | ≥ 0.99 | {eq['pass']['greedy_tf']} |",
            "",
        ]
    pr = _load(res / "traces" / "parse_report.json")
    if pr is None:
        lines += ["Trace parsing: **NOT RUN**", ""]
    else:
        acc = ", ".join(f"{k}: {v:.3f}" for k, v in pr["accuracy_by_variant"].items())
        lines += [
            f"Traces: {pr['n_traces']}; accuracy by variant: {acc}; `Final Answer:` marker rate "
            f"{pr['final_marker_rate']:.3f}; steps {pr['n_steps']}, step parse rate "
            f"{pr['step_parse_rate']:.3f}; error types (parsed steps) {pr['error_type_counts']}.",
            "",
        ]
    return lines


def baseline_a(res: Path) -> list[str]:
    lines = ["## Baseline A: Sun et al. (2507.12379)", ""]
    pil = _load(res / "baseline_a" / "error_pilot.json")
    an = _load(res / "baseline_a" / "analysis.json")
    rp = _load(res / "baseline_a" / "reprompt.json")
    if pil:
        lines += [
            "Error pilot (errors / n by digit length): "
            + ", ".join(f"{k}: {v['errors']}/{v['n']}" for k, v in pil["rates"].items())
            + f". Chosen: {pil['chosen']}.",
            "",
        ]
    if an is None:
        return lines + ["Probing: **NOT RUN**", ""]
    pr = an["primary_P1_correct_only"]
    counts = {int(k): v for k, v in an["error_place_counts"].items()}
    main_place = max(counts, key=counts.get) if counts else 0
    dec = pr["decodability_selected_layer"][str(main_place)]
    c1 = dec["test_bacc"] >= MIN_METRIC and dec["acc_minus_shuffled"]["lo"] > 0
    c2 = pr["recovery_gold"]["estimate"] >= MIN_RECOVERY and pr["recovery_minus_shuffled"]["lo"] > 0
    lines += [
        f"n = {an['n']} problems ({an['meta']['n_digits']}-digit addition), {an['n_errors_parsed']} errors, "
        f"{an['n_errors_analyzed']} analysed (same digit count as gold). Most errors at place "
        f"{main_place} (counts {counts}).",
        "",
        "| quantity (protocol P1: probes trained on correct cases only) | value [95% CI] |",
        "|---|---|",
        f"| C1: held-out balanced acc., gold digit at place {main_place}, layer {dec['layer']} | {dec['test_bacc']:.3f} (shuffled {dec['test_bacc_shuffled']:.3f}) |",
        f"| C1: accuracy − shuffled | {_ci(dec['acc_minus_shuffled'])} |",
        f"| C2: error cases where probe = **correct** digit | {_ci(pr['recovery_gold'])} |",
        f"| error cases where probe = model's **wrong** digit | {_ci(pr['tracks_model_output'])} |",
        f"| shuffled-label probe = correct digit | {_ci(pr['recovery_gold_shuffled'])} |",
        f"| random-weight probe = correct digit | {_ci(pr['recovery_gold_random_probe'])} |",
        f"| prior (gold digit's train frequency) | {pr['prior_baseline']:.3f} |",
        f"| C2: recovery − shuffled | {_ci(pr['recovery_minus_shuffled'])} |",
        "",
        f"**Verdict (pre-registered C1 ∧ C2): {_verdict([c1, c2])}** (C1 {c1}, C2 {c2}).",
        "",
    ]
    if rp:
        acc = rp["accuracy"]
        lines += [
            f"Selective re-prompting (test n = {rp['n_test']}, {rp['n_test_errors']} errors; detector layer "
            f"{rp['detector_layer']}, test AUROC {rp['detector_test_auroc']:.3f}; flagged {rp['n_flagged']}: "
            f"{rp['true_positives']} TP / {rp['false_positives']} FP; TP-correction "
            f"{rp['tp_correction']}, FP-preservation {rp['fp_preservation']}):",
            "",
            "| condition | accuracy % [95% CI] |",
            "|---|---|",
        ]
        lines += [f"| {k} | {_ci(v, pct=True)} |" for k, v in acc.items()]
        lines += [
            f"| selective − none (points) | {_ci(rp['selective_minus_none'], pct=True)} |",
            f"| selective − always (points) | {_ci(rp['selective_minus_always'], pct=True)} |",
            "",
        ]
    else:
        lines += ["Selective re-prompting: **NOT RUN**", ""]
    return lines


def baseline_b(res: Path) -> list[str]:
    lines = ["## Baseline B: Yuan et al. (2605.09502)", ""]
    pb = _load(res / "baseline_b" / "probe.json")
    st = _load(res / "baseline_b" / "steer.json")
    if pb is None:
        return lines + ["Probes: **NOT RUN**", ""]
    lines += [
        f"{pb['n_traces']} traces ({pb['n_train_templates']} train / {pb['n_test_templates']} test templates); "
        f"test failure rate {pb['failure_rate_test']:.3f}; test accuracy by variant "
        + ", ".join(f"{k} {v:.3f}" for k, v in pb["accuracy_test_by_variant"].items())
        + ".",
        "",
        "| failure predictor (held-out test templates) | layer | AUROC [95% CI] | AUPRC | AUROC − shuffled |",
        "|---|---|---|---|---|",
    ]
    for pos, r in pb["positions"].items():
        if "error" in r:
            lines.append(f"| probe @ {pos} | - | {r['error']} | | |")
            continue
        lg = r["logistic"]
        lines.append(
            f"| logistic probe @ {pos} | {r['layer']} | {_ci(lg['test_auroc'])} | "
            f"{lg['test_auprc']['estimate']:.3f} | {_ci(lg['auroc_minus_shuffled'])} |"
        )
        dm = r["diffmeans"]
        lines.append(
            f"| diff-in-means @ {pos} | {r['layer']} | {_ci(dm['test_auroc'])} | "
            f"{dm['test_auprc']['estimate']:.3f} | {_ci(dm['auroc_minus_shuffled'])} |"
        )
    for name, b in pb["baselines"].items():
        lines.append(f"| baseline: {name} | - | {_ci(b['test_auroc'])} | | |")
    last = pb["positions"].get("last", {})
    d1 = bool(
        last
        and "error" not in last
        and last["logistic"]["test_auroc"]["estimate"] >= MIN_METRIC
        and last["logistic"]["auroc_minus_shuffled"]["lo"] > 0
    )
    if last and "error" not in last:
        lines += [
            "",
            f"Calibration (last-token logistic): ECE {last['logistic']['ece']:.3f}, Brier "
            f"{last['logistic']['brier']:.3f} (base-rate Brier {last['logistic']['brier_base_rate']:.3f}).",
        ]
    lines.append("")
    if st is None:
        return lines + [f"D1 (diagnostic): {d1}. Steering: **NOT RUN** → verdict pending.", ""]
    lines += [
        f"Steering at layer {st['layer']} (‖μ_c − μ_w‖ = {st['norm_mu_diff']:.2f}); eval sets: "
        f"{st['n_eval_train']} train / {st['n_eval_test']} test traces; zero-strength control diverged on "
        f"{st['zero_strength_n_diverged']} traces.",
        "",
        "| direction | selected strength (train) | repair % [CI] | damage % [CI] | net gain, pts [CI] | random null net (mean; 2.5-97.5%) | p (null ≥ real) |",
        "|---|---|---|---|---|---|---|",
    ]
    d2 = []
    for kind, r in st["test"].items():
        nl = r["null"]
        lines.append(
            f"| {kind} | {r['strength']} | {_ci(r['repair_ci'], True)} | {_ci(r['damage_ci'], True)} | "
            f"{_ci(r['net_natural_ci'], True)} | {100 * nl['net_mean']:.1f}; {100 * nl['net_q025']:.1f} to "
            f"{100 * nl['net_q975']:.1f} (n={nl['n']}) | {nl['p_value_net_ge']:.3f} |"
        )
        d2.append(r["net_natural"] < MIN_EFFECT)
    v = "PASS" if d1 and all(d2) else ("FAIL" if not d1 else "PARTIAL")
    lines += [
        "",
        f"**Verdict: {v}** (D1 diagnostic {d1}; D2 net gain < 5 pts for every direction {all(d2)}).",
        "",
    ]
    return lines


def pilot(res: Path) -> list[str]:
    lines = ["## Phase 3 PILOT: self-sourced repair feasibility", ""]
    rd = _load(res / "pilot" / "readability.json")
    if rd is None:
        return lines + ["Readability: **NOT RUN**", ""]
    lines += [
        f"Steps: {rd['n_steps']}; train steps per variable (equalised): {rd['n_train_per_variable']}.",
        "",
        "| variable | probe | layer | held-out metric [95% CI] | shuffled | metric − shuffled | readable |",
        "|---|---|---|---|---|---|---|",
    ]
    readable = {}
    for v, per in rd["variables"].items():
        for kind, r in per.items():
            ok = r["test_metric"]["estimate"] >= MIN_METRIC and r["metric_minus_shuffled"]["lo"] > 0
            if kind == "logistic":
                readable[v] = ok
            lines.append(
                f"| {v} | {kind} | {r['layer']} | {r['metric_name']} {_ci(r['test_metric'])} | "
                f"{r['test_metric_shuffled']:.3f} | {_ci(r['metric_minus_shuffled'])} | {ok} |"
            )
    rq = rd["rq3"]
    t = rq["by_split"].get("test", {"n": 0})
    lines += [
        "",
        f"RQ3: {rq['n_distractor_failures']} distractor failures (main solved, NoOp failed), "
        f"{rq['n_content_attributable']} content-attributable (all splits).",
        "",
    ]
    if t.get("n", 0):
        rq3_ok = t["gold_top1_rate"]["estimate"] >= MIN_RECOVERY and t["difference"]["lo"] > 0
        lines += [
            "| test-template content-attributable failures | value [95% CI] |",
            "|---|---|",
            f"| n (templates) | {t['n']} ({t['n_templates']}); by variable {t['by_variable']} |",
            f"| gold value is probe top-1 | {_ci(t['gold_top1_rate'])} |",
            f"| shuffled-label probe | {_ci(t['gold_top1_rate_shuffled'])} |",
            f"| difference | {_ci(t['difference'])} |",
            f"| top-1 = the bound (wrong) value | {t['bound_top1_rate']:.3f} |",
            f"| op step-position prior = gold | {t['op_step_position_prior_rate']} |",
            "",
            f"**RQ3 pilot criterion (≥ 20% and > shuffled): {'MET' if rq3_ok else 'NOT MET'}.**",
            "",
        ]
    else:
        lines += ["RQ3: no test-template content-attributable failures.", ""]
    rp = _load(res / "pilot" / "repair.json")
    if rp is None:
        return lines + ["Repair test: **NOT RUN**", ""]
    lines += [
        f"Repair test: strength multiple {rp['selected_multiple']} (chosen on train templates); "
        f"self-sourced gate open on {rp['self_sourced_gate_open_rate_failures']} of failures / "
        f"{rp['self_sourced_gate_open_rate_solved']} of solved items.",
        "",
        "| condition | n fail / solved | items whose continuation changed | repair % [CI] | damage % [CI] | net gain, pts [CI] | p vs random |",
        "|---|---|---|---|---|---|---|",
    ]
    nl = rp["random null"]
    for name, r in rp.items():
        if not isinstance(r, dict) or "repair" not in r or name == "random null":
            continue
        lines.append(
            f"| {name} | {r['n_failures']} / {r['n_solved']} | {r.get('frac_items_changed', float('nan')):.2f} | "
            f"{_ci(r['repair'], True)} | {_ci(r['damage'], True)} | {_ci(r['net'], True)} | "
            f"{nl['p_value_net_ge'][name]:.3f} |"
        )
    lines += [
        f"| random null (n={nl['n']}) | | | {100 * nl['repair_mean']:.1f} (mean) | {100 * nl['damage_mean']:.1f} (mean) | "
        f"{100 * nl['net_mean']:.1f}; {100 * nl['net_q025']:.1f} to {100 * nl['net_q975']:.1f} | |",
        "",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/BASELINE_REPORT_TABLES.md")
    args = ap.parse_args()
    res = Path(args.results)
    lines = [
        "# Baseline report: generated tables",
        "",
        f"Generated by `make report` from `{res}/`. Do not edit by hand; the narrative and",
        "interpretation are in `docs/BASELINE_REPORT.md`. CIs: 95%, template-clustered",
        "bootstrap (10,000 resamples) unless stated (Baseline A resamples instances).",
        "",
    ]
    for section in (phase0, baseline_a, baseline_b, pilot):
        lines += section(res)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
