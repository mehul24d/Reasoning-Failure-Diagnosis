"""`make estimate`: a timed smoke test on Kaggle T4s and a projection of the full minor-project run.

The smoke test runs the REAL pipeline code (harness.pipeline) on 64 GSM8K *train* problems x
6 prompts (never the GSM-Symbolic study population, as in M0): batched greedy generation,
scoring and positions, the TL verification + residual cache, the sampled runs, and a few traces
with the TL fallback engine. From the measured throughputs it projects GPU-hours per stage and
the disk use for the pre-registered sizes (configs/experiment.yaml), and compares the total with
the budget (`budget.max_t4_gpu_hours`). Over budget, the pre-registered rule of amendment A13 is
applied automatically (`apply_budget_rule`). Smoke outputs are never reported as results.

    python -m harness.estimate --model gemma-2-2b-it            # on Kaggle, GPU T4 x2
    python -m harness.estimate --allow-any-gpu                  # elsewhere (numbers not for the plan)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time

from .config import experiment_config, model_config
from .pipeline import (Run, generation_summary, n_gpus, read_jsonl, run_stages, stage_generate,
                       write_jsonl)


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def build_smoke_problems(run: Run, n: int, seed: int) -> None:
    from .datasets import gsm8k

    if run.problems.exists():
        return
    rows = gsm8k("train", n=n, seed=seed)
    write_jsonl(rows, run.problems)
    # every smoke problem is its own "template" and all are "test", so the sampled stage runs
    run.split.write_text(json.dumps({"train": [], "dev": [], "test": sorted(r["template_id"] for r in rows)}))


def _busy_seconds(recs: list[dict]) -> dict[int, float]:
    """GPU-busy seconds per worker: the sum of its batch times (each batch counted once)."""
    seen, busy = set(), {}
    for r in recs:
        k = (r["worker"], r["batch_keys_crc"])
        if k not in seen:
            seen.add(k)
            busy[r["worker"]] = busy.get(r["worker"], 0.0) + r["batch_seconds"]
    return busy


def throughput(recs: list[dict]) -> dict:
    busy = _busy_seconds(recs)
    toks = sum(r["n_new_tokens"] for r in recs)
    return {"new_tokens": toks, "gpu_busy_seconds": sum(busy.values()),
            "tokens_per_gpu_second": toks / max(sum(busy.values()), 1e-9),
            "mean_new_tokens": toks / max(len(recs), 1), "traces": len(recs)}


def _verify_load(stage_status: dict, vrecs: list[dict], in_sample: bool) -> float:
    """Stage wall time minus the time its workers spent on traces (in parallel): the model load."""
    per_worker = {}
    for r in vrecs:
        if r["in_eq_sample"] == in_sample:
            per_worker[r["worker"]] = per_worker.get(r["worker"], 0.0) + r["seconds"]
    return max(0.0, stage_status.get("seconds", 0) - max(per_worker.values(), default=0.0))


def sampled_failure_rate(pw: dict) -> float:
    """Natural-failure rate used to size the sampled runs: the higher of the planning value (M0,
    GSM8K) and the pilot (GSM-Symbolic main), so the compute projection errs on the high side."""
    return max(1 - pw["planning_accuracy"], 1 - pw.get("pilot_accuracy", pw["planning_accuracy"]))


def distractor_failure_rates(pw: dict) -> tuple[float, float]:
    """Rate of primary distractor failures per pair (A14): (high, low). High sizes the compute
    (planning accuracy x the pilot's P(fail with clause | right without), or the pilot's paired
    rate, whichever is larger); low sizes the power floor (the smaller one)."""
    if "pilot_distractor_failure_rate" not in pw:
        return 0.0, 0.0
    a = pw["planning_accuracy"] * pw["pilot_distractor_failure_given_clean_right"]
    b = pw["pilot_distractor_failure_rate"]
    return max(a, b), min(a, b)


def n_conditions(exp: dict) -> int:
    return len((exp["data"].get("population") or {}).get("conditions", ["clean"]))


def project(meas: dict, n_gpu: int, instances: int | None = None, sampled: bool = True,
            engine: str = "hf_batched") -> dict:
    exp = experiment_config()
    d, pw = exp["data"], exp["data"]["power"]
    mc = model_config(meas["model"])
    inst = instances or d["instances_per_template"]
    n_prompts = 2 * exp["generation"]["paraphrases_per_style"]
    n_cond = n_conditions(exp)
    n_greedy = d["n_templates"] * inst * n_prompts * n_cond
    n_test = d["min_test_templates"] * inst * n_prompts          # per condition
    k = exp["generation"]["sampled_subset"]["k"]
    n_sampled_natural = math.ceil(n_test * sampled_failure_rate(pw)) * k if sampled else 0
    n_sampled_primary = (math.ceil(n_test * distractor_failure_rates(pw)[0]) * k
                         if sampled and n_cond > 1 else 0)
    n_sampled = n_sampled_natural + n_sampled_primary
    g, v, s, f = meas["greedy"], meas["verify"], meas["sampled"], meas["tl_fallback"]
    load = meas["load_seconds_per_stage"]
    if engine == "tl" and not f:
        raise ValueError("no TL fallback measurement")
    gen_tps = f["tokens_per_gpu_second"] if engine == "tl" else g["tokens_per_gpu_second"]
    hours = {
        "generation": n_greedy * g["mean_new_tokens"] / gen_tps / 3600,
        "verification_and_caching": n_greedy * v["mean_seq_tokens"] * v["seconds_per_seq_token"] / 3600,
        "sampled_runs": n_sampled * s["mean_new_tokens"] / s["tokens_per_gpu_second"] / 3600,
    }
    loads = {k: n_gpu * load.get(k, 0) / 3600 if hours[k] > 0 else 0.0 for k in hours}   # every GPU loads the model
    gpu_hours = {k: hours[k] + loads[k] for k in hours}
    total = sum(gpu_hours.values())
    setup_h = meas["session_setup_minutes"] / 60
    wall = total / n_gpu
    sessions = max(1, math.ceil(wall / 11.0))
    quota = wall + sessions * setup_h
    row_bytes = (mc.n_layers + 1) * mc.d_model * 2
    disk = {
        "activation_store": n_greedy * meas["rows_per_trace"] * row_bytes,
        "traces_and_sidecars": n_greedy * meas["bytes_per_greedy_trace"]
                               + n_sampled * meas["bytes_per_sampled_trace"],
        "fp32_reference": exp["cache"]["fp16_check_traces"] * meas["rows_per_trace"] * row_bytes * 2,
    }
    disk["total"] = sum(disk.values())
    sf = exp["budget"]["estimate_safety_factor"]
    budget = exp["budget"]["max_t4_gpu_hours"]
    fallback_gen = n_greedy * g["mean_new_tokens"] / f["tokens_per_gpu_second"] / 3600 if f else None
    return {
        "sizes": {"greedy_traces": n_greedy, "test_traces": n_test * n_cond, "sampled_runs": n_sampled,
                  "sampled_runs_natural": n_sampled_natural, "sampled_runs_primary": n_sampled_primary,
                  "conditions": n_cond, "templates": d["n_templates"], "instances_per_template": inst,
                  "prompts": n_prompts, "sampled_runs_included": sampled},
        "engine": engine,
        "t4_gpu_hours": {k: round(x, 3) for k, x in gpu_hours.items()},
        "t4_gpu_hours_total": round(total, 3),
        "t4_gpu_hours_total_with_safety": round(total * sf, 3),
        "safety_factor": sf,
        "kaggle_quota_hours": round(quota, 3),
        "kaggle_sessions": sessions,
        "n_gpus": n_gpu,
        "disk_gb": {k: round(x / 1e9, 3) for k, x in disk.items()},
        "budget_t4_gpu_hours": budget,
        "exceeds_budget": total * sf > budget,
        "fallback_tl_generation_t4_gpu_hours": None if fallback_gen is None else round(fallback_gen, 3),
        "total_if_fallback_t4_gpu_hours": None if fallback_gen is None else
            round(total - gpu_hours["generation"] + fallback_gen + loads["generation"], 3),
    }


def apply_budget_rule(meas: dict, n_gpu: int, engine: str = "hf_batched") -> dict:
    """The over-budget rule of amendment A13 (DECISIONS D33), fixed before any measurement and
    applied automatically, in this order:
      1. reduce instances per template, never below the power floor: the fewest instances whose
         expected test failures with s0 reach the target from the 40 test templates. With the
         distractor population (A14, A15) the units are primary failures, the target is 320 (the
         deff <= 1.5 floor; the primary-outcome deff is not known before data) and the rate is
         the lower of the planning and pilot paired rates. Without it (smoke tests, the old
         natural-failure design) the target is the pilot N* at the planning accuracy;
      2. if still over, drop the sampled runs (the stable/borderline split is reported as not
         done).
    Returns the first projection, the steps taken and the final projection."""
    from .power import min_instances_for

    exp = experiment_config()
    d, pw = exp["data"], exp["data"]["power"]
    n_prompts = 2 * exp["generation"]["paraphrases_per_style"]
    cur = d["instances_per_template"]
    if n_conditions(exp) > 1:
        need = pw.get("primary_target_before_outcome_deff", pw["min_test_failures"])
        rate = distractor_failure_rates(pw)[1]
    else:
        need = (pw.get("deff") or {}).get("required_test_failures_with_s0", pw["min_test_failures"])
        rate = 1 - pw["planning_accuracy"]
    floor = min_instances_for(need, d["template_split"]["test"], n_prompts, rate, pw["planning_s0_rate"])
    first = project(meas, n_gpu, engine=engine)
    final, steps = first, []
    if first["exceeds_budget"]:
        if floor is not None and floor < cur:
            final = project(meas, n_gpu, instances=floor, engine=engine)
            steps.append({"step": "reduce_instances_to_power_floor", "applied": True,
                          "instances_per_template": floor, "from": cur, "power_floor_failures": need})
        else:
            steps.append({"step": "reduce_instances_to_power_floor", "applied": False,
                          "power_floor_failures": need, "floor_instances": floor,
                          "reason": "the power floor needs at least the planned instances"
                                    if floor is not None else
                                    "the power floor is above what any instance count reaches"})
        if final["exceeds_budget"]:
            final = project(meas, n_gpu, instances=final["sizes"]["instances_per_template"],
                            sampled=False, engine=engine)
            steps.append({"step": "drop_sampled_runs", "applied": True,
                          "consequence": "stable vs borderline split reported as not done"})
    return {"rule": exp["budget"]["over_budget_rule"], "engine": engine, "power_floor_failures": need,
            "power_floor_instances": floor, "first": first, "steps": steps, "final": final,
            "still_over_budget": final["exceeds_budget"]}


def markdown(rep: dict) -> str:
    p, m = rep["projection"], rep["measured"]
    lines = [
        f"# Compute estimate: {rep['model']} (`make estimate`, {rep['date']})",
        "",
        f"Measured on **{', '.join(rep['gpus']) or 'no GPU'}**, code `{rep['commit']}`, smoke test on "
        f"{m['greedy']['traces']} greedy traces (GSM8K train, never the study population).",
        "",
        "| Stage | measured | projected T4 GPU-hours |",
        "|---|---|---|",
        f"| Generation (HF batched, batch {rep['batch_size']}) | {m['greedy']['tokens_per_gpu_second']:.0f} "
        f"new tokens/s per GPU, mean {m['greedy']['mean_new_tokens']:.0f} tokens | {p['t4_gpu_hours']['generation']:.2f} |",
        f"| Verification + caching (TL) | {m['verify']['seconds_per_seq_token'] * 1000:.2f} ms per sequence "
        f"token, {m['verify']['seconds_per_trace']:.2f} s per trace | {p['t4_gpu_hours']['verification_and_caching']:.2f} |",
        f"| Sampled runs (T = 0.7, k = 5) | {m['sampled']['tokens_per_gpu_second']:.0f} new tokens/s per GPU, "
        f"mean {m['sampled']['mean_new_tokens']:.0f} tokens | {p['t4_gpu_hours']['sampled_runs']:.2f} |",
        f"| **Total** (model loads included) | | **{p['t4_gpu_hours_total']:.2f}** "
        f"(x{p['safety_factor']} = {p['t4_gpu_hours_total_with_safety']:.2f}) |",
        "",
        f"* Sizes: {p['sizes']['greedy_traces']} greedy traces ({p['sizes']['templates']} templates x "
        f"{p['sizes']['instances_per_template']} instances x {p['sizes']['prompts']} prompts x "
        f"{p['sizes'].get('conditions', 1)} condition(s)), {p['sizes']['sampled_runs']} sampled runs"
        f" ({p['sizes'].get('sampled_runs_primary', 0)} for primary distractor failures, "
        f"{p['sizes'].get('sampled_runs_natural', p['sizes']['sampled_runs'])} for natural failures).",
        f"* Kaggle quota: about **{p['kaggle_quota_hours']:.1f} h** of session time with {p['n_gpus']} GPU(s) "
        f"in parallel, in {p['kaggle_sessions']} session(s) (setup {m['session_setup_minutes']} min each).",
        f"* Disk: activation store {p['disk_gb']['activation_store']:.2f} GB, traces "
        f"{p['disk_gb']['traces_and_sidecars']:.2f} GB, total **{p['disk_gb']['total']:.2f} GB** "
        f"({m['rows_per_trace']:.1f} cached positions per trace).",
        f"* Smoke checks (not results): cap hit {m['greedy_summary']['hit_max_tokens_fraction']:.3f}, "
        f"stop reasons {m['greedy_summary']['stop_reasons']}, TL reproduction {m['verify']['reproduction']:.4f}, "
        f"max logit diff {m['verify']['max_abs_logit_diff']}, float16 max rel. error "
        f"{(m['verify']['fp16_check'] or {}).get('max_rel_err')}.",
        f"* If the equivalence check failed, TL generation would cost about "
        f"{p['fallback_tl_generation_t4_gpu_hours']} T4 GPU-hours for generation "
        f"(total {p['total_if_fallback_t4_gpu_hours']}).",
        "",
        ("**Over budget** (" if p["exceeds_budget"] else "Within budget (")
        + f"{p['budget_t4_gpu_hours']} T4 GPU-hours, safety factor included)."
        + (" The pre-registered budget rule (A13) applies; see below." if p["exceeds_budget"] else ""),
    ]
    br = rep.get("budget_rule")
    if br and br["steps"]:
        f = br["final"]
        lines += ["", "**Budget rule (A13), applied automatically:**"]
        for st in br["steps"]:
            lines.append(f"* {st['step']}: " + ("applied" if st["applied"] else "not possible")
                         + "".join(f", {k} = {v}" for k, v in st.items() if k not in ("step", "applied")))
        lines.append(f"* Final plan: {f['sizes']['instances_per_template']} instances per template, "
                     f"sampled runs {'kept' if f['sizes']['sampled_runs_included'] else 'dropped'}: "
                     f"{f['t4_gpu_hours_total']:.2f} T4 GPU-hours (x{f['safety_factor']} = "
                     f"{f['t4_gpu_hours_total_with_safety']:.2f}), "
                     + ("**still over budget: the run does not start; the group decides.**"
                        if br["still_over_budget"] else "within budget."))
    return "\n".join(lines) + "\n"


def top_up_by_budget(meas: dict, n_gpu: int, instances: int = 16) -> dict:
    """Amendment A20 (D42), decided before any study data: generate instances 9-16 for all study
    templates iff the projected total with 16 instances (both conditions, sampled runs included,
    x the safety factor, the same comparison as the budget rule) stays within the budget;
    otherwise no top-up and the study is reported as underpowered."""
    p = project(meas, n_gpu, instances=instances)
    fits = not p["exceeds_budget"]
    return {"rule": "generate instances 9-16 iff projected T4-hours with 16 instances x safety <= budget",
            "instances": instances, "projection": p,
            "t4_gpu_hours_total": p["t4_gpu_hours_total"],
            "t4_gpu_hours_total_with_safety": p["t4_gpu_hours_total_with_safety"],
            "budget_t4_gpu_hours": p["budget_t4_gpu_hours"], "fits": fits,
            "decision": "generate_instances_9_16" if fits else "skip",
            "reported_as": None if fits else "underpowered"}


def reproject(model: str) -> dict:
    """Projection of the CURRENT design from the measured throughputs (no GPU needed). Writes
    reports/m1/estimate_<model>_reprojected.{json,md}; the measurement file is not changed."""
    from .config import REPO_ROOT
    from .pipeline import git_commit

    src = REPO_ROOT / "reports" / "m1" / f"estimate_{model}.json"
    old = json.loads(src.read_text())
    meas, n_gpu = old["measured"], len(old["gpus"]) or 1
    rep = {**{k: old[k] for k in ("model", "gpus", "batch_size")},
           "date": time.strftime("%Y-%m-%d"), "commit": old["commit"], "reprojected_at_commit": git_commit(),
           "measured_on": {"date": old["date"], "commit": old["commit"], "file": src.name},
           "measured": meas, "projection": project(meas, n_gpu),
           "budget_rule": apply_budget_rule(meas, n_gpu),
           "if_equivalence_fails": apply_budget_rule(meas, n_gpu, engine="tl"),
           "top_up": top_up_by_budget(meas, n_gpu)}
    out = src.with_name(f"estimate_{model}_reprojected.json")
    out.write_text(json.dumps(rep, indent=2, default=str))
    md = markdown(rep).replace("(`make estimate`,", "(re-projected for the current design,")
    fb = rep["if_equivalence_fails"]
    md += ("\n**If the equivalence gate fails (TL generation), with the budget rule:** first projection "
           f"{fb['first']['t4_gpu_hours_total']:.2f} T4 GPU-hours (x1.25 = {fb['first']['t4_gpu_hours_total_with_safety']:.2f}); "
           + "; ".join(f"{st['step']}: {'applied' if st['applied'] else 'not possible'}" for st in fb["steps"])
           + f"; final {fb['final']['t4_gpu_hours_total_with_safety']:.2f} with safety: "
           + ("**still over budget: the run does not start; the group decides.**" if fb["still_over_budget"]
              else "within budget.") + "\n")
    tu = rep["top_up"]
    md += (f"\n**Top-up (A20):** with instances 9-16 the projection is {tu['t4_gpu_hours_total']:.2f} T4 GPU-hours "
           f"(x{tu['projection']['safety_factor']} = {tu['t4_gpu_hours_total_with_safety']:.2f}) against "
           f"{tu['budget_t4_gpu_hours']}: " + ("**generate instances 9-16.**" if tu["fits"] else
           "**no top-up; the study is reported as underpowered.**") + "\n")
    md += f"\nThroughputs measured on {old['date']} (code `{old['commit']}`), {src.name}.\n"
    out.with_suffix(".md").write_text(md)
    print(md)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--n-problems", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sampled-per-worker", type=int, default=80)
    ap.add_argument("--tl-fallback-traces", type=int, default=6)
    ap.add_argument("--setup-minutes", type=float, default=15.0,
                    help="per-session install/restore time (the notebook passes its measurement)")
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete an earlier smoke run first")
    ap.add_argument("--reproject", action="store_true",
                    help="no GPU: re-project the saved measurement (reports/m1/estimate_<model>.json) "
                         "for the current configs, and apply the budget rule for both engines")
    a = ap.parse_args()
    if a.reproject:
        reproject(a.model)
        return

    gpus = gpu_names()
    if not a.allow_any_gpu and not (gpus and all("T4" in g for g in gpus)):
        raise SystemExit(f"make estimate must run on Kaggle T4s (found {gpus or 'no GPU'}); "
                         "--allow-any-gpu runs it anyway, but the numbers are then not for the plan")
    run = Run.smoke(a.model)
    if a.fresh and run.base.exists():
        shutil.rmtree(run.base)
    build_smoke_problems(run, a.n_problems, a.seed)
    exp = experiment_config()
    t0 = time.time()
    log = run_stages(run, ["generate", "score", "verify", "cache"], require_binding=False)
    if any(x["status"] not in ("complete", "complete (skipped)") for x in log):
        raise SystemExit(f"smoke run did not finish: {log}")
    st = {s: run.status(s) for s in ("generate", "verify", "cache")}
    smp = run.status("sampled")
    if smp["status"] != "complete":
        smp = stage_generate(run, "sampled", limit=a.sampled_per_worker)
        smp.pop("_status", None)
        run.set_status("sampled", "complete", **smp)
    # TL fallback engine on a few traces (its own directory)
    fb = Run.smoke(a.model)
    fb.base = run.base / "tl_fallback"
    fb.problems, fb.split = run.problems, run.split
    fb_info = stage_generate(fb, "greedy", engine="tl", limit=a.tl_fallback_traces, n_workers=1)
    fb_recs = read_jsonl(fb.traces) if fb.traces.exists() else []
    total_s = time.time() - t0

    greedy = read_jsonl(run.traces)
    sampled = read_jsonl(run.sampled)
    vrecs = [r for f in sorted((run.base / "parts").glob("verify.part*.jsonl")) for r in read_jsonl(f)]
    from cache.verify_cache import summarise

    vs = summarise(vrecs)
    score = run.status("score")

    def load_s(stage_status, busy):
        return max(0.0, stage_status.get("seconds", 0) - max(busy.values(), default=0.0))

    meas = {
        "model": a.model,
        "greedy": throughput(greedy), "greedy_summary": generation_summary(greedy),
        "sampled": throughput(sampled),
        "tl_fallback": throughput(fb_recs) if fb_recs else None,
        "verify": {**{k: vs[k] for k in ("reproduction", "max_abs_logit_diff", "traces_with_any_mismatch",
                                         "fp16_check", "seconds_per_trace", "seconds_per_seq_token", "tokens")},
                   "mean_seq_tokens": sum(r["n_seq_tokens"] for r in vrecs) / max(len(vrecs), 1)},
        "rows_per_trace": score["rows_per_trace"],
        "char_map_failed": score["char_map_failed"],
        "bytes_per_greedy_trace": (run.traces.stat().st_size + run.top2.stat().st_size) / max(len(greedy), 1),
        "bytes_per_sampled_trace": run.sampled.stat().st_size / max(len(sampled), 1),
        "load_seconds_per_stage": {
            "generation": load_s(st["generate"], _busy_seconds(greedy)),
            "verification_and_caching": _verify_load(st["verify"], vrecs, True)
                                        + _verify_load(st["cache"], vrecs, False),
            "sampled_runs": load_s(smp, _busy_seconds(sampled)),
        },
        "session_setup_minutes": a.setup_minutes,
        "smoke_wall_seconds": round(total_s, 1),
        "fallback_info": {k: fb_info.get(k) for k in ("seconds", "lines")},
    }
    n_gpu = max(1, n_gpus())
    rep = {"model": a.model, "date": time.strftime("%Y-%m-%d"), "gpus": gpus,
           "commit": __import__("harness.pipeline", fromlist=["git_commit"]).git_commit(),
           "batch_size": exp["generation"]["batched"]["batch_size"],
           "measured": meas, "projection": project(meas, n_gpu),
           "budget_rule": apply_budget_rule(meas, n_gpu)}
    run.reports.mkdir(parents=True, exist_ok=True)
    out = run.reports / f"estimate_{a.model}.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    md = markdown(rep)
    (run.reports / f"estimate_{a.model}.md").write_text(md)
    print(md)
    print(f"wrote {out}")
    raise SystemExit(2 if rep["budget_rule"]["still_over_budget"] else 0)


if __name__ == "__main__":
    main()
