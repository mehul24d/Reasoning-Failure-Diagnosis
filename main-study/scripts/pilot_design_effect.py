"""Design effect from the pilot (prereg/S2.md amendment A11, DECISIONS D31).

Reads the baseline's Gemma-2-2B-IT traces on GSM-Symbolic `main` (repository root,
results/traces/traces.jsonl.gz: 97 templates x 4 instances, main and no-op variants), keeps the
`main` variant of the study templates, and writes the ICC, the Kish design effect and N*:

    python scripts/pilot_design_effect.py            # from main-study/
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from harness.config import REPO_ROOT, experiment_config
from harness.power import pilot_design_effect

PILOT = REPO_ROOT.parent / "results" / "traces" / "traces.jsonl.gz"
OUT = REPO_ROOT / "reports" / "m1" / "pilot_design_effect.json"


def main() -> None:
    exp = experiment_config()
    d = exp["data"]
    with gzip.open(PILOT, "rt") as f:
        rows = [json.loads(line) for line in f]
    study = set(d["study_templates"])
    n_prompts = 2 * exp["generation"]["paraphrases_per_style"]
    args = dict(study_templates=study, instances=d["instances_per_template"], n_prompts=n_prompts,
                s0_rate=d["power"]["planning_s0_rate"], power=d["power"],
                n_test_templates=d["template_split"]["test"])
    main_v = pilot_design_effect(rows, variant="main", **args)
    noop_v = pilot_design_effect(rows, variant="noop", **args)
    # A14/A15: paired distractor failures in the pilot (descriptive): clean right, no-op wrong
    from collections import defaultdict

    from harness.power import icc_anova, kish_deff

    pairs = defaultdict(dict)
    for r in rows:
        if int(r["template_id"]) in study:
            pairs[(int(r["template_id"]), r["instance"])][r["variant"]] = bool(r["correct"])
    full = {k: v for k, v in pairs.items() if "main" in v and "noop" in v}
    at_risk = defaultdict(list)
    for (t, _), v in full.items():
        if v["main"]:
            at_risk[t].append(0 if v["noop"] else 1)
    icc_p = icc_anova(at_risk)
    n_fail = sum(sum(g) for g in at_risk.values())
    rate = n_fail / len(full)
    sizes = [args["instances"] * n_prompts * args["s0_rate"] * sum(1 for (tt, _), v in full.items()
             if tt == t and v["main"] and not v["noop"]) / max(sum(1 for (tt, _) in full if tt == t), 1)
             for t in {t for t, _ in full}]
    k = kish_deff(sizes, icc_p or 0.0)
    paired = {"pairs": len(full), "clean_right": sum(v["main"] for v in full.values()),
              "distractor_failures": n_fail, "rate_of_all_pairs": rate,
              "rate_given_clean_right": n_fail / max(sum(v["main"] for v in full.values()), 1),
              "icc_distractor_failure_given_clean_right": icc_p, **k,
              "expected_test_failures_with_s0_at_8_instances": args["n_test_templates"] * args["instances"]
              * n_prompts * rate * args["s0_rate"],
              "role": "descriptive (A15): failure occurrence, not the primary outcome"}
    rep = {
        "pilot_file": str(PILOT.relative_to(REPO_ROOT.parent)),
        "pilot_sha256": hashlib.sha256(PILOT.read_bytes()).hexdigest(),
        "study_templates": len(study),
        "study_templates_missing_from_pilot": sorted(study - {int(r["template_id"]) for r in rows}),
        "primary": main_v,
        "sensitivity_noop_variant": noop_v,
        "paired_distractor_failures": paired,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    p = main_v
    print(f"pilot ({p['pilot_traces']} traces, {p['pilot_templates']} templates): accuracy "
          f"{p['pilot_accuracy']:.3f}, ICC {p['icc_failure']:.3f}, m_bar {p['m_bar']:.2f}, CV "
          f"{p['cv']:.3f}, m_eff {p['m_eff']:.2f}, deff {p['deff']:.2f} (unadjusted "
          f"{p['unadjusted_deff']:.2f}) -> N* = {p['required_test_failures_with_s0']}; expected "
          f"{p['expected_test_failures_with_s0']:.0f}; effective-n ceiling {p['effective_n_ceiling']:.0f}")
    print(f"paired distractor failures: {n_fail}/{len(full)} pairs ({rate:.4f}), ICC {icc_p:.3f}, "
          f"deff {k['deff']:.2f}, expected primary test failures with s0 at 8 instances "
          f"{paired['expected_test_failures_with_s0_at_8_instances']:.0f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
