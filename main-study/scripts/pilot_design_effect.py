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
    rep = {
        "pilot_file": str(PILOT.relative_to(REPO_ROOT.parent)),
        "pilot_sha256": hashlib.sha256(PILOT.read_bytes()).hexdigest(),
        "study_templates": len(study),
        "study_templates_missing_from_pilot": sorted(study - {int(r["template_id"]) for r in rows}),
        "primary": main_v,
        "sensitivity_noop_variant": noop_v,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    p = main_v
    print(f"pilot ({p['pilot_traces']} traces, {p['pilot_templates']} templates): accuracy "
          f"{p['pilot_accuracy']:.3f}, ICC {p['icc_failure']:.3f}, m_bar {p['m_bar']:.2f}, CV "
          f"{p['cv']:.3f}, m_eff {p['m_eff']:.2f}, deff {p['deff']:.2f} (unadjusted "
          f"{p['unadjusted_deff']:.2f}) -> N* = {p['required_test_failures_with_s0']}; expected "
          f"{p['expected_test_failures_with_s0']:.0f}; effective-n ceiling {p['effective_n_ceiling']:.0f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
