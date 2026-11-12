"""Power floor, Kish design effect and instance sizing (prereg/S2.md amendments A6, A11, A13).

Design effect (decision of 2026-11-13, DECISIONS D31):

    deff = 1 + (m_eff - 1) * ICC,   m_eff = m_bar * (1 + CV^2) = sum(m_i^2) / sum(m_i)

* ICC: one-way ANOVA intraclass correlation of the failure indicator (1 = wrong answer) across
  templates, with the unequal-group-size n0 of the ANOVA estimator.
* m_i: the number of analysis units (test failures with s0) in template i, so m_bar is the mean
  cluster size and CV the coefficient of variation of the cluster sizes. m_bar * (1 + CV^2) is
  Kish's adjustment for unequal cluster sizes; templates with no failure add nothing to it.
* Required test failures with s0: N* = max(320, ceil(213 * deff)) (synopsis power analysis).

`pilot_design_effect` estimates ICC and the cluster sizes from the pilot (the baseline's
Gemma-2-2B-IT traces on GSM-Symbolic `main`, `results/traces/traces.jsonl.gz`); that value fixes
N* before any S2 data exist. `harness.pipeline.design_effect` recomputes the same quantities on
the S2 traces as a reported check.
"""
from __future__ import annotations

import math
from collections import defaultdict


def icc_anova(groups: dict) -> float | None:
    """One-way ANOVA ICC of a 0/1 indicator, groups = {cluster: [0/1, ...]}. Clipped at 0."""
    groups = {k: v for k, v in groups.items() if v}
    k = len(groups)
    n = sum(len(g) for g in groups.values())
    if k < 2 or n <= k:
        return None
    grand = sum(sum(g) for g in groups.values()) / n
    msb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values()) / (k - 1)
    msw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups.values()) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups.values()) / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    return max(0.0, (msb - msw) / den) if den > 0 else 0.0


def kish_deff(sizes: list[float], icc: float) -> dict:
    """Kish design effect with the unequal-cluster-size adjustment."""
    pos = [s for s in sizes if s > 0]
    if not pos:
        return {"m_bar": 0.0, "cv": None, "m_eff": 0.0, "deff": 1.0}
    m_bar = sum(pos) / len(pos)
    var = sum((s - m_bar) ** 2 for s in pos) / len(pos)
    cv = math.sqrt(var) / m_bar
    m_eff = m_bar * (1 + cv ** 2)              # = sum(m^2) / sum(m)
    return {"m_bar": m_bar, "cv": cv, "m_eff": m_eff, "deff": 1 + (m_eff - 1) * icc}


def required_failures(deff: float, power: dict) -> int:
    return max(power["min_test_failures"], math.ceil(power["n_after_holm"] * deff))


def expected_test_failures(instances: int, n_test_templates: int, n_prompts: int,
                           failure_rate: float, s0_rate: float) -> float:
    return n_test_templates * instances * n_prompts * failure_rate * s0_rate


def min_instances_for(target: int, n_test_templates: int, n_prompts: int, failure_rate: float,
                      s0_rate: float, max_instances: int = 50) -> int | None:
    """Fewest instances per template whose expected test failures with s0 reach `target`, with
    the design effect held at its pre-registered value. None if even `max_instances` (all of
    GSM-Symbolic's 50 per template) cannot."""
    per = expected_test_failures(1, n_test_templates, n_prompts, failure_rate, s0_rate)
    n = math.ceil(target / per) if per > 0 else None
    return n if n is not None and n <= max_instances else None


def effective_n_ceiling(n_test_templates: int, icc: float) -> float | None:
    """n / deff tends to n_templates / ICC as instances grow (equal clusters): the most any number
    of instances can give."""
    return None if not icc else n_test_templates / icc


def pilot_design_effect(pilot_rows: list[dict], study_templates: set[int], instances: int,
                        n_prompts: int, s0_rate: float, power: dict, n_test_templates: int,
                        variant: str = "main") -> dict:
    """ICC and Kish deff from the pilot traces, projected to the planned S2 cluster sizes.

    m_i = instances x prompts x s0_rate x (pilot failure rate of template i): the expected test
    failures with s0 template i would contribute in S2."""
    groups = defaultdict(list)
    for r in pilot_rows:
        if r.get("variant", "main") != variant or int(r["template_id"]) not in study_templates:
            continue
        groups[int(r["template_id"])].append(0 if r["correct"] else 1)
    icc = icc_anova(groups)
    rates = {t: sum(g) / len(g) for t, g in groups.items()}
    fail = sum(sum(g) for g in groups.values()) / max(sum(len(g) for g in groups.values()), 1)
    sizes = [instances * n_prompts * s0_rate * f for f in rates.values()]
    k = kish_deff(sizes, icc or 0.0)
    need = required_failures(k["deff"], power)
    exp_fail = expected_test_failures(instances, n_test_templates, n_prompts, fail, s0_rate)
    return {
        "variant": variant, "pilot_templates": len(groups),
        "pilot_traces": sum(len(g) for g in groups.values()),
        "instances_per_template_in_pilot": sorted({len(g) for g in groups.values()}),
        "pilot_failure_rate": fail, "pilot_accuracy": 1 - fail,
        "templates_without_failure": sum(1 for f in rates.values() if f == 0),
        "icc_failure": icc, **k,
        "unadjusted_deff": 1 + (k["m_bar"] - 1) * (icc or 0.0),
        "required_test_failures_with_s0": need,
        "expected_test_failures_with_s0": exp_fail,
        "meets_target": exp_fail >= need,
        "instances_needed_at_fixed_deff": min_instances_for(need, n_test_templates, n_prompts,
                                                            fail, s0_rate),
        "effective_n_ceiling": effective_n_ceiling(n_test_templates, icc or 0.0),
        "n_after_holm": power["n_after_holm"],
        "sizing": {"instances": instances, "prompts": n_prompts, "s0_rate": s0_rate,
                   "test_templates": n_test_templates},
    }
