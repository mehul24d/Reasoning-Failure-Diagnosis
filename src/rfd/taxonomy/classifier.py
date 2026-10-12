"""Heuristic failure-type classifier.

This is a FIRST-PASS heuristic, not a validated classifier -- it is the
reason `csv_template.py` exists: at least 20% of labels must be manually
checked (configs/experiment.yaml:annotation.taxonomy_manual_check_fraction)
before any claim is made about taxonomy accuracy. TODO for Semester 7:
replace/augment with a model- or feature-based classifier and report
agreement with manual labels.
"""

from __future__ import annotations

from typing import Any

from rfd.taxonomy.arithmetic_check import has_arithmetic_inconsistency

ERROR_TYPES = [
    "correct",
    "parsing_error",
    "arithmetic_error",
    "planning_error",
    "verification_error",
    "execution_error",
]


def classify_error(trace_record: dict[str, Any], segmentation: dict[str, Any] | None = None) -> str:
    if trace_record.get("correct"):
        return "correct"

    if trace_record.get("predicted_answer") is None:
        return "parsing_error"

    if has_arithmetic_inconsistency(trace_record["generated_trace"]):
        return "arithmetic_error"

    if segmentation is not None:
        phases_present = {seg["phase"] for seg in segmentation.get("segments", [])}
        if "operation_planning" not in phases_present:
            return "planning_error"
        if "self_verification" not in phases_present:
            return "verification_error"

    return "execution_error"


def build_taxonomy_rows(
    traces: list[dict[str, Any]], segmentations_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        segmentation = segmentations_by_id.get(trace["id"])
        error_type = classify_error(trace, segmentation)
        rows.append(
            {
                "id": trace["id"],
                "correct": trace.get("correct"),
                "predicted_answer": trace.get("predicted_answer"),
                "gold_answer": trace.get("gold_answer"),
                "error_type": error_type,
            }
        )
    return rows
