"""Segmentation evaluation metrics: boundary F1 (with tolerance window),
temporal IoU per phase, and phase recovery rate.

All three compare a predicted segmentation (list of {start, end, phase})
against a manually annotated gold segmentation of the same shape.
"""

from __future__ import annotations

from typing import Any


def _boundaries_of(segments: list[dict[str, Any]]) -> list[int]:
    """Internal boundaries only (excludes 0 and the trace length)."""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    return [s["start"] for s in sorted_segs[1:]]


def boundary_f1(
    pred_segments: list[dict[str, Any]], gold_segments: list[dict[str, Any]], tolerance: int = 2
) -> dict[str, float]:
    pred_b = _boundaries_of(pred_segments)
    gold_b = _boundaries_of(gold_segments)

    if not pred_b and not gold_b:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_b or not gold_b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gold: set[int] = set()
    true_positives = 0
    for p in pred_b:
        candidates = [g for g in gold_b if g not in matched_gold and abs(g - p) <= tolerance]
        if candidates:
            closest = min(candidates, key=lambda g: abs(g - p))
            matched_gold.add(closest)
            true_positives += 1

    precision = true_positives / len(pred_b)
    recall = true_positives / len(gold_b)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _merged_intervals_by_phase(segments: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    by_phase: dict[str, list[tuple[int, int]]] = {}
    for seg in segments:
        by_phase.setdefault(seg["phase"], []).append((seg["start"], seg["end"]))
    return by_phase


def _interval_set_iou(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    def covered_tokens(intervals: list[tuple[int, int]]) -> set[int]:
        tokens: set[int] = set()
        for start, end in intervals:
            tokens.update(range(start, end))
        return tokens

    tokens_a, tokens_b = covered_tokens(a), covered_tokens(b)
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(tokens_a & tokens_b) / len(union)


def temporal_iou(
    pred_segments: list[dict[str, Any]], gold_segments: list[dict[str, Any]], phases: list[str]
) -> dict[str, float]:
    pred_by_phase = _merged_intervals_by_phase(pred_segments)
    gold_by_phase = _merged_intervals_by_phase(gold_segments)

    per_phase = {
        phase: _interval_set_iou(pred_by_phase.get(phase, []), gold_by_phase.get(phase, []))
        for phase in phases
    }
    # Average only over phases that actually appear in gold and/or pred --
    # otherwise phases absent from both sides would vacuously inflate the
    # mean toward 1.0.
    present_phases = [p for p in phases if p in gold_by_phase or p in pred_by_phase]
    per_phase["mean"] = (
        sum(per_phase[p] for p in present_phases) / len(present_phases) if present_phases else 1.0
    )
    return per_phase


def phase_recovery_rate(
    pred_segments: list[dict[str, Any]],
    gold_segments: list[dict[str, Any]],
    phases: list[str],
    iou_threshold: float = 0.1,
) -> float:
    """Fraction of gold phases that are present in gold AND have >= some
    predicted overlap (IoU >= iou_threshold) with the same phase label."""
    gold_by_phase = _merged_intervals_by_phase(gold_segments)
    pred_by_phase = _merged_intervals_by_phase(pred_segments)

    present_phases = [p for p in phases if p in gold_by_phase]
    if not present_phases:
        return 0.0

    recovered = sum(
        1
        for phase in present_phases
        if _interval_set_iou(pred_by_phase.get(phase, []), gold_by_phase[phase]) >= iou_threshold
    )
    return recovered / len(present_phases)
