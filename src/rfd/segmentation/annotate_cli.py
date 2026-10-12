"""Small interactive CLI for manually annotating phase boundaries on a
sample of traces. See docs/ANNOTATION_GUIDELINES.md for the procedure and
docs/annotation_schema.json for the output schema.

Usage: python -m rfd.segmentation.annotate_cli --experiment configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rfd.segmentation.phase_segmenter import PHASES
from rfd.utils.config import load_experiment
from rfd.utils.io import ensure_parent, read_jsonl


def _print_trace_with_ruler(tokens: list[str]) -> None:
    for i in range(0, len(tokens), 5):
        chunk = tokens[i : i + 5]
        print(f"[{i:3d}] " + "".join(chunk).replace("\n", "\\n"))


def _annotate_one(trace: dict[str, Any], annotator: str) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"Trace id: {trace['id']}  (correct={trace.get('correct')})")
    print("-" * 70)
    _print_trace_with_ruler(trace["tokens"])
    print("-" * 70)
    print(f"Phases: {PHASES}")

    segments = []
    cursor = 0
    n_tokens = len(trace["tokens"])
    while cursor < n_tokens:
        raw_end = input(f"  segment starting at {cursor} -- end index (exclusive, blank=end of trace): ").strip()
        end = n_tokens if raw_end == "" else int(raw_end)
        phase = input(f"  phase for [{cursor}, {end}) {PHASES}: ").strip()
        if phase not in PHASES:
            print(f"  !! '{phase}' not a known phase, recording anyway")
        segments.append({"start": cursor, "end": end, "phase": phase})
        cursor = end

    notes = input("  notes (optional): ").strip()
    record = {"id": trace["id"], "annotator": annotator, "segments": segments}
    if notes:
        record["notes"] = notes
    return record


def run_annotation(experiment_cfg: dict[str, Any], annotator: str, n_traces: int) -> None:
    annotation_path = ensure_parent(experiment_cfg["annotation"]["annotation_file"])
    existing: list[dict[str, Any]] = []
    if Path(annotation_path).exists():
        existing = json.loads(Path(annotation_path).read_text())
    already_done = {rec["id"] for rec in existing if rec["annotator"] == annotator}

    traces = list(read_jsonl(experiment_cfg["traces_file"]))
    todo = [t for t in traces if t["id"] not in already_done][:n_traces]

    if not todo:
        print("Nothing left to annotate for this annotator.")
        return

    for trace in todo:
        record = _annotate_one(trace, annotator)
        existing.append(record)
        annotation_path.write_text(json.dumps(existing, indent=2))
        print(f"  saved ({len(existing)} total annotations in file)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--annotator", default="annotator1")
    parser.add_argument("--n-traces", type=int, default=None)
    args = parser.parse_args()

    experiment_cfg = load_experiment(args.experiment)
    n_traces = args.n_traces or experiment_cfg["annotation"]["n_traces_to_annotate"]
    run_annotation(experiment_cfg, args.annotator, n_traces)


if __name__ == "__main__":
    main()
