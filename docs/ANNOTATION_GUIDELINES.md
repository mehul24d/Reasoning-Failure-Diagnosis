# Manual Phase Annotation Guidelines

## Purpose

Manual annotations are the gold standard that `boundary_f1`, `temporal_iou`,
and `phase_recovery_rate` (in `src/rfd/metrics/segmentation_metrics.py`)
are evaluated against. Annotate at least **20 traces** (per
`configs/experiment.yaml:annotation.n_traces_to_annotate`), and at least
**20%** of all traces for the failure-taxonomy manual check
(`configs/experiment.yaml:annotation.taxonomy_manual_check_fraction`).

## The four phases

1. **problem_understanding** — restating/parsing the question, identifying
   given quantities and what is being asked. Usually the opening tokens.
2. **operation_planning** — deciding which operations/equations to use,
   without yet computing a numeric result.
3. **arithmetic_execution** — carrying out computations: arithmetic,
   substituting numbers, intermediate results.
4. **self_verification** — checking the computed result, restating the
   final answer, sanity-checking against the question.

Phases do not have to appear in this order, and may repeat (e.g. a trace
can plan, execute, re-plan, execute again, verify). Boundaries are drawn at
the **token index** where a new phase clearly begins.

## Annotation unit

Boundaries are token indices into the `tokens` list of a trace record in
`results/pilot/traces.jsonl` (0-indexed, over the *generated* tokens only —
not the prompt). A segment is `{"start": int, "end": int, "phase": str}`
with `start` inclusive and `end` exclusive, covering `[0, n_tokens)` with
no gaps or overlaps.

## Procedure

1. Run `python -m rfd.segmentation.annotate_cli --experiment
   configs/experiment.yaml` (see its `--help` for sampling options).
2. For each sampled trace, the CLI prints the generated trace text with
   token indices every 5 tokens as a ruler.
3. Read the trace, decide where each phase starts, and enter boundary
   token indices and a phase label per segment when prompted.
4. If a trace doesn't cleanly fit the 4-phase model (e.g. it fails before
   reaching arithmetic), annotate only the phases that are actually
   present — do not force all 4 phases into every trace.
5. Annotations are appended to `results/pilot/annotations.json`
   (schema: `docs/annotation_schema.json`), one record per trace, keyed by
   trace `id`. Re-running the CLI skips traces already annotated by the
   same annotator.

## Edge cases

- **Incorrect/garbled traces**: still annotate whatever phases are
  present; segmentation quality should be measured independently of
  answer correctness (that's what the failure taxonomy is for).
- **No verification present**: do not insert a `self_verification` segment
  that isn't there — omit the phase.
- **Ambiguous boundary**: prefer the token where the *topic* of the text
  changes (e.g. "So the total is" -> execution, "Let me check" ->
  verification) over exact punctuation.
