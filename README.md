# Reasoning Failure Diagnosis

**Self-Sourced Repair: Correcting Mathematical Reasoning Failures in Open
LLMs with Their Own Content Representations**

B.Tech Minor Project, Semester 7, DTU (Group GP099).

- **Team:** Mehul Dahiya (23/CS/253), Neil Negi (23/CS/278), Harshit Yadav (23/CS/172)
- **Supervisor:** Prof. Shailender Kumar

Core idea: a model's internal variables split into *content* variables
(consumed by the forward pass: the chosen operation, a bound operand, an
intermediate result, whether a distractor quantity is bound) and
*evaluation* variables (assessments: is this step correct, what kind of
error, is the final answer correct). Hypothesis: editing content variables
repairs reasoning; editing evaluation variables does not. *Self-sourced
repair* reads the model's own decoded correct value and writes it where the
computation uses it, only when the decoded and bound values disagree.

This repo currently holds the **baseline / feasibility work** for that
direction (replications of Sun et al. 2025 and Yuan et al. 2026, plus a
small pilot of our method) — see `docs/BASELINE_REPORT.md`.

> **Archived direction.** The earlier SAE phase-segmentation pipeline
> (sections below, `rfd.legacy.*`) is deprecated and kept only for
> reproducibility; see `src/rfd/legacy/README.md`.

## Research questions (baseline / feasibility stage)

- **RQ1 (readability):** are content variables (chosen operation, whether a
  distractor quantity is bound) and evaluation variables (step correct, error
  type) linearly readable from the residual stream at the start of a step?
- **RQ2 (causal asymmetry):** does editing content variables repair reasoning
  while editing evaluation variables does not?
- **RQ3 (self-sourcing):** in failures, is the *correct* content value still
  decodable from the model's own activations, so that it can be written back?

The baseline work replicates the two closest papers first:
**A** = Sun et al. 2025 (correct answers are decodable even when the output is
wrong); **B** = Yuan et al. 2026 (a correctness probe predicts failure well,
but steering along it barely repairs). See `docs/BASELINE_REPORT.md`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
make test-fast          # unit tests: synthetic data, no GPU, no network
make debug              # tiny end-to-end run with Qwen2.5-0.5B -> results/debug/ (not reported)
```

Full runs (GPU recommended; see "Compute" below). Each target writes under
`results/` and reads its configuration from `configs/*.yaml`:

```bash
make phase0       # TransformerLens vs HF equivalence + padding/greedy checks
make baseline-a   # Sun et al.: arithmetic error pilot, probes, selective re-prompting
make traces       # GSM-Symbolic main + NoOp traces, step labels, activations
make baseline-b   # Yuan et al.: failure probes + steering with random-direction nulls
make pilot        # Phase 3 pilot: 2x2 variable readability, RQ3, mini repair test
make pilot-sample # optional: T=0.7 x5 samples -> stable vs borderline failures
make report       # regenerate docs/BASELINE_REPORT_TABLES.md (numbers + verdicts) from results/
```

`HARNESS=configs/other.yaml make baseline-b` swaps the model / device
configuration. Pre-registrations for every experiment are in `docs/prereg/`,
and every decision taken along the way is in `docs/DECISIONS.md`.

## Repo layout

```
src/rfd/
  data/          GSM8K loaders, GSM-Symbolic + gold programs, NoOp distractors, arithmetic
  generation/    answer extraction and scoring, HF model loading
  harness/       hooks (capture / add / project / patch), batched greedy generation,
                 prompts, TL-equivalence check support
  parsing/       evaluation-validated equation parser, step labeller (vs gold program)
  probing/       grouped-CV logistic / diff-in-means probes, shuffled-label nulls
  steering/      steering engine with the exact greedy divergence shortcut
  experiments/   phase0_equivalence, collect_traces, baseline_a, baseline_b, pilot, report
  stats.py       template-clustered bootstrap, AUROC/AUPRC/ECE/Brier
  plotting.py    report figures
  legacy/        ARCHIVED SAE phase-segmentation code (see its README)
configs/         harness, traces, baseline_a, baseline_b, pilot, debug; legacy/
docs/            BASELINE_REPORT.md, DECISIONS.md, prereg/, PROGRESS.md (archived)
tests/           pytest suite (synthetic data; tiny random Qwen2 for hook tests); legacy/
results/         committed measurement files (json/jsonl/csv/png); *.npy caches ignored
```

## Compute

- The project Mac (Apple M2, 8 GB) can run only the 0.5B debug pipeline next
  to normal desktop apps (docs/DECISIONS.md D7/D8). The primary model
  (Qwen2.5-1.5B-Instruct, fp16 3.3 GB) needs about 4.5 GB free plus headroom.
- On a CUDA GPU (bf16), raise `generation.batch_size` and
  `forward_batch_size` in `configs/harness.yaml` as memory allows.
- Outputs are resumable where it matters (`collect_traces generate` appends in
  chunks).
- Gemma-2-2B-IT (the brief's primary model) needs an HF token with the licence
  accepted. Switching means changing `model_name` and re-running every phase;
  do not mix models in one report.

## Tests

```bash
make test        # everything, including tests/legacy (marker: legacy)
make test-fast   # current code only
```

## Archived: SAE phase segmentation (deprecated)

### Research questions

- **RQ1:** Can a language model's mathematical reasoning trace be
  automatically segmented into interpretable phases (problem
  understanding, operation planning, arithmetic execution,
  self-verification) using the coactivation structure of its SAE
  features, without supervision from labeled phase boundaries?
- **RQ2:** Do specific failure modes (arithmetic slips, planning errors,
  execution errors, failure to self-verify, answer-parsing failures)
  correspond to distinguishable patterns in SAE feature sparsity (L0) and
  feature entropy, such that a taxonomy of failures can be built from
  activation statistics rather than from the final answer alone?

### Pipeline

```
 GSM8K / MATH / BBH-Logic
          |
          v
 [1] generation: greedy decode trace, extract+score answer
          |   (rfd.generation.generate_traces)
          v
   traces.jsonl {id, prompt, trace, tokens, pred, gold, correct}
          |
          v
 [2] SAE: TransformerLens hook_resid_post -> JumpReLU SAE encode
          |   (rfd.sae.train_sae, rfd.sae.build_sae_features)
          v
   per-token SAE feature matrix [n_tokens, d_sae] + L0 + entropy
          |
          v
 [3] segmentation: feature coactivation graph -> Louvain communities
     -> community activation time series -> changepoint detection
          |   (rfd.segmentation.*)
          v
   predicted phase boundaries (4 phases) --compared against-->
   manual annotations (rfd.segmentation.annotate_cli)
          |                                      |
          v                                      v
   boundary F1 / temporal IoU / phase recovery rate (rfd.metrics)
          |
          v
 [4] taxonomy: heuristic error classifier (parsing / arithmetic /
     planning / execution / verification) + per-type L0/entropy/
     sparsity-collapse-delta, with a 20% manual-check CSV
          |   (rfd.taxonomy.*)
          v
   taxonomy.csv, metrics.json
```

The archived pilot is reproducible with `make legacy-pilot` (configs in
`configs/legacy/`, outputs in `results/legacy_pilot/`); its write-up is
`docs/PROGRESS.md`.

