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
# + requirements-legacy.txt for the archived SAE code (make install does both)
make test-fast          # unit tests: synthetic data, no GPU, no network
make debug              # tiny end-to-end run with Qwen2.5-0.5B -> results/debug/ (not reported)
```

Full runs need a CUDA GPU; see "Compute" below. On Colab, run
`notebooks/run_on_gpu.ipynb`; on Kaggle, `notebooks/run_on_kaggle.ipynb` (see
"Running on Kaggle"). Each target writes under `$(RESULTS)` (default
`results/`), reads its configuration from `configs/*.yaml`, skips itself once
complete and resumes after a crash, so re-running a target is always safe:

```bash
export HF_TOKEN=...   # Gemma-2 is gated; read from the environment, never stored
make smoke        # few-minute check of memory, generation config, prompts (not reported)
make phase0       # TransformerLens vs HF equivalence + padding/greedy/generation-config checks
make baseline-a   # Sun et al.: arithmetic error pilot, probes, selective re-prompting
make traces       # GSM-Symbolic main + NoOp traces, step labels, activations
make baseline-b   # Yuan et al.: failure probes + steering with random-direction nulls
make pilot        # Phase 3 pilot: 2x2 variable readability, RQ3, mini repair test
make pilot-sample # optional: T=0.7 x5 samples -> stable vs borderline failures
make report       # regenerate docs/BASELINE_REPORT_TABLES.md (numbers + verdicts) from results/
make all          # all of the above, in order
```

The primary model is **Gemma-2-2B-IT** (`configs/gemma2_2b.yaml`: float32,
eager attention, no system role). The secondary model, Qwen2.5-1.5B-Instruct,
goes into its own results directory:
`make all HARNESS=configs/qwen2_5_1_5b.yaml RESULTS=results/qwen2_5_1_5b`.
A results directory is pinned to one model and refuses another. Pre-registrations
for every experiment are in `docs/prereg/`, and every decision taken along the
way is in `docs/DECISIONS.md`.

## Running on Kaggle

`notebooks/run_on_kaggle.ipynb` is written for **Save & Run All**: it asks
nothing, stops itself before Kaggle's 12-hour limit, and leaves the results as
the version's output (DECISIONS D16).

1. **Settings (once).**
   - Accelerator: *GPU T4 x2* (one GPU is used). Internet: *on*.
   - *Add-ons → Secrets*: `HF_TOKEN`, from an HF account that accepted the
     Gemma-2 licence, attached to the notebook.
2. **First session.** *Save Version → Save & Run All*. The notebook:
   - clones the repo to `/tmp`;
   - installs with `scripts/install_gpu_env.py`:
     - Kaggle's torch / triton / nvidia-* are frozen in a constraints file, so
       the preinstalled CUDA torch is never replaced (pip fails instead);
     - wheels only, so nothing is compiled;
     - the results-relevant libraries at their pinned versions;
     - Kaggle's own numpy / scipy / pandas / matplotlib are kept;
     - it writes `install_report.json`;
   - asserts `torch.cuda.is_available()`;
   - runs the tests and `make smoke`, stopping on any failure;
   - runs the phases in order until the session budget (11.5 h minus 25 min
     for zipping) runs out.
3. **Output** (`/kaggle/working`):
   - `results/`: the resume state, with `RFD_RESULTS_DIR=/kaggle/working/results`;
   - `results.zip`: activation shards are left out above 8 GB;
   - `session_status.json`: per-phase status, also appended to
     `results/sessions.jsonl`.
4. **Next sessions.** *Add Input → Your Work →* the previous version's output,
   or a dataset made from its `results.zip`, then *Save & Run All* again. The
   restore cell copies the most complete `results/` it finds in `/kaggle/input`
   without overwriting anything. Finished stages are skipped, and the stage cut
   off by the deadline resumes where it stopped.

Every cell re-creates its own state, so after a kernel restart you can re-run
just the cell that stopped. In the interactive editor, `/kaggle/working` does
not survive the end of a session, so use *Save & Run All* for long runs.

Expect 2-3 sessions in total (about 20-35 T4-hours against a weekly GPU quota of
about 30 h). When `session_status.json` says `all_complete`, download
`results.zip`.

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
  experiments/   smoke, phase0_equivalence, collect_traces, baseline_a, baseline_b, pilot, report
  utils/resume   results root (RFD_RESULTS_DIR), atomic writes, stage markers, shards, caches
  utils/session  time-limited notebook sessions: restore, deadline runner, zip (Kaggle)
  stats.py       template-clustered bootstrap, AUROC/AUPRC/ECE/Brier
  plotting.py    report figures
  legacy/        ARCHIVED SAE phase-segmentation code (see its README)
configs/         gemma2_2b (primary), qwen2_5_1_5b (secondary), traces, baseline_a,
                 baseline_b, pilot, debug; legacy/
docs/            BASELINE_REPORT.md, DECISIONS.md, prereg/, PROGRESS.md (archived)
tests/           pytest suite (synthetic data; tiny random Qwen2 for hook tests); legacy/
results/         committed measurement files (json/jsonl/csv/png); *.npy caches ignored
```

## Compute

- **Target: a Colab T4 (15 GB).** Gemma-2-2B-IT in float32 takes about
  10.5 GB, so batches are 8 for generation and 4 for teacher-forced passes;
  `make smoke` measures the worst-case peak.
  - It needs an HF token with the Gemma licence accepted, in the `HF_TOKEN`
    environment variable (a Colab secret in the notebook).
  - On a larger GPU, raise `generation.batch_size` and `forward_batch_size` in
    `configs/gemma2_2b.yaml`.
- **Resumability** (DECISIONS D15). Outputs are written atomically and in
  chunks, shards and per-condition caches under `$RFD_RESULTS_DIR`, a Drive
  folder in the notebook. A disconnected runtime loses at most the unit of work
  in progress.
- The project Mac (Apple M2, 8 GB) runs only the 0.5B debug pipeline
  (docs/DECISIONS.md D7/D8).

## Tests

```bash
make test        # everything, including tests/legacy (needs requirements-legacy.txt)
make test-fast   # current code only (no legacy dependencies needed)
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

