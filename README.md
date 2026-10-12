# Reasoning Failure Diagnosis

**Automated Phase Segmentation and Diagnostic Taxonomy of Mathematical
Reasoning Failures using SAE Feature Coactivation**

B.Tech Minor Project, Semester 7.

- **Team:** TODO — add team member names/roll numbers
- **Supervisor:** TODO — add supervisor name

> Scope note: this repo is the **minor project** only. It does not
> implement anything from the related major project (no feature
> injection, no circuit transfer) — see `docs/PROGRESS.md` for the
> boundary between the two.

## Research questions

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

## Pipeline

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

## Repo layout

```
src/rfd/
  data/           dataset loaders (GSM8K primary, MATH/BBH secondary)
  generation/     trace generation, answer extraction/scoring
  sae/            TransformerLens activation extraction, JumpReLU SAE
  segmentation/   coactivation graph, community detection, changepoints
  taxonomy/       heuristic failure classifier, CSV templates
  metrics/        segmentation + sparsity + taxonomy metrics
  utils/          config/IO/seed helpers
configs/          model / dataset / sae / segmentation / experiment YAML
scripts/          run_pilot.sh
tests/            pytest unit tests (synthetic data, no network/GPU)
docs/             annotation guidelines + schema, PROGRESS.md
results/          pilot outputs (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires internet access on first run to download the pilot model
(`Qwen/Qwen2.5-0.5B-Instruct`, ~1GB) and the GSM8K dataset from the
HuggingFace Hub.

## Running the pilot

```bash
bash scripts/run_pilot.sh
# or:
make pilot
```

This runs, in order: trace generation (50 GSM8K test problems) -> pilot
SAE training (on a disjoint 20-problem slice of the GSM8K train split) ->
SAE feature encoding of the pilot traces -> phase segmentation -> failure
taxonomy. Outputs land in `results/pilot/` (gitignored — see
`docs/PROGRESS.md` for what an actual run produced).

## Tests

```bash
make test   # or: python3 -m pytest -q
```

All unit tests run on synthetic data — no network access, no model
download, no GPU required.

## Current status

See `docs/PROGRESS.md` for the authoritative, non-fabricated account of
what has been run and what is still TODO. Short version:

- [x] Repo skeleton, configs, pytest suite (37 tests, synthetic data)
- [x] GSM8K loader wired end-to-end; MATH/BBH loaders implemented but not
      exercised beyond the pilot
- [x] Trace generation + answer extraction/scoring
- [x] TransformerLens activation extraction (confirmed working on
      Qwen2.5-0.5B-Instruct, 24 layers, d_model=896)
- [x] JumpReLU SAE: implemented and trained at pilot scale (small D, few
      steps) — **not** the Semester-7 target config (D=65536, L0≈64)
- [x] Coactivation graph -> Louvain -> changepoint segmentation pipeline,
      with a positional heuristic for the 4 phase labels
- [x] Boundary F1 / temporal IoU / phase recovery rate metrics
- [x] Heuristic failure taxonomy + 20%-manual-check CSV generator
- [ ] Manual annotation of 20 traces — **not done**, requires a human
      annotator (CLI tool is ready: `rfd.segmentation.annotate_cli`)
- [ ] Full-scale SAE training (D=65536) — TODO, Semester 7
- [ ] MATH / BBH-Logic full runs — TODO, Semester 7
- [ ] 3-model comparison — TODO, Semester 7
