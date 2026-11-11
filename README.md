# Diagnosing Reasoning Failures: Content vs. Evaluation Variables

DTU CSE minor project (BTP), Semester VII 2026–27, group DTU/CSE/BTP/GP099.
A matched probing study of operations, operands and intermediate results in language-model
arithmetic reasoning, and a feasibility test for self-sourced repair. The full design is in
the Detailed Research Plan; section and table numbers in the code refer to it.

## Layout (plan Table 23)

| Directory | Contents |
|---|---|
| `configs/` | Model and SAE IDs, hook sites, dtypes, every threshold (Appendix E) |
| `prereg/` | One pre-registration file per experiment (E0–E4), committed before the run |
| `harness/` | Model loading, equivalence checks, prompt templates, generation |
| `data/` | Problem sources, template splits, failure sets, annotations |
| `parsing/` | Step segmentation, equation extraction, alignment to gold programs |
| `cache/` | Residual and SAE activation caching scripts (data stored outside Git) |
| `probes/` | Direction and probe construction, shuffled-label nulls |
| `analysis/` | Readout maps, feasibility test, attribution, typology test, statistics |
| `reports/` | Generated tables and figures, the report |
| `scripts/` | Shell entry points for the GPU server |

## Where things run

* **GPU server** (≥ 40 GB CUDA GPU; 24 GB is enough for Gemma-2-2B): steps S1–S4
  (equivalence checks, SAE validation, generation, activation caching).
* **Laptop** (CPU or Apple MPS): parsing, probes on cached activations, statistics, report.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # 3.13: see requirements.txt
pip install -r requirements.txt
huggingface-cli login        # Gemma-2 is gated: accept the licence on its HF page first
pytest                       # fast + offline tests (tiny random model), no downloads
BTP_SMOKE=1 pytest tests/test_harness_smoke.py -s   # harness on a 0.5B model, CPU is fine
```

## Step S1 on the GPU server

```bash
bash scripts/s1_equivalence.sh gemma-2-2b-it     # then gemma-2-9b-it, qwen2.5-7b-instruct
```

Then, after `prereg/E0.md` is committed:

```bash
bash scripts/s1_sae.sh gemma-2-2b-it     # SAE validation + E0; then gemma-2-9b-it
```

Progress against the plan's work plan is tracked in `docs/STEPS.md`.
