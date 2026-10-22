.PHONY: install test test-fast lint smoke phase0 traces baseline-a baseline-b pilot pilot-sample \
	report all debug legacy-pilot

PY ?= python3
# primary model: Gemma-2-2B-IT; secondary: HARNESS=configs/qwen2_5_1_5b.yaml
# (give the secondary its own RESULTS directory; see DECISIONS D13)
HARNESS ?= configs/gemma2_2b.yaml
# results root: every "results/..." path is written under $(RESULTS)
# (default: $RFD_RESULTS_DIR if set, else results/; a Google Drive folder on
# Colab). Each stage skips itself once complete and resumes mid-way after a
# crash, so re-running a target is always safe.
ifeq ($(origin RESULTS), undefined)
RESULTS := $(if $(RFD_RESULTS_DIR),$(RFD_RESULTS_DIR),results)
endif
export RFD_RESULTS_DIR := $(RESULTS)
export PYTHONPATH := src

install:
	pip install -r requirements.txt && pip install -e ".[dev]" --no-deps

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not legacy"

lint:
	ruff check src tests

# ---- quick GPU-machine check before the long runs (outputs in $(RESULTS)/smoke, not reported)
smoke:
	$(PY) -m rfd.experiments.smoke --harness $(HARNESS)

# ---- Phase 0: TL/HF equivalence + GSM-Symbolic traces (shared by B and the pilot)
phase0:
	$(PY) -m rfd.experiments.phase0_equivalence --config $(HARNESS) --out results/phase0

traces:
	$(PY) -m rfd.experiments.collect_traces generate --harness $(HARNESS)
	$(PY) -m rfd.experiments.collect_traces label --harness $(HARNESS)
	$(PY) -m rfd.experiments.collect_traces extract --harness $(HARNESS)

# ---- Phase 1: Baseline A (Sun et al.)
baseline-a:
	$(PY) -m rfd.experiments.baseline_a pilot --harness $(HARNESS)
	$(PY) -m rfd.experiments.baseline_a collect --harness $(HARNESS)
	$(PY) -m rfd.experiments.baseline_a analyze --harness $(HARNESS)
	$(PY) -m rfd.experiments.baseline_a reprompt --harness $(HARNESS)

# ---- Phase 2: Baseline B (Yuan et al.); needs `make traces`
baseline-b:
	$(PY) -m rfd.experiments.baseline_b probe --harness $(HARNESS)
	$(PY) -m rfd.experiments.baseline_b steer --harness $(HARNESS)

# ---- Phase 3: pilot of self-sourced repair; needs `make traces`
pilot:
	$(PY) -m rfd.experiments.pilot readability --harness $(HARNESS)
	$(PY) -m rfd.experiments.pilot repair --harness $(HARNESS)

pilot-sample:
	$(PY) -m rfd.experiments.pilot sample --harness $(HARNESS)

# ---- Phase 4: regenerate the numeric tables of docs/BASELINE_REPORT.md from results/
report:
	$(PY) -m rfd.experiments.report

# everything, in order (each step resumes / skips)
all: phase0 baseline-a traces baseline-b pilot pilot-sample report

# ---- tiny end-to-end run with Qwen2.5-0.5B (outputs in results/debug/, never reported)
debug:
	$(PY) -m rfd.experiments.collect_traces generate --harness configs/debug.yaml --out results/debug/traces --limit 96
	$(PY) -m rfd.experiments.collect_traces label --harness configs/debug.yaml --out results/debug/traces
	$(PY) -m rfd.experiments.collect_traces extract --harness configs/debug.yaml --out results/debug/traces
	$(PY) -m rfd.experiments.baseline_b probe --harness configs/debug.yaml --out results/debug/baseline_b --traces results/debug/traces
	$(PY) -m rfd.experiments.baseline_b steer --config configs/debug_baseline_b.yaml --harness configs/debug.yaml --out results/debug/baseline_b --traces results/debug/traces
	$(PY) -m rfd.experiments.pilot readability --config configs/debug_pilot.yaml --harness configs/debug.yaml --out results/debug/pilot --traces results/debug/traces
	$(PY) -m rfd.experiments.pilot repair --config configs/debug_pilot.yaml --harness configs/debug.yaml --out results/debug/pilot --traces results/debug/traces

# Archived SAE phase-segmentation pilot (see src/rfd/legacy/README.md).
legacy-pilot:
	bash scripts/legacy/run_pilot.sh
