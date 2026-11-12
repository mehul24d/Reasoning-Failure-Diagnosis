#!/usr/bin/env bash
# S1c (M0): SAE validation (Section 3.12.2) and the E0 sparsity check (Appendix A) for one model.
# Commit prereg/E0.md BEFORE running this.
# Usage: bash scripts/s1_sae.sh gemma-2-2b-it [n_gpus_for_TL]     (Kaggle 2 x T4: ... gemma-2-2b-it 2)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?gemma-2-2b-it or gemma-2-9b-it}"
NDEV="${2:-1}"
NLAYERS="$(python -c "from harness.config import model_config as m; print(m('$MODEL').n_layers)")"
D=data/generated/$MODEL; R=reports/m0; mkdir -p "$D" "$R"

# 1. 500 GSM8K-train problems x 2 prompt styles (paraphrase 0) = 1,000 traces (Appendix A)
python -m harness.datasets --source gsm8k --split train --n 500 --seed 0 --out data/m0_gsm8k_train500.jsonl
python - <<PY
from harness.generate import run
from harness.prompts import PromptSpec
run("$MODEL", "data/m0_gsm8k_train500.jsonl", "$D/m0_traces.jsonl",
    specs=[PromptSpec("marked", 0), PromptSpec("terse", 0)])
PY
python -m harness.score --traces "$D/m0_traces.jsonl" --out "$D/m0_traces_scored.jsonl"

# 2. SAE validation on the first 200 traces, every layer
python -m analysis.sae_validation --model "$MODEL" --traces "$D/m0_traces_scored.jsonl" \
  --max-traces 200 --n-devices "$NDEV" --out "$R/sae_validation_${MODEL}.json"

# 3. E0: token records on passing layers (3 per depth band), then the matched analysis
python -m analysis.sparsity_e0 collect --model "$MODEL" --traces "$D/m0_traces_scored.jsonl" \
  --sae-report "$R/sae_validation_${MODEL}.json" --max-traces 1000 --n-devices "$NDEV" \
  --out "$D/e0_tokens.csv.gz"
python -m analysis.sparsity_e0 analyse --tokens "$D/e0_tokens.csv.gz" --n-layers "$NLAYERS" \
  --out "$R/e0_${MODEL}.json"
