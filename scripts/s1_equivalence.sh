#!/usr/bin/env bash
# S1 (M0): the six TL-HF equivalence checks of plan Table 5 for one model.
# Usage: bash scripts/s1_equivalence.sh gemma-2-2b-it [device] [tl_n_devices]
#   Kaggle 2 x T4:  bash scripts/s1_equivalence.sh gemma-2-2b-it cuda 2
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?model key from configs/models.yaml}"
DEVICE="${2:-cuda}"
TL_N="${3:-1}"
mkdir -p reports/m0
pip freeze > "reports/m0/versions_$(hostname).txt"
nvidia-smi > "reports/m0/gpu_$(hostname).txt" 2>/dev/null || true
python -m harness.equivalence --model "$MODEL" --device "$DEVICE" --tl-n-devices "$TL_N" \
  --out "reports/m0/equivalence_${MODEL}.json"
