#!/usr/bin/env bash
# End-to-end pilot pipeline: Qwen2.5-0.5B-Instruct on ~50 GSM8K problems.
# Each stage is a separate process so models are freed between stages
# (keeps peak RAM low on constrained machines).
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPERIMENT="${1:-configs/legacy/experiment.yaml}"

echo "== 1/5: generating pilot traces =="
python3 -m rfd.generation.generate_traces --experiment "$EXPERIMENT"

echo "== 2/5: training pilot SAE =="
python3 -m rfd.legacy.sae.train_sae --experiment "$EXPERIMENT"

echo "== 3/5: encoding SAE features for pilot traces =="
python3 -m rfd.legacy.sae.build_sae_features --experiment "$EXPERIMENT"

echo "== 4/5: running phase segmentation =="
python3 -m rfd.legacy.segmentation.run_segmentation --experiment "$EXPERIMENT"

echo "== 5/5: building failure taxonomy =="
python3 -m rfd.legacy.taxonomy.run_taxonomy --experiment "$EXPERIMENT"

echo "Pilot pipeline complete. See results/legacy_pilot/ for outputs."
