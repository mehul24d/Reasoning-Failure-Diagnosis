#!/usr/bin/env bash
# Runs on the GCP VM (started by startup.sh): install, smoke test, every phase in order,
# periodic sync of results to the bucket, and a final shutdown so the VM stops billing.
# Resumable: after a reboot the same script continues (finished stages are skipped).
# The HF token is read from Secret Manager into this process's environment only.
set -uo pipefail
ROOT=/opt/rfd
REPO=$ROOT/repo
export RFD_RESULTS_DIR=$ROOT/results RFD_SCRATCH_DIR=$ROOT/scratch HF_HOME=$ROOT/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
BUCKET=${RFD_BUCKET:-gs://icl-failure-rfd-results}
HARNESS=${RFD_HARNESS:-configs/gemma2_2b_l4.yaml}
SECRET=${RFD_SECRET:-hf-token}
WAIT_SECRET_MIN=${RFD_WAIT_SECRET_MIN:-45}
TARGETS=(phase0 baseline-a traces baseline-b pilot pilot-sample report)
mkdir -p "$ROOT"/{logs,results,scratch,hf,state}
LOG=$ROOT/logs/run.log
exec >>"$LOG" 2>&1

status() {  # status <state> [detail]: local file + bucket, for whoever is watching
  printf '{"state": "%s", "detail": "%s", "time": "%s", "host": "%s", "commit": "%s"}\n' \
    "$1" "${2:-}" "$(date -u +%FT%TZ)" "$(hostname)" "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" \
    >"$ROOT/state/status.json"
  gcloud storage cp -q "$ROOT/state/status.json" "$BUCKET/status.json" || true
}
sync_up() {
  gcloud storage rsync -r -q "$RFD_RESULTS_DIR" "$BUCKET/results" || true
  gcloud storage cp -q "$LOG" "$BUCKET/logs/run.log" || true
}
finish() {  # finish <state> <detail>: sync, then stop the VM (a stopped VM only costs its disk)
  status "$1" "${2:-}"; sync_up
  echo "[run] $1 ${2:-}; shutting down"; sleep 5; shutdown -h now
}

echo "=== run_phases.sh start $(date -u +%FT%TZ) harness=$HARNESS"
status starting
# results from a previous VM / disk, if this disk is empty
if [ -z "$(ls -A "$RFD_RESULTS_DIR" 2>/dev/null)" ]; then
  gcloud storage rsync -r -q "$BUCKET/results" "$RFD_RESULTS_DIR" || true
fi
( while true; do sleep 600; sync_up; done ) &   # periodic sync while the phases run

cd "$REPO" && git pull -q --ff-only || echo "[run] git pull failed; using the checked-out commit"
# python with the image's CUDA torch; a venv that sees it, so pip never replaces it
if [ ! -x "$ROOT/venv/bin/python" ]; then
  BASEPY=""
  for p in /opt/conda/bin/python3 /usr/bin/python3; do
    "$p" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && BASEPY=$p && break
  done
  [ -z "$BASEPY" ] && finish failed "no python with CUDA torch on this image"
  "$BASEPY" -m venv --system-site-packages "$ROOT/venv" || finish failed "venv creation"
fi
PY=$ROOT/venv/bin/python
status installing
"$PY" scripts/install_gpu_env.py --report "$ROOT/state/install_report.json" || finish failed "install (see logs/run.log)"
"$PY" -m pip install -q -e . --no-deps || finish failed "pip install -e"
gcloud storage cp -q "$ROOT/state/install_report.json" "$BUCKET/install_report.json" || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

status testing
PATH=$ROOT/venv/bin:$PATH make test-fast PY="$PY" || finish failed "unit tests"

# HF token (needed from here on): wait for the secret to have a version, never print it
waited=0
until HF_TOKEN=$(gcloud secrets versions access latest --secret="$SECRET" 2>/dev/null) && [ -n "$HF_TOKEN" ]; do
  [ $waited -ge $((WAIT_SECRET_MIN * 60)) ] && finish waiting_for_token "no version of secret $SECRET after ${WAIT_SECRET_MIN} min"
  status waiting_for_token "add a version of secret $SECRET"
  sleep 30; waited=$((waited + 30))
done
export HF_TOKEN

status smoke
PATH=$ROOT/venv/bin:$PATH make smoke PY="$PY" HARNESS="$HARNESS" || { sync_up; finish failed "smoke test (see results/smoke/smoke.json)"; }

for t in "${TARGETS[@]}"; do
  status running "$t"
  echo "=== make $t $(date -u +%FT%TZ)"
  PATH=$ROOT/venv/bin:$PATH make "$t" PY="$PY" HARNESS="$HARNESS" || finish failed "phase $t (see logs/run.log)"
  sync_up
done
cp -f docs/BASELINE_REPORT_TABLES.md "$RFD_RESULTS_DIR/" 2>/dev/null || true
finish complete "all phases finished"
