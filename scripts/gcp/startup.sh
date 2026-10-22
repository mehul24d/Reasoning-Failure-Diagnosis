#!/usr/bin/env bash
# GCE startup script (runs as root on every boot): fetch the repo and start the runner
# detached. Rebooting the VM (e.g. after a GPU maintenance event) therefore resumes the run.
set -u
ROOT=/opt/rfd
mkdir -p $ROOT/logs
exec >>$ROOT/logs/startup.log 2>&1
echo "=== startup $(date -u +%FT%TZ)"
REPO_URL=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/rfd-repo || true)
BRANCH=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/rfd-branch || echo main)
HARNESS=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/rfd-harness || true)
if [ ! -d $ROOT/repo/.git ]; then
  git clone -q --branch "${BRANCH:-main}" "$REPO_URL" $ROOT/repo || { echo "clone failed"; exit 1; }
fi
systemctl reset-failed rfd.service 2>/dev/null || true
# a transient unit: runs detached from this script, survives it exiting
systemd-run --unit=rfd --description="RFD experiment run" \
  --setenv=RFD_HARNESS="${HARNESS:-configs/gemma2_2b_l4.yaml}" \
  /bin/bash $ROOT/repo/scripts/gcp/run_phases.sh
