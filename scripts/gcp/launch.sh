#!/usr/bin/env bash
# Create (or restart) the GPU VM for the experiments. Idempotent.
#   scripts/gcp/launch.sh            create the VM if missing, otherwise start it
# Settings via environment: PROJECT ZONE MACHINE VM DISK_GB HARNESS BRANCH
set -euo pipefail
PROJECT=${PROJECT:-icl-failure}
ZONE=${ZONE:-us-central1-a}
MACHINE=${MACHINE:-g2-standard-8}        # 8 vCPU, 32 GB RAM, 1x NVIDIA L4 (24 GB)
VM=${VM:-rfd-gemma}
DISK_GB=${DISK_GB:-200}
HARNESS=${HARNESS:-configs/gemma2_2b_l4.yaml}
BRANCH=${BRANCH:-main}
REPO_URL=${REPO_URL:-https://github.com/mehul24d/Reasoning-Failure-Diagnosis.git}
SA=rfd-runner@$PROJECT.iam.gserviceaccount.com
HERE=$(cd "$(dirname "$0")" && pwd)

if gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  echo "VM $VM exists: starting it (no-op if running)"
  gcloud compute instances start "$VM" --project="$PROJECT" --zone="$ZONE"
  exit 0
fi
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
  --service-account="$SA" --scopes=cloud-platform \
  --metadata="rfd-repo=$REPO_URL,rfd-branch=$BRANCH,rfd-harness=$HARNESS,install-nvidia-driver=True" \
  --metadata-from-file=startup-script="$HERE/startup.sh" \
  --labels=purpose=rfd-experiments
echo "created $VM. Watch: gcloud storage cat gs://icl-failure-rfd-results/status.json"
