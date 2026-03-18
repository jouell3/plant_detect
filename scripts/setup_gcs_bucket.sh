#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_gcs_bucket.sh
# Creates and configures a GCS bucket to host plant detection model files.
#
# Usage:
#   export GCP_PROJECT=my-project-id
#   export GCS_BUCKET_NAME=plant-detect-models
#   export GCP_REGION=europe-west1        # optional, default: europe-west1
#   export GCP_SERVICE_ACCOUNT=...        # optional, grants bucket access
#   bash scripts/setup_gcs_bucket.sh
# ---------------------------------------------------------------------------
export GCP_PROJECT=bootcamparomatic
export GCS_BUCKET_NAME=plant-detect-models
export GCP_REGION=europe-west1

set -euo pipefail

# ── Required env vars ──────────────────────────────────────────────────────
: "${GCP_PROJECT:?Please set GCP_PROJECT}"
: "${GCS_BUCKET_NAME:?Please set GCS_BUCKET_NAME}"

GCP_REGION="${GCP_REGION:-europe-west1}"
MODELS_PREFIX="models"

echo "==> Setting active project: ${GCP_PROJECT}"
gcloud config set project "${GCP_PROJECT}"

# ── Create bucket ──────────────────────────────────────────────────────────
echo "==> Creating bucket: gs://${GCS_BUCKET_NAME}"
if gsutil ls -b "gs://${GCS_BUCKET_NAME}" &>/dev/null; then
    echo "    Bucket already exists, skipping creation."
else
    gsutil mb \
        -p "${GCP_PROJECT}" \
        -l "${GCP_REGION}" \
        -b on \
        "gs://${GCS_BUCKET_NAME}"
    echo "    Bucket created."
fi

# ── Uniform bucket-level access (best practice) ───────────────────────────
echo "==> Enabling uniform bucket-level access"
gsutil uniformbucketlevelaccess set on "gs://${GCS_BUCKET_NAME}"

# ── Versioning (keeps old model files safe) ───────────────────────────────
echo "==> Enabling object versioning"
gsutil versioning set on "gs://${GCS_BUCKET_NAME}"

# ── Service account access (optional) ────────────────────────────────────
if [[ -n "${GCP_SERVICE_ACCOUNT:-}" ]]; then
    echo "==> Granting objectAdmin to ${GCP_SERVICE_ACCOUNT}"
    gsutil iam ch \
        "serviceAccount:${GCP_SERVICE_ACCOUNT}:roles/storage.objectAdmin" \
        "gs://${GCS_BUCKET_NAME}"
fi

# ── Create placeholder models/ prefix ────────────────────────────────────
echo "==> Initialising gs://${GCS_BUCKET_NAME}/${MODELS_PREFIX}/ prefix"
echo "plant-detect model store" \
    | gsutil cp - "gs://${GCS_BUCKET_NAME}/${MODELS_PREFIX}/.keep"

echo ""
echo "✓ Bucket ready: gs://${GCS_BUCKET_NAME}"
echo ""
echo "Add the following variables to your environment / .env file:"
echo ""
echo "  GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
echo "  GCS_MODELS_PREFIX=${MODELS_PREFIX}"
echo ""
echo "For Cloud Run deployment add them with:"
echo "  gcloud run services update <SERVICE> \\"
echo "    --set-env-vars GCS_BUCKET_NAME=${GCS_BUCKET_NAME},GCS_MODELS_PREFIX=${MODELS_PREFIX}"
