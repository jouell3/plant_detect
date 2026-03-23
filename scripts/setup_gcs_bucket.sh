#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_gcs_bucket.sh
# Creates and configures a GCS bucket to host plant detection model files.
#
# Usage:
#   bash scripts/setup_gcs_bucket.sh
# ---------------------------------------------------------------------------
export GCP_PROJECT=bootcamparomatic
export GCS_BUCKET_NAME=plant-detect-models
export GCP_REGION=europe-west1
export CLOUD_RUN_SERVICE=plant-detect-backend   # name of your Cloud Run service
export GCP_SERVICE_ACCOUNT=plant-detect-sa      # service account name (without @project...)

set -euo pipefail

SA_EMAIL="${GCP_SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com"
MODELS_PREFIX="models"

echo "==> Setting active project: ${GCP_PROJECT}"
gcloud config set project "${GCP_PROJECT}"

# ── Create service account if it doesn't exist ────────────────────────────
echo "==> Checking service account: ${SA_EMAIL}"
if gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    echo "    Service account already exists, skipping creation."
else
    gcloud iam service-accounts create "${GCP_SERVICE_ACCOUNT}" \
        --display-name="Plant Detect API" \
        --project="${GCP_PROJECT}"
    echo "    Service account created."
fi

# ── Create bucket ──────────────────────────────────────────────────────────
echo "==> Creating bucket: gs://${GCS_BUCKET_NAME}"
if gcloud storage buckets describe "gs://${GCS_BUCKET_NAME}" &>/dev/null; then
    echo "    Bucket already exists, skipping creation."
else
    gcloud storage buckets create "gs://${GCS_BUCKET_NAME}" \
        --project="${GCP_PROJECT}" \
        --location="${GCP_REGION}" \
        --uniform-bucket-level-access
    echo "    Bucket created."
fi

# ── Versioning (keeps old model files safe) ───────────────────────────────
echo "==> Enabling object versioning"
gcloud storage buckets update "gs://${GCS_BUCKET_NAME}" --versioning

# ── Grant service account access to the bucket ───────────────────────────
echo "==> Granting objectAdmin to ${SA_EMAIL}"
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET_NAME}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

# ── Create placeholder models/ prefix ────────────────────────────────────
echo "==> Initialising gs://${GCS_BUCKET_NAME}/${MODELS_PREFIX}/ prefix"
echo "plant-detect model store" \
    | gcloud storage cp - "gs://${GCS_BUCKET_NAME}/${MODELS_PREFIX}/.keep"

# ── Attach service account to Cloud Run service (only if it exists) ───────
echo ""
if gcloud run services describe "${CLOUD_RUN_SERVICE}" --region="${GCP_REGION}" &>/dev/null; then
    echo "==> Attaching service account to Cloud Run service: ${CLOUD_RUN_SERVICE}"
    gcloud run services update "${CLOUD_RUN_SERVICE}" \
        --service-account "${SA_EMAIL}" \
        --region "${GCP_REGION}"
else
    echo "    Cloud Run service '${CLOUD_RUN_SERVICE}' not deployed yet — skipping."
    echo "    After your first deploy, run:"
    echo "      gcloud run services update ${CLOUD_RUN_SERVICE} \\"
    echo "        --service-account ${SA_EMAIL} \\"
    echo "        --region ${GCP_REGION}"
fi

echo ""
echo "Done! Bucket ready: gs://${GCS_BUCKET_NAME}"
echo ""
echo "Env vars for Cloud Run:"
echo "  GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
echo "  GCS_MODELS_PREFIX=${MODELS_PREFIX}"
echo "  GCS_PROJECT=${GCP_PROJECT}"
