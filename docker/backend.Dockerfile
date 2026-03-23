FROM python:3.11-slim

WORKDIR /plant_detect

RUN pip install --upgrade pip setuptools wheel
COPY backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY backend/ backend/

ENV PYTHONPATH=/plant_detect/backend/app/src

# ── GCS / model config ────────────────────────────────────────────────────────
# Override at deploy time:  gcloud run deploy --set-env-vars KEY=VALUE
ENV GCS_BUCKET_NAME="plant-detect-models"
ENV GCS_MODELS_PREFIX="models"
ENV GCS_PROJECT="bootcamparomatic"

# MODEL_PATH: where downloaded model files are cached inside the container
ENV MODEL_PATH=/tmp/plant_models


CMD uvicorn backend.app.api.main:api --host 0.0.0.0 --port ${PORT:-8080}