FROM python:3.11-slim

WORKDIR /plant_detect

RUN pip install --upgrade pip setuptools wheel
COPY backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY backend/ backend/

ENV MODEL_PATH=/plant_detect/backend/app/models
ENV PYTHONPATH=/plant_detect/backend/app/src

# Set at deploy time (e.g. gcloud run deploy --set-env-vars)
ENV GCS_BUCKET_NAME="plant-detect-models"
ENV GCS_MODELS_PREFIX="models"

CMD ["uvicorn", "backend.app.api.main:api", "--host", "0.0.0.0", "--port", "8080"]