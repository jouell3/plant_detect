
# ── GCP config ────────────────────────────────────────────────────────────────
GCP_PROJECT    ?= bootcamparomatic
LOCATION       ?= europe-west1
GCP_REPOSITORY ?= plant-detect
IMAGE          ?= plant-detect-backend
PORT           ?= 8080

.PHONY: test serve build run build_gcp push_gcp

serve: ## Run the API locally with the correct PYTHONPATH
	PYTHONPATH=backend/app/src uvicorn backend.app.api.main:api --reload --host 0.0.0.0 --port 8080

test:
	curl -X POST http://localhost:8080/predict_herb \
	  -F "file=@data/raw/all_images/dill_0.jpg"

build: ## Build Docker image locally
	docker build -f docker/backend.Dockerfile -t plant-detect-backend .

run: build ## Build and run container locally (mounts GCP ADC credentials)
	docker run --rm -p 8080:8080 \
	  -v ~/.config/gcloud/application_default_credentials.json:/tmp/adc.json:ro \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/adc.json \
	  -e GCS_BUCKET_NAME=plant-detect-models \
	  -e GCS_PROJECT=bootcamparomatic \
	  plant-detect-backend

build_gcp: ## Build image for GCP (Linux/amd64 platform)
	@echo "Building the image for GCP..."
	docker buildx build --platform linux/amd64 -f docker/backend.Dockerfile -t ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE} . --push

push_gcp: build_gcp ## Build and push image to Artifact Registry
	docker push ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE}

get_log_gcp: 
	gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=plant-detect-backend" \
  		--project=bootcamparomatic \
  		--limit=50 \
  		--format="table(timestamp, severity, textPayload)"