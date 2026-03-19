
.PHONY: test

serve: ## Run the API locally with the correct PYTHONPATH
	PYTHONPATH=backend/app/src uvicorn backend.app.api.main:api --reload --host 0.0.0.0 --port 8080

test:
	curl -X POST http://localhost:8080/predict_herb \
	  -F "file=@data/raw/all_images/dill_0.jpg"

build: ## Build Docker image locally
	docker build -f docker/backend.Dockerfile -t plant-detect-backend .

run: build ## Build and run container locally
	docker run -p ${PORT}:${PORT} -e PORT=${PORT} ${IMAGE} 

build_gcp: ## Build image for GCP (Linux/amd64 platform)
	@echo "Building the image for GCP..."
	docker buildx build --platform linux/amd64 -f docker/backend.Dockerfile -t ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE} . --push

push_gcp: build_gcp ## Build and push image to Artifact Registry
	docker push ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE}