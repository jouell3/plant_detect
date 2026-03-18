
.PHONY: test train_tf train_pt api frontend backend_status

test:
	curl -X POST http://localhost:8000/predict_herb \
	  -F "file=@data/all_images/dill_0.jpg" \
	  -F "backend=tensorflow"

train_tf:
	PYTHONPATH=backend/app/src python -m herbs_detection.tensorflow_pipeline

train_pt:
	PYTHONPATH=backend/app/src python -m herbs_detection.pytorch_pipeline

api:
	PYTHONPATH=backend/app/src:. uvicorn backend.app.api.main:api --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/main.py --server.headless true

backend_status:
	PYTHONPATH=backend/app/src:. python -c "from herbs_detection.model import get_all_backend_statuses; print(get_all_backend_statuses())"

build: ## Build Docker image locally
	docker build -f docker/backend.Dockerfile -t plant-detect-backend .

run: build ## Build and run container locally
	docker run -p ${PORT}:${PORT} -e PORT=${PORT} ${IMAGE} 

build_gcp: ## Build image for GCP (Linux/amd64 platform)
	@echo "Building the image for GCP..."
	docker buildx build --platform linux/amd64 -f docker/backend.Dockerfile -t ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE} . --push

push_gcp: build_gcp ## Build and push image to Artifact Registry
	docker push ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE}
