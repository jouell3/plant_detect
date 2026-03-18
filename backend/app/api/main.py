import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from herbs_detection.model import (
    get_all_backend_statuses,
    get_backend_status,
    predict_set,
    predict_top3,
)
from herbs_detection.params import SUPPORTED_BACKENDS, normalize_backend_name
import uvicorn

api = FastAPI()

## to start the server: uvicorn app.api.main:api --reload

@api.get("/")
def root():
    return {"message": "Hello World"}


@api.get("/model-options")
def model_options():
    return {
        "default_backend": "pytorch",
        "supported_backends": list(SUPPORTED_BACKENDS),
        "backends": get_all_backend_statuses(),
    }


@api.post("/predict_herb")
async def predict_endpoint(file: UploadFile, backend: str = Form("pytorch")):
    """Predict species for a single uploaded image. Returns top-3 predictions."""
    try:
        normalized_backend = normalize_backend_name(backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    backend_status = get_backend_status(normalized_backend)
    if not backend_status["available"]:
        raise HTTPException(status_code=503, detail=backend_status["error"])

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        results = predict_top3(tmp_path, backend=normalized_backend)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "backend": normalized_backend,
        "predictions": [{"species": species, "confidence": confidence} for species, confidence in results],
    }


@api.post("/predict-set")
async def predict_set_endpoint(files: list[UploadFile], backend: str = Form("pytorch")):
    """Predict species for a batch of uploaded images."""
    try:
        normalized_backend = normalize_backend_name(backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    backend_status = get_backend_status(normalized_backend)
    if not backend_status["available"]:
        raise HTTPException(status_code=503, detail=backend_status["error"])

    tmp_paths, filenames = [], []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_paths.append(tmp_file.name)
            filenames.append(file.filename)

    try:
        results = predict_set(tmp_paths, backend=normalized_backend)
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)

    return {
        "backend": normalized_backend,
        "predictions": [
            {"filename": filename, "species": species, "confidence": confidence}
            for filename, (species, confidence) in zip(filenames, results)
        ],
    }

if __name__ == "__main__":
    uvicorn.run("backend.app.api.main:api", host="0.0.0.0", port=8080)
