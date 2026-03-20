import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from ..src.herbs_detection.model import predict_top3, predict_set, load_model
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in a background thread so the server binds its port immediately
    # and Cloud Run's health check can succeed before the download finishes.
    threading.Thread(target=load_model, daemon=True).start()
    yield


api = FastAPI(lifespan=lifespan)

## to start the server: uvicorn app.api.main:api --reload

@api.get("/")
def root():
    return {"message": "Hello World"}


@api.post("/predict_herb")
async def predict_endpoint(file: UploadFile):
    """Predict species for a single uploaded image. Returns top-3 predictions."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    results = predict_top3(tmp_path)
    Path(tmp_path).unlink()  # clean up temp file
    return {"predictions": [{"species": s, "confidence": c} for s, c in results]}


@api.post("/predict-set")
async def predict_set_endpoint(files: list[UploadFile]):
    """Predict species for a batch of uploaded images."""
    tmp_paths, filenames = [], []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_paths.append(tmp.name)
            filenames.append(file.filename)

    results = predict_set(tmp_paths)

    for p in tmp_paths:  # clean up temp files
        Path(p).unlink()

    return [
        {"filename": f, "species": s, "confidence": c}
        for f, (s, c) in zip(filenames, results)
    ]

if __name__ == "__main__":
    uvicorn.run("app.api.main:api", host="0.0.0.0", port=8080)
