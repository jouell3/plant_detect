import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from loguru import logger
from ..src.herbs_detection.model import predict_top3, predict_set, load_model
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model in background thread...")
    threading.Thread(target=load_model, daemon=True).start()
    yield
    logger.info("Shutting down.")


api = FastAPI(lifespan=lifespan)

## to start the server: uvicorn app.api.main:api --reload

@api.get("/")
def root():
    return {"message": "Hello World"}


@api.post("/predict_herb")
async def predict_endpoint(file: UploadFile):
    """Predict species for a single uploaded image. Returns top-3 predictions."""
    logger.info("predict_herb | file={}", file.filename)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    results = predict_top3(tmp_path)
    Path(tmp_path).unlink()
    logger.debug("predict_herb | results={}", results)
    return {"predictions": [{"species": s, "confidence": c} for s, c in results]}


@api.post("/predict-set")
async def predict_set_endpoint(files: list[UploadFile]):
    """Predict species for a batch of uploaded images."""
    logger.info("predict_set | {} files", len(files))
    tmp_paths, filenames = [], []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_paths.append(tmp.name)
            filenames.append(file.filename)

    results = predict_set(tmp_paths)

    for p in tmp_paths:
        Path(p).unlink()

    logger.debug("predict_set | results={}", results)
    return [
        {"filename": f, "species": s, "confidence": c}
        for f, (s, c) in zip(filenames, results)
    ]

if __name__ == "__main__":
    uvicorn.run("app.api.main:api", host="0.0.0.0", port=8080)
