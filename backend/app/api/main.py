import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from ..src.herbs_detection.model import predict_top3, predict_set
import uvicorn

api = FastAPI()

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
