import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from herbs_detection.model import predict_top3 as pt_top3, predict_set as pt_set
from herbs_detection.model_sklearn import predict_top3 as sk_top3, predict_set as sk_set
import uvicorn

api = FastAPI()

## to start the server: uvicorn app.api.main:api --reload


@api.get("/")
def root():
    return {"message": "Hello World"}


@api.post("/predict_herb")
async def predict_endpoint(file: UploadFile):
    """Predict species for a single uploaded image.
    Returns top-3 predictions from both the PyTorch (ResNet18) and
    sklearn (EfficientNet-B3 + LogisticRegression) models for comparison.
    """
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    pytorch_preds = pt_top3(tmp_path)
    sklearn_preds = sk_top3(tmp_path)
    Path(tmp_path).unlink()

    return {
        "pytorch": [{"species": s, "confidence": c} for s, c in pytorch_preds],
        "sklearn": [{"species": s, "confidence": c} for s, c in sklearn_preds],
    }


@api.post("/predict-set")
async def predict_set_endpoint(files: list[UploadFile]):
    """Predict species for a batch of uploaded images.
    Returns predictions from both models for each image.
    """
    tmp_paths, filenames = [], []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_paths.append(tmp.name)
            filenames.append(file.filename)

    pytorch_results = pt_set(tmp_paths)
    sklearn_results = sk_set(tmp_paths)

    for p in tmp_paths:
        Path(p).unlink()

    return [
        {
            "filename": f,
            "pytorch": {"species": pt_s, "confidence": pt_c},
            "sklearn": {"species": sk_s, "confidence": sk_c},
        }
        for f, (pt_s, pt_c), (sk_s, sk_c)
        in zip(filenames, pytorch_results, sklearn_results)
    ]


if __name__ == "__main__":
    uvicorn.run("app.api.main:api", host="0.0.0.0", port=8080)
