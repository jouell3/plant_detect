import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from ..src.herbs_detection.model import predict_top3 as pt_top3, predict_set as pt_set, load_model as load_model_pytorch
from ..src.herbs_detection.model_sklearn import predict_top3 as sk_top3, predict_set as sk_set, load_model as load_model_sklearn
from ..src.herbs_detection.model_pytorch_large import predict_top3 as ptl_top3, predict_set as ptl_set, load_model as load_model_pytorch_large
from ..src.herbs_detection.model_tensorflow import predict_top3 as tf_top3, predict_set as tf_set, load_model as load_model_tensorflow
from ..src.herbs_detection.model_illness import predict_top3 as illness_top3, predict_set as illness_set, load_model as load_model_illness


from loguru import logger
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model in background thread...")
    threading.Thread(target=load_model_pytorch, daemon=True).start()
    threading.Thread(target=load_model_sklearn, daemon=True).start()
    threading.Thread(target=load_model_pytorch_large, daemon=True).start()
    threading.Thread(target=load_model_tensorflow, daemon=True).start()
    threading.Thread(target=load_model_illness, daemon=True).start()
    yield
    logger.info("Shutting down.")

api = FastAPI(lifespan=lifespan)

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
    logger.info("predict_herb | file={}", file.filename)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    pytorch_preds = pt_top3(tmp_path)
    sklearn_preds = sk_top3(tmp_path)
    pytorch_large_preds = ptl_top3(tmp_path)
    tensorflow_preds = tf_top3(tmp_path)
    Path(tmp_path).unlink()

    return {
        "pytorch": [{"species": s, "confidence": c} for s, c in pytorch_preds],
        "sklearn": [{"species": s, "confidence": c} for s, c in sklearn_preds],
        "pytorch_large": [{"species": s, "confidence": c} for s, c in pytorch_large_preds],
        "tensorflow": [{"species": s, "confidence": c} for s, c in tensorflow_preds],
    }

@api.post("/predict_illness")
async def predict_illness_endpoint(file: UploadFile):
    """Predict illness for a single uploaded image.
    Returns top-3 predictions from the PyTorch illness model.
    """
    logger.info("predict_illness | file={}", file.filename)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    pytorch_preds = illness_top3(tmp_path)
    Path(tmp_path).unlink()

    return {
        "pytorch": [{"illness": s, "confidence": c} for s, c in pytorch_preds]
    }


@api.post("/predict-set")
async def predict_set_endpoint(files: list[UploadFile]):
    """Predict species for a batch of uploaded images.
    Returns predictions from both models for each image.
    """
    logger.info("predict_set | {} files", len(files))
    tmp_paths, filenames = [], []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_paths.append(tmp.name)
            filenames.append(file.filename)

    pytorch_results = pt_set(tmp_paths)
    sklearn_results = sk_set(tmp_paths)
    pytorch_large_results = ptl_set(tmp_paths)
    tensorflow_results = tf_set(tmp_paths)
    for p in tmp_paths:
        Path(p).unlink()
    return [
        {
            "filename": f,
            "pytorch": {"species": pt_s, "confidence": pt_c},
            "sklearn": {"species": sk_s, "confidence": sk_c},
            "pytorch_large": {"species": ptl_s, "confidence": ptl_c},
            "tensorflow": {"species": tf_s, "confidence": tf_c},
        }
        for f, (pt_s, pt_c), (sk_s, sk_c), (ptl_s, ptl_c), (tf_s, tf_c)
        in zip(filenames, pytorch_results, sklearn_results, pytorch_large_results, tensorflow_results)
    ]


@api.post("/predict-set_illness")
async def predict_set_illness_endpoint(files: list[UploadFile]):
    """Predict illness for a batch of uploaded images.
    Returns top-1 prediction from the PyTorch illness model for each image.
    """
    logger.info("predict_set_illness | {} files", len(files))
    tmp_paths, filenames = [], []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_paths.append(tmp.name)
            filenames.append(file.filename)

    illness_results = illness_set(tmp_paths)
    for p in tmp_paths:
        Path(p).unlink()

    return [
        {"filename": f, "pytorch": {"illness": s, "confidence": c}}
        for f, (s, c) in zip(filenames, illness_results)
    ]


if __name__ == "__main__":
    uvicorn.run("app.api.main:api", host="0.0.0.0", port=8080)
