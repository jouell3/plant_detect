import os
import pickle
import threading
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# GCS download helper
# ---------------------------------------------------------------------------

_MODEL_FILES  = ["resnet18_plants.pt", "label_encoder.pkl"]


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------
def _resolve_model_dir() -> Path:
    """Return a local directory that contains both model files.

    Resolution order:
    1. Try to download fresh files from GCS into MODEL_PATH (or /tmp/plant_models).
    2. If GCS download fails (no bucket configured, network error, etc.),
       fall back to the first local directory that already has both files:
         - MODEL_PATH env var
         - backend/app/models/ relative to the source tree
    """
    logger.info("Resolving model directory...")

    fallback_candidates: list[Path] = []

    here = Path(__file__).resolve()
    fallback_candidates.append(here.parents[2] / "models")      # backend/app/models
    fallback_candidates.append(Path.cwd() / "backend/app/models")
    fallback_candidates.append(Path.cwd() / "app/models")
    
    for p in fallback_candidates:
        if p.is_dir() and all((p / f).exists() for f in _MODEL_FILES):
            logger.info("Using local model files from {}", p)
            return p

    raise FileNotFoundError(
        "GCS download failed and no local model files were found. "
        "Set GCS_BUCKET_NAME or place resnet18_plants.pt + label_encoder.pkl"
        "in backend/app/models/."
    )

IMG_SIZE = 400
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first use (or via load_model())
# ---------------------------------------------------------------------------
_le    = None
_model = None
_ready = threading.Event()  # set once model is fully loaded


def load_model() -> None:
    """Resolve model dir, download from GCS if needed, and load weights.

    Called explicitly from the FastAPI startup event so the server can bind
    its port before the (potentially slow) GCS download begins.
    """
    global _le, _model

    model_dir     = _resolve_model_dir()
    weights_path  = model_dir / "resnet18_plants.pt"
    encoder_path  = model_dir / "label_encoder.pkl"

    with open(encoder_path, "rb") as f:
        _le = pickle.load(f)

    num_classes = len(_le.classes_)
    _model = models.resnet18(weights=None)
    _model.fc = torch.nn.Linear(_model.fc.in_features, num_classes)
    _model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()
    _ready.set()
    logger.info("Model ready. device={} classes={}", DEVICE, num_classes)


def _ensure_loaded() -> None:
    _ready.wait()  # blocks until load_model() completes (no-op if already ready)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_tensor(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return _preprocess(img).unsqueeze(0).to(DEVICE)


def _load_batch(img_paths: list[str]) -> torch.Tensor:
    tensors = [_preprocess(Image.open(p).convert("RGB")) for p in img_paths]
    return torch.stack(tensors).to(DEVICE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict(img_path: str) -> tuple[str, float]:
    _ensure_loaded()
    with torch.no_grad():
        proba = torch.softmax(_model(_load_tensor(img_path)), dim=1).squeeze()
    confidence, class_idx = proba.max(dim=0)
    species = _le.inverse_transform([class_idx.item()])[0]
    return species, round(confidence.item(), 4)


def predict_top3(img_path: str) -> list[tuple[str, float]]:
    _ensure_loaded()
    with torch.no_grad():
        proba = torch.softmax(_model(_load_tensor(img_path)), dim=1).squeeze()
    top3 = proba.topk(3)
    return [
        (_le.classes_[i.item()], round(p.item(), 4))
        for i, p in zip(top3.indices, top3.values)
    ]


def predict_set(img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    _ensure_loaded()
    results = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start : start + batch_size]
        batch = _load_batch(chunk)
        with torch.no_grad():
            proba = torch.softmax(_model(batch), dim=1)
        confidences, class_idxs = proba.max(dim=1)
        species = _le.inverse_transform(class_idxs.cpu().tolist())
        results.extend(
            (s, round(c, 4))
            for s, c in zip(species, confidences.cpu().tolist())
        )
    return results
