import fnmatch
import os
import pickle
import threading
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torchvision import models, transforms

_MODEL_FILES  = ["EFFICIENTNET_B3_*.pt", "EFFICIENTNET_B3_label_encoder_*.pkl", "EFFICIENTNET_B3_metadata_*.pkl"]
_MODEL_FILE_PATTERNS = {
    "weights": "*.pt",
    "encoder": "*_label_encoder_*.pkl",
    "metadata": "*_metadata_*.pkl",
}

_MODEL_SPECS = {
    "efficientnet_b3": {"builder": models.efficientnet_b3, "img_size": 300},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------
def _pick_latest_path(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return matches[-1]


def _resolve_latest_artifacts(directory: Path) -> dict[str, Path]:
    return {
        name: _pick_latest_path(directory, pattern)
        for name, pattern in _MODEL_FILE_PATTERNS.items()
    }


def _resolve_pytorch_large_dir() -> Path:
    logger.info("Resolving large pytorch model directory...")

    candidates: list[Path] = []

    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "models_pytorch_large")
    candidates.append(Path.cwd() / "backend/app/models_pytorch_large")
    candidates.append(Path.cwd() / "app/models_pytorch_large")

    logger.debug("Large pytorch model directory candidates: {}", candidates)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find a models_pytorch_large directory with matching .pt, "
        "label encoder, and metadata files. Set MODEL_PYTORCH_LARGE_PATH or "
        "place the files in backend/app/models_pytorch_large/."
    )


def _load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected dict metadata in {metadata_path}, got {type(metadata)}")
    return metadata


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format for pytorch large model.")

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def _build_model(model_name: str, num_classes: int, dropout_rate: float) -> tuple[torch.nn.Module, int]:
    spec = _MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"Unsupported backbone in metadata: {model_name}")

    model = spec["builder"](weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_rate, inplace=True),
        torch.nn.Linear(in_features, num_classes),
    )
    return model, spec["img_size"]


# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first use (or via load_model())
# ---------------------------------------------------------------------------
_le = None
_model = None
_preprocess = None
_ready = threading.Event()


def load_model() -> None:
    """Resolve model dir, download from GCS if needed, and load weights."""
    global _le, _model, _preprocess

    model_dir = _resolve_pytorch_large_dir()
    artifacts = _resolve_latest_artifacts(model_dir)
    metadata = _load_metadata(artifacts["metadata"])

    with open(artifacts["encoder"], "rb") as f:
        _le = pickle.load(f)

    model_name = str(metadata.get("model_name", "efficientnet_b3")).lower()
    num_classes = int(metadata.get("num_classes", len(_le.classes_)))
    dropout_rate = float(metadata.get("dropout_rate", 0.4))
    _model, img_size = _build_model(model_name, num_classes, dropout_rate)

    checkpoint = torch.load(artifacts["weights"], map_location=DEVICE)
    _model.load_state_dict(_extract_state_dict(checkpoint))
    _model.to(DEVICE)
    _model.eval()

    _preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _ready.set()
    logger.info(
        "Large pytorch model ready. device={} backbone={} classes={} weights={}",
        DEVICE,
        model_name,
        num_classes,
        artifacts["weights"].name,
    )


def _ensure_loaded() -> None:
    _ready.wait()


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
