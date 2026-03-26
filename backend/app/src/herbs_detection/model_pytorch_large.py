import fnmatch
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
_GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "")
_GCS_PYTORCH_LARGE_PREFIX = os.getenv("GCS_PYTORCH_LARGE_PREFIX", "models_pytorch_large").rstrip("/")
_GCS_PROJECT = os.getenv("GCS_PROJECT", "bootcamparomatic")

_MODEL_FILE_PATTERNS = {
    "weights": "*.pt",
    "encoder": "*_label_encoder_*.pkl",
    "metadata": "*_metadata_*.pkl",
}

_MODEL_SPECS = {
    "efficientnet_b3": {"builder": models.efficientnet_b3, "img_size": 300},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _pick_latest_blob(blobs: list, pattern: str):
    matches = sorted(
        (blob for blob in blobs if fnmatch.fnmatch(Path(blob.name).name, pattern)),
        key=lambda blob: Path(blob.name).name,
    )
    if not matches:
        raise FileNotFoundError(f"No blob matching '{pattern}'")
    return matches[-1]


def _download_from_gcs_pytorch_large(local_dir: Path) -> None:
    from google.cloud import storage

    logger.info(
        "Downloading large pytorch model from gs://{}/{}/",
        _GCS_BUCKET,
        _GCS_PYTORCH_LARGE_PREFIX,
    )
    client = storage.Client(project=_GCS_PROJECT)
    bucket = client.bucket(_GCS_BUCKET)

    prefix = f"{_GCS_PYTORCH_LARGE_PREFIX}/"
    all_blobs = list(bucket.list_blobs(prefix=prefix))
    if not all_blobs:
        raise FileNotFoundError(
            f"No blobs found under gs://{_GCS_BUCKET}/{_GCS_PYTORCH_LARGE_PREFIX}/"
        )

    selected = {
        name: _pick_latest_blob(all_blobs, pattern)
        for name, pattern in _MODEL_FILE_PATTERNS.items()
    }

    local_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in selected.items():
        dest = local_dir / Path(blob.name).name
        logger.debug("  {} [{}] -> {}", blob.name, name, dest)
        blob.download_to_filename(str(dest))

    logger.info("GCS pytorch large download complete.")


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------
def _resolve_pytorch_large_dir() -> Path:
    from .gcs_cache import is_cache_valid_by_patterns

    logger.info("Resolving large pytorch model directory...")
    if _GCS_BUCKET:
        gcs_dest = Path.cwd() / "models_pytorch_large/gcp_download"
        if is_cache_valid_by_patterns(gcs_dest, list(_MODEL_FILE_PATTERNS.values())):
            return gcs_dest
        try:
            _download_from_gcs_pytorch_large(gcs_dest)
            return gcs_dest
        except Exception as exc:
            logger.warning(
                "GCS large pytorch download failed ({}), falling back to local files.",
                exc,
            )

    candidates: list[Path] = []

    for env_var in ("MODEL_PYTORCH_LARGE_PATH", "MODEL_PATH"):
        env_path = os.getenv(env_var)
        if env_path:
            p = Path(env_path)
            candidates.append(p.parent if p.is_file() else p)

    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "models_pytorch_large")
    candidates.append(Path.cwd() / "backend/app/models_pytorch_large")
    candidates.append(Path.cwd() / "app/models_pytorch_large")

    logger.debug("Large pytorch model directory candidates: {}", candidates)
    for directory in candidates:
        if not directory.exists():
            continue
        try:
            _resolve_latest_artifacts(directory)
            logger.info("Using local large pytorch model files from {}", directory)
            return directory
        except FileNotFoundError:
            logger.error("Directory {} does not contain required model files, skipping.", directory)
            continue

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
