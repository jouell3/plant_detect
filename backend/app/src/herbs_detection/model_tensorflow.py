import fnmatch
import os
import pickle
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger
from PIL import Image

# ---------------------------------------------------------------------------
# GCS download helper
# ---------------------------------------------------------------------------
_GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "plant-detect-models")
_GCS_TF_PREFIX = os.getenv("GCS_TF_PREFIX", "models_tensorflow").rstrip("/")
_GCS_PROJECT = os.getenv("GCS_PROJECT", "bootcamparomatic")

_MODEL_FILE_PATTERNS = {
    "model": "*_model_*.keras",
    "encoder": "*_label_encoder_*.pkl",
    "metadata": "*_metadata_*.pkl",
}

DEVICE = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


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


def _download_from_gcs_tensorflow(local_dir: Path) -> None:
    from google.cloud import storage

    logger.info(
        "Downloading tensorflow model from gs://{}/{}/",
        _GCS_BUCKET,
        _GCS_TF_PREFIX,
    )
    client = storage.Client(project=_GCS_PROJECT)
    bucket = client.bucket(_GCS_BUCKET)

    prefix = f"{_GCS_TF_PREFIX}/"
    all_blobs = list(bucket.list_blobs(prefix=prefix))
    if not all_blobs:
        raise FileNotFoundError(
            f"No blobs found under gs://{_GCS_BUCKET}/{_GCS_TF_PREFIX}/"
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

    logger.info("GCS tensorflow download complete.")


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------
def _resolve_tensorflow_dir() -> Path:
    from .gcs_cache import is_cache_valid_by_patterns

    logger.info("Resolving tensorflow model directory...")
    if _GCS_BUCKET:
        gcs_dest = Path.cwd() / "models_tensorflow/gcp_download"
        if is_cache_valid_by_patterns(gcs_dest, list(_MODEL_FILE_PATTERNS.values())):
            return gcs_dest
        try:
            _download_from_gcs_tensorflow(gcs_dest)
            return gcs_dest
        except Exception as exc:
            logger.warning(
                "GCS tensorflow download failed ({}), falling back to local files.",
                exc,
            )

    candidates: list[Path] = []

    for env_var in ("MODEL_TF_PATH", "MODEL_PATH"):
        env_path = os.getenv(env_var)
        if env_path:
            p = Path(env_path)
            candidates.append(p.parent if p.is_file() else p)

    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "models_tensorflow")
    candidates.append(Path.cwd() / "backend/app/models_tensorflow")
    candidates.append(Path.cwd() / "app/models_tensorflow")

    for directory in candidates:
        if not directory.exists():
            continue
        try:
            _resolve_latest_artifacts(directory)
            logger.info("Using local tensorflow model files from {}", directory)
            return directory
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Could not find a models_tensorflow directory with matching .keras, "
        "label encoder, and metadata files. Set MODEL_TF_PATH or "
        "place the files in backend/app/models_tensorflow/."
    )


def _load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected dict metadata in {metadata_path}, got {type(metadata)}")
    return metadata


# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first use (or via load_model())
# ---------------------------------------------------------------------------
_le = None
_model = None
_input_size = (384, 384)
_ready = threading.Event()


def load_model() -> None:
    """Resolve model dir, download from GCS if needed, and load the keras model."""
    global _le, _model, _input_size

    model_dir = _resolve_tensorflow_dir()
    artifacts = _resolve_latest_artifacts(model_dir)
    metadata = _load_metadata(artifacts["metadata"])

    with open(artifacts["encoder"], "rb") as f:
        _le = pickle.load(f)

    input_size = metadata.get("input_size", (384, 384))
    if len(input_size) != 2:
        raise ValueError(f"Unsupported input_size in metadata: {input_size}")
    _input_size = tuple(int(dim) for dim in input_size)

    _model = tf.keras.models.load_model(artifacts["model"], compile=False)

    _ready.set()
    logger.info(
        "TensorFlow model ready. device={} model={} classes={} weights={}",
        DEVICE,
        metadata.get("model_name", "unknown"),
        len(_le.classes_),
        artifacts["model"].name,
    )


def _ensure_loaded() -> None:
    _ready.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_array(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((_input_size[1], _input_size[0]))
    array = np.asarray(img, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _load_batch(img_paths: list[str]) -> np.ndarray:
    arrays = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((_input_size[1], _input_size[0]))
        arrays.append(np.asarray(img, dtype=np.float32))
    return np.stack(arrays, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_top3(img_path: str) -> list[tuple[str, float]]:
    _ensure_loaded()
    proba = _model.predict(_load_array(img_path), verbose=0)[0]
    top3 = np.argsort(proba)[::-1][:3]
    return [(_le.classes_[i], round(float(proba[i]), 4)) for i in top3]


def predict_set(img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    _ensure_loaded()
    results = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start : start + batch_size]
        batch = _load_batch(chunk)
        proba = _model.predict(batch, verbose=0)
        for row in proba:
            best = int(np.argmax(row))
            results.append((_le.classes_[best], round(float(row[best]), 4)))
    return results
