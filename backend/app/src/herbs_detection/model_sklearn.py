import json
import os
import pickle
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# GCS download helper
# ---------------------------------------------------------------------------
_GCS_BUCKET        = os.getenv("GCS_BUCKET_NAME", "")
_GCS_SKLEARN_PREFIX = os.getenv("GCS_SKLEARN_PREFIX", "models_sklearn").rstrip("/")
_GCS_PROJECT       = os.getenv("GCS_PROJECT", "bootcamparomatic")

_SKLEARN_BLOB_PATTERNS = (
    "config_sklearn__",
    "label_encoder_sklearn__",
    "efficientnet_b3__logistic_regression__",
)


def _download_from_gcs_sklearn(local_dir: Path) -> None:
    from google.cloud import storage

    logger.info("Downloading sklearn models from gs://{}/{}/", _GCS_BUCKET, _GCS_SKLEARN_PREFIX)
    client = storage.Client(project=_GCS_PROJECT)
    bucket = client.bucket(_GCS_BUCKET)

    local_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{_GCS_SKLEARN_PREFIX}/"
    all_blobs = list(bucket.list_blobs(prefix=prefix))
    matched = [
        b for b in all_blobs
        if any(Path(b.name).name.startswith(p) for p in _SKLEARN_BLOB_PATTERNS)
    ]

    if not matched:
        raise FileNotFoundError(
            f"No blobs matching patterns {_SKLEARN_BLOB_PATTERNS} found under gs://{_GCS_BUCKET}/{_GCS_SKLEARN_PREFIX}/"
        )

    for blob in matched:
        filename = Path(blob.name).name
        dest = local_dir / filename
        logger.debug("  {} → {}", blob.name, dest)
        blob.download_to_filename(str(dest))

    logger.info("GCS sklearn download complete ({} files).", len(matched))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _resolve_sklearn_dir() -> Path:
    from .gcs_cache import is_cache_valid_by_patterns

    # ── 1. Try GCS first ─────────────────────────────────────────────────
    logger.info("Resolving sklearn model directory...")
    if _GCS_BUCKET:
        gcs_dest = Path.cwd() / "models_sklearn/gcp_download"
        _cache_patterns = [f"{p}*" for p in _SKLEARN_BLOB_PATTERNS]
        if is_cache_valid_by_patterns(gcs_dest, _cache_patterns):
            return gcs_dest
        try:
            _download_from_gcs_sklearn(gcs_dest)
            return gcs_dest
        except Exception as exc:
            logger.warning("GCS sklearn download failed ({}), falling back to local files.", exc)

    # ── 2. Fallback: use pre-existing local files ─────────────────────────
    candidates = []

    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        candidates.append(p.parent if p.suffix == ".json" else p)

    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "models_sklearn")
    candidates.append(Path.cwd() / "backend/app/models_sklearn")
    candidates.append(Path.cwd() / "app/models_sklearn")

    for p in candidates:
        if p.exists():
            return p

    print("Searched for sklearn model files in the following locations:")
    for c in candidates:
        print(f"  - {c}")

    raise FileNotFoundError(
        "Could not find models_sklearn directory. "
        "Set MODEL_SKLEARN_PATH to the folder containing the sklearn model files."
    )


def _load_config(models_dir: Path) -> dict:
    env_path = os.getenv("MODEL_SKLEARN_PATH")
    if env_path:
        p = Path(env_path)
        if p.suffix == ".json" and p.is_file():
            with open(p) as f:
                return json.load(f)

    # Check for plain config_sklearn.json first (downloaded from GCS)
    plain = models_dir / "config_sklearn.json"
    if plain.is_file():
        with open(plain) as f:
            return json.load(f)

    # Use the most recently trained config (sorted by timestamp in filename)
    configs = sorted(models_dir.glob("config_sklearn__*.json"))
    if not configs:
        raise FileNotFoundError(f"No config_sklearn.json or config_sklearn__*.json found in {models_dir}")

    with open(configs[-1]) as f:
        return json.load(f)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first use (or via load_model())
# ---------------------------------------------------------------------------
_pipeline  = None
_le        = None
_backbone  = None
_preprocess = None
_ready     = threading.Event()


def load_model() -> None:
    """Resolve model dir, download from GCS if needed, and load weights.

    Called explicitly from the FastAPI startup event so the server can bind
    its port before the (potentially slow) GCS download begins.
    """
    global _pipeline, _le, _backbone, _preprocess

    sklearn_dir = _resolve_sklearn_dir()
    config      = _load_config(sklearn_dir)

    img_size = config["img_size"]
    backbone_name = config["backbone"]

    with open(sklearn_dir / config["pipeline_file"], "rb") as f:
        _pipeline = pickle.load(f)

    with open(sklearn_dir / config["encoder_file"], "rb") as f:
        _le = pickle.load(f)

    # Frozen EfficientNet backbone — weights are never updated at inference
    backbone_map = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b3": models.efficientnet_b3,
    }
    if backbone_name not in backbone_map:
        raise ValueError(f"Unsupported backbone in config: {backbone_name}")

    _backbone = backbone_map[backbone_name](weights="IMAGENET1K_V1")
    _backbone.classifier = nn.Identity()
    _backbone = _backbone.to(DEVICE).eval()
    for param in _backbone.parameters():
        param.requires_grad = False

    _preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _ready.set()
    logger.info("Sklearn model ready. device={} backbone={}", DEVICE, backbone_name)


def _ensure_loaded() -> None:
    _ready.wait()  # blocks until load_model() completes (no-op if already ready)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _extract_features(img_paths: list[str]) -> np.ndarray:
    tensors = [_preprocess(Image.open(p).convert("RGB")) for p in img_paths]
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        feats = _backbone(batch)
    return feats.cpu().numpy()   # (N, feat_dim)


# ---------------------------------------------------------------------------
# Public API  — same signatures as model.py
# ---------------------------------------------------------------------------
def predict_top3(img_path: str) -> list[tuple[str, float]]:
    """Return the top-3 predicted species with confidence scores."""
    _ensure_loaded()
    feats = _extract_features([img_path])          # (1, feat_dim)
    proba = _pipeline.predict_proba(feats)[0]      # (num_classes,)
    top3  = np.argsort(proba)[::-1][:3]
    return [(_le.classes_[i], round(float(proba[i]), 4)) for i in top3]


def predict_set(img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    """Run batch inference. Returns one (species, confidence) tuple per image."""
    _ensure_loaded()
    results = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start : start + batch_size]
        feats = _extract_features(chunk)            # (N, feat_dim)
        proba = _pipeline.predict_proba(feats)      # (N, num_classes)
        for p in proba:
            best = int(np.argmax(p))
            results.append((_le.classes_[best], round(float(p[best]), 4)))
    return results
