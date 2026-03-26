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
_GCS_BUCKET   = os.getenv("GCS_BUCKET_NAME", "plant-detect-models")
_GCS_PREFIX   = os.getenv("GCS_MODELS_ILLNESS_PREFIX", "models_illness").rstrip("/")
_GCS_PROJECT  = os.getenv("GCS_PROJECT", "bootcamparomatic")
_MODEL_FILES  = ["resnet18_plants_illness.pt", "label_encoder_illness.pkl"]


def _download_from_gcs(local_dir: Path) -> None:
    """Download model files from GCS into local_dir."""
    from google.cloud import storage  # lazy import — only needed when files are missing

    logger.info("Downloading models from gs://{}/{}/", _GCS_BUCKET, _GCS_PREFIX)
    client = storage.Client(project=_GCS_PROJECT)
    bucket = client.bucket(_GCS_BUCKET)

    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in _MODEL_FILES:
        blob_name = f"{_GCS_PREFIX}/{filename}"
        dest = local_dir / filename
        logger.debug("  {} → {}", blob_name, dest)
        bucket.blob(blob_name).download_to_filename(str(dest))
    logger.info("GCS download complete.")


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------
def _resolve_model_dir() -> Path:
    """Return a local directory that contains both model files.

    Resolution order:
    1. Try to download fresh files from GCS into MODEL_ILLNESS_PATH (or /models_illness/gcp_download).
    2. If GCS download fails (no bucket configured, network error, etc.),
       fall back to the first local directory that already has both files:
         - MODEL_ILLNESS_PATH env var
         - models_illness/ relative to the source tree
    """
    from .gcs_cache import is_cache_valid

    # ── 1. Try GCS first ─────────────────────────────────────────────────
    if _GCS_BUCKET:
        gcs_dest = Path(os.getenv("MODEL_ILLNESS_PATH", "models_illness/gcp_download"))
        if is_cache_valid(gcs_dest, _MODEL_FILES):
            return gcs_dest
        try:
            _download_from_gcs(gcs_dest)
            return gcs_dest
        except Exception as exc:
            logger.warning("GCS download failed ({}), falling back to local files.", exc)

    # ── 2. Fallback: use pre-existing local files ─────────────────────────
    fallback_candidates: list[Path] = []

    env_path = os.getenv("MODEL_ILLNESS_PATH")
    if env_path:
        fallback_candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    fallback_candidates.append(here.parents[2] / "models_illness")      # backend/app/models_illness
    fallback_candidates.append(Path.cwd() / "backend/app/models_illness")
    fallback_candidates.append(Path.cwd() / "app/models_illness")

    for p in fallback_candidates:
        if p.is_dir() and all((p / f).exists() for f in _MODEL_FILES):
            logger.info("Using local model files from {}", p)
            return p

    raise FileNotFoundError(
        "GCS download failed and no local model files were found. "
        "Set GCS_BUCKET_NAME or place resnet18_plants_illness.pt + label_encoder_illness.pkl "
        "in backend/app/models_illness/."
    )

IMG_SIZE = 256
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
    weights_path  = model_dir / "resnet18_plants_illness.pt"
    encoder_path  = model_dir / "label_encoder_illness.pkl"

    with open(encoder_path, "rb") as f:
        _le = pickle.load(f)

    num_classes = len(_le.classes_)
    _model = models.resnet18(weights=None)
    _model.fc = torch.nn.Linear(_model.fc.in_features, num_classes)
    _model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()
    _ready.set()
    logger.info("Model illness ready. device={} classes={}", DEVICE, num_classes)


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