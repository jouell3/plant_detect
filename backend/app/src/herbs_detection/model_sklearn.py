import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _resolve_sklearn_dir() -> Path:
    candidates = []

    env_path = os.getenv("MODEL_SKLEARN_PATH")
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

    # Use the most recently trained config (sorted by timestamp in filename)
    configs = sorted(models_dir.glob("config_sklearn__*.json"))
    if not configs:
        raise FileNotFoundError(f"No config_sklearn__*.json found in {models_dir}")

    with open(configs[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load once at import time (singleton)
# ---------------------------------------------------------------------------
_SKLEARN_DIR = _resolve_sklearn_dir()
_config      = _load_config(_SKLEARN_DIR)

_IMG_SIZE = _config["img_size"]
_FEAT_DIM = _config["feat_dim"]
_BACKBONE = _config["backbone"]   # e.g. "efficientnet_b3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(_SKLEARN_DIR / _config["pipeline_file"], "rb") as f:
    _pipeline = pickle.load(f)

with open(_SKLEARN_DIR / _config["encoder_file"], "rb") as f:
    _le = pickle.load(f)

# Frozen EfficientNet backbone — weights are never updated at inference
_backbone_map = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b3": models.efficientnet_b3,
}
if _BACKBONE not in _backbone_map:
    raise ValueError(f"Unsupported backbone in config: {_BACKBONE}")

_backbone = _backbone_map[_BACKBONE](weights="IMAGENET1K_V1")
_backbone.classifier = nn.Identity()
_backbone = _backbone.to(DEVICE).eval()
for param in _backbone.parameters():
    param.requires_grad = False

_preprocess = transforms.Compose([
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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
    feats = _extract_features([img_path])          # (1, feat_dim)
    proba = _pipeline.predict_proba(feats)[0]      # (num_classes,)
    top3  = np.argsort(proba)[::-1][:3]
    return [(_le.classes_[i], round(float(proba[i]), 4)) for i in top3]


def predict_set(img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    """Run batch inference. Returns one (species, confidence) tuple per image."""
    results = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start : start + batch_size]
        feats = _extract_features(chunk)            # (N, feat_dim)
        proba = _pipeline.predict_proba(feats)      # (N, num_classes)
        for p in proba:
            best = int(np.argmax(p))
            results.append((_le.classes_[best], round(float(p[best]), 4)))
    return results
