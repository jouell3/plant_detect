import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _resolve_model_dir() -> Path:
    candidates = []

    env_path = os.getenv("MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "models")     # source tree: backend/app/models
    candidates.append(Path.cwd() / "backend/app/models")
    candidates.append(Path.cwd() / "app/models")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find model directory. "
        "Set MODEL_PATH to the folder containing label_encoder.pkl and resnet18_plants.pt."
    )


_MODEL_DIR = _resolve_model_dir()  # backend/app/models/
_WEIGHTS_PATH = _MODEL_DIR / "resnet18_plants_20260322_22h50.pt"
_ENCODER_PATH = _MODEL_DIR / "label_encoder_20260322_22h50.pkl"

IMG_SIZE = 224

# ---------------------------------------------------------------------------
# Load once at import time (singleton — reused across all API requests)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(_ENCODER_PATH, "rb") as f:
    _le = pickle.load(f)

NUM_CLASSES = len(_le.classes_)

_model = models.resnet18(weights=None)
_model.fc = torch.nn.Linear(_model.fc.in_features, NUM_CLASSES)
_model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=DEVICE))
_model.to(DEVICE)
_model.eval()

# ---------------------------------------------------------------------------
# Preprocessing (same pipeline as training)
# ---------------------------------------------------------------------------
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                                       # HWC uint8 → CHW float32 in [0,1]
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),                 # ImageNet stats
])

def _load_tensor(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return _preprocess(img).unsqueeze(0).to(DEVICE)             # (1, 3, 224, 224)

def _load_batch(img_paths: list[str]) -> torch.Tensor:
    tensors = [_preprocess(Image.open(p).convert("RGB")) for p in img_paths]
    return torch.stack(tensors).to(DEVICE)                      # (N, 3, 224, 224)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(img_path: str) -> tuple[str, float]:
    """Return the top predicted species and its confidence score."""
    with torch.no_grad():
        proba = torch.softmax(_model(_load_tensor(img_path)), dim=1).squeeze()

    confidence, class_idx = proba.max(dim=0)
    species = _le.inverse_transform([class_idx.item()])[0]
    return species, round(confidence.item(), 4)


def predict_top3(img_path: str) -> list[tuple[str, float]]:
    """Return the top-3 predicted species with confidence scores."""
    with torch.no_grad():
        proba = torch.softmax(_model(_load_tensor(img_path)), dim=1).squeeze()

    top3 = proba.topk(3)
    return [
        (_le.classes_[i.item()], round(p.item(), 4))
        for i, p in zip(top3.indices, top3.values)
    ]


def predict_set(img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    """Run batch inference on a list of image paths.

    Processes images in chunks of batch_size so large sets don't exhaust GPU memory.
    Returns one (species, confidence) tuple per input image, in the same order.
    """
    results = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start : start + batch_size]
        batch = _load_batch(chunk)                               # (N, 3, 224, 224)
        with torch.no_grad():
            proba = torch.softmax(_model(batch), dim=1)          # (N, num_classes)
        confidences, class_idxs = proba.max(dim=1)
        species = _le.inverse_transform(class_idxs.cpu().tolist())
        results.extend(
            (s, round(c, 4))
            for s, c in zip(species, confidences.cpu().tolist())
        )
    return results
