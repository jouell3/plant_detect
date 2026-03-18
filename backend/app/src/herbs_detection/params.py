from __future__ import annotations

import os
import pickle
from pathlib import Path

IMG_SIZE = 224
BATCH_SIZE = 32
SUPPORTED_BACKENDS = ("pytorch", "tensorflow")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALL_IMAGES_DIR = PROJECT_ROOT / "data" / "all_images"
DEFAULT_API_URL = os.getenv("PLANT_DETECT_API_URL", "http://localhost:8000")

PYTORCH_MODEL_FILENAME = "resnet18_plants.pt"
TENSORFLOW_MODEL_FILENAME = "EfficientNetB0_best.keras"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
CLASS_NAMES_FILENAME = "class_names.txt"


def resolve_model_dir() -> Path:
    candidates: list[Path] = []

    env_path = os.getenv("MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "backend" / "app" / "models",
            Path.cwd() / "models",
            Path.cwd() / "backend" / "app" / "models",
            Path.cwd() / "app" / "models",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / "models"


MODEL_DIR = resolve_model_dir()
PYTORCH_WEIGHTS_PATH = MODEL_DIR / PYTORCH_MODEL_FILENAME
TENSORFLOW_WEIGHTS_PATH = MODEL_DIR / TENSORFLOW_MODEL_FILENAME
LABEL_ENCODER_PATH = MODEL_DIR / LABEL_ENCODER_FILENAME
CLASS_NAMES_PATH = MODEL_DIR / CLASS_NAMES_FILENAME


def normalize_backend_name(backend: str | None) -> str:
    if backend is None:
        return "pytorch"

    normalized = backend.strip().lower()
    aliases = {
        "pt": "pytorch",
        "torch": "pytorch",
        "pytorch": "pytorch",
        "tf": "tensorflow",
        "keras": "tensorflow",
        "tensorflow": "tensorflow",
    }

    if normalized not in aliases:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Unknown backend '{backend}'. Supported values: {supported}.")

    return aliases[normalized]


def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        return [line.strip() for line in CLASS_NAMES_PATH.read_text().splitlines() if line.strip()]

    if LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, "rb") as file:
            label_encoder = pickle.load(file)
        return list(label_encoder.classes_)

    if ALL_IMAGES_DIR.exists():
        classes = sorted(
            {
                path.stem.rsplit("_", 1)[0]
                for path in ALL_IMAGES_DIR.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and "_" in path.stem
                and path.stem.rsplit("_", 1)[1].isdigit()
            }
        )
        if classes:
            return classes

    raise FileNotFoundError(
        "Unable to resolve class names. Provide one of: "
        f"{CLASS_NAMES_PATH}, {LABEL_ENCODER_PATH}, or herb images under {ALL_IMAGES_DIR}."
    )
