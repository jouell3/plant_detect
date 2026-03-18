from __future__ import annotations

from functools import lru_cache
from typing import Any

from herbs_detection.params import SUPPORTED_BACKENDS, normalize_backend_name


def _load_predictor(backend: str):
    normalized_backend = normalize_backend_name(backend)

    if normalized_backend == "pytorch":
        from herbs_detection.pytorch_backend import PyTorchHerbClassifier

        return PyTorchHerbClassifier()

    if normalized_backend == "tensorflow":
        from herbs_detection.tensorflow_backend import TensorFlowHerbClassifier

        return TensorFlowHerbClassifier()

    raise ValueError(f"Unsupported backend '{backend}'.")


@lru_cache(maxsize=len(SUPPORTED_BACKENDS))
def get_predictor(backend: str):
    return _load_predictor(backend)


def get_backend_status(backend: str) -> dict[str, Any]:
    normalized_backend = normalize_backend_name(backend)

    try:
        predictor = get_predictor(normalized_backend)
        return {
            "backend": normalized_backend,
            "available": True,
            "model_path": str(predictor.model_path),
            "num_classes": predictor.num_classes,
        }
    except Exception as exc:  # pragma: no cover - status endpoint must stay resilient
        return {
            "backend": normalized_backend,
            "available": False,
            "error": str(exc),
        }


def get_all_backend_statuses() -> list[dict[str, Any]]:
    return [get_backend_status(backend) for backend in SUPPORTED_BACKENDS]


def predict(img_path: str, backend: str = "pytorch") -> tuple[str, float]:
    return get_predictor(normalize_backend_name(backend)).predict(img_path)


def predict_top3(img_path: str, backend: str = "pytorch") -> list[tuple[str, float]]:
    return get_predictor(normalize_backend_name(backend)).predict_topk(img_path, k=3)


def predict_set(
    img_paths: list[str],
    backend: str = "pytorch",
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    return get_predictor(normalize_backend_name(backend)).predict_batch(
        img_paths,
        batch_size=batch_size,
    )
