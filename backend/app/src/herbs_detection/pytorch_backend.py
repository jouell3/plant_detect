from __future__ import annotations

import pickle

from PIL import Image
import torch
from torchvision import models, transforms

from herbs_detection.params import IMG_SIZE, LABEL_ENCODER_PATH, PYTORCH_WEIGHTS_PATH, load_class_names


class PyTorchHerbClassifier:
    name = "pytorch"

    def __init__(self) -> None:
        self.model_path = PYTORCH_WEIGHTS_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()
        self._preprocess = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._model = self._load_model()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def _load_class_names(self) -> list[str]:
        if LABEL_ENCODER_PATH.exists():
            with open(LABEL_ENCODER_PATH, "rb") as file:
                label_encoder = pickle.load(file)
            return list(label_encoder.classes_)

        return load_class_names()

    def _load_model(self) -> torch.nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PyTorch weights not found at {self.model_path}. "
                "Train/export the PyTorch model or point MODEL_PATH to the right directory."
            )

        state_dict = torch.load(self.model_path, map_location=self.device)
        fc_weight = state_dict.get("fc.weight")
        if fc_weight is not None and int(fc_weight.shape[0]) != self.num_classes:
            raise ValueError(
                f"PyTorch model expects {int(fc_weight.shape[0])} classes, "
                f"but the herb dataset defines {self.num_classes}. Retrain/export the PyTorch model."
            )

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_tensor(self, img_path: str) -> torch.Tensor:
        image = Image.open(img_path).convert("RGB")
        return self._preprocess(image).unsqueeze(0).to(self.device)

    def _load_batch(self, img_paths: list[str]) -> torch.Tensor:
        batch = [self._preprocess(Image.open(path).convert("RGB")) for path in img_paths]
        return torch.stack(batch).to(self.device)

    def predict(self, img_path: str) -> tuple[str, float]:
        with torch.no_grad():
            probabilities = torch.softmax(self._model(self._load_tensor(img_path)), dim=1).squeeze(0)

        confidence, class_index = probabilities.max(dim=0)
        return self.class_names[class_index.item()], round(confidence.item(), 4)

    def predict_topk(self, img_path: str, k: int = 3) -> list[tuple[str, float]]:
        with torch.no_grad():
            probabilities = torch.softmax(self._model(self._load_tensor(img_path)), dim=1).squeeze(0)

        topk = probabilities.topk(min(k, self.num_classes))
        return [
            (self.class_names[index.item()], round(score.item(), 4))
            for index, score in zip(topk.indices, topk.values)
        ]

    def predict_batch(self, img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []

        for start in range(0, len(img_paths), batch_size):
            current_paths = img_paths[start : start + batch_size]
            batch = self._load_batch(current_paths)
            with torch.no_grad():
                probabilities = torch.softmax(self._model(batch), dim=1)

            confidences, class_indices = probabilities.max(dim=1)
            for index, confidence in zip(class_indices.tolist(), confidences.tolist()):
                results.append((self.class_names[index], round(confidence, 4)))

        return results
