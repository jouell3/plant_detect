from __future__ import annotations

import numpy as np
from PIL import Image

from herbs_detection.params import IMG_SIZE, TENSORFLOW_WEIGHTS_PATH, load_class_names


class TensorFlowHerbClassifier:
    name = "tensorflow"

    def __init__(self) -> None:
        self.model_path = TENSORFLOW_WEIGHTS_PATH
        self.class_names = load_class_names()
        self._tf = None
        self._model = self._load_model()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TensorFlow weights not found at {self.model_path}. "
                "Train/export the TensorFlow model or point MODEL_PATH to the right directory."
            )

        import tensorflow as tf

        self._tf = tf
        model = tf.keras.models.load_model(self.model_path)
        output_width = int(model.output_shape[-1])
        if output_width != self.num_classes:
            raise ValueError(
                f"TensorFlow model expects {output_width} classes, "
                f"but the herb dataset defines {self.num_classes}. Retrain/export the TF model."
            )
        return model

    def _load_array(self, img_path: str) -> np.ndarray:
        image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        # EfficientNet in TF 2.15 expects raw float pixels in [0, 255].
        image_array = np.asarray(image, dtype=np.float32)
        return np.expand_dims(image_array, axis=0)

    def _load_batch(self, img_paths: list[str]) -> np.ndarray:
        return np.concatenate([self._load_array(path) for path in img_paths], axis=0)

    def _predict_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        probabilities = self._model.predict(inputs, verbose=0)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        if probabilities.ndim == 1:
            probabilities = np.expand_dims(probabilities, axis=0)

        row_sums = probabilities.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-2):
            probabilities = self._tf.nn.softmax(probabilities, axis=1).numpy()

        return probabilities

    def predict(self, img_path: str) -> tuple[str, float]:
        probabilities = self._predict_probabilities(self._load_array(img_path))[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        return self.class_names[class_index], round(confidence, 4)

    def predict_topk(self, img_path: str, k: int = 3) -> list[tuple[str, float]]:
        probabilities = self._predict_probabilities(self._load_array(img_path))[0]
        top_indices = np.argsort(probabilities)[::-1][: min(k, self.num_classes)]
        return [
            (self.class_names[index], round(float(probabilities[index]), 4))
            for index in top_indices
        ]

    def predict_batch(self, img_paths: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []

        for start in range(0, len(img_paths), batch_size):
            current_paths = img_paths[start : start + batch_size]
            probabilities = self._predict_probabilities(self._load_batch(current_paths))
            class_indices = np.argmax(probabilities, axis=1)
            confidences = probabilities[np.arange(len(class_indices)), class_indices]
            for index, confidence in zip(class_indices.tolist(), confidences.tolist()):
                results.append((self.class_names[index], round(float(confidence), 4)))

        return results
