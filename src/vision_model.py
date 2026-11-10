"""Local pretrained image classification and supported-label mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from PIL import Image

from src.utils import ingredient_from_label


@dataclass(frozen=True)
class Prediction:
    """A supported ingredient candidate and its ImageNet class probability."""

    ingredient: str
    confidence: float
    label: str


class FoodRecognizer:
    """Run a lazily loaded MobileNetV3-Small ImageNet classifier."""

    def __init__(self) -> None:
        self._model: Any = None
        self._weights: Any = None
        self._transform: Any = None
        self._torch: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision.models import (
                MobileNet_V3_Small_Weights,
                mobilenet_v3_small,
            )

            self._weights = MobileNet_V3_Small_Weights.DEFAULT
            self._model = mobilenet_v3_small(weights=self._weights).eval()
            self._transform = self._weights.transforms()
            self._torch = torch
        except Exception as exc:
            raise RuntimeError(
                "Could not load the pretrained MobileNetV3 model or weights. "
                "Check the PyTorch installation and network access, then try again."
            ) from exc

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.15,
        top_k: int = 8,
    ) -> list[Prediction]:
        """Return distinct supported ingredients above a probability threshold."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self._load_model()

        try:
            tensor = self._transform(image.convert("RGB")).unsqueeze(0)
            with self._torch.inference_mode():
                probs = self._torch.nn.functional.softmax(
                    self._model(tensor)[0], dim=0
                )
            values, indices = self._torch.topk(probs, k=min(100, probs.numel()))
        except Exception as exc:
            raise RuntimeError("The uploaded image could not be classified.") from exc

        categories = self._weights.meta["categories"]
        predictions: list[Prediction] = []
        seen: set[str] = set()
        for prob, idx in zip(values.tolist(), indices.tolist()):
            if prob < confidence_threshold or len(predictions) >= top_k:
                break
            label = categories[idx]
            ingredient = ingredient_from_label(label)
            if ingredient and ingredient not in seen:
                predictions.append(Prediction(ingredient, prob, label))
                seen.add(ingredient)
        return predictions
