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

            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Could not load the pretrained MobileNetV3 model or weights. "
                "Check the PyTorch installation and network access, then try again."
            ) from exc
        self._torch = torch
        self._weights = weights
        self._transform = weights.transforms()
        self._model = model

    def _get_image_regions(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size

        regions = [
            image,
            image.crop((0, 0, width // 2, height // 2)),
            image.crop((width // 2, 0, width, height // 2)),
            image.crop((0, height // 2, width // 2, height)),
            image.crop((width // 2, height // 2, width, height))
        ]

        return regions

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
        assert self._torch is not None
        assert self._model is not None
        assert self._weights is not None
        assert self._transform is not None

        try:
            regions = self._get_image_regions(image)
            all_probabilities = []

            for region in regions:
                tensor = self._transform(region.convert("RGB")).unsqueeze(0)

                with self._torch.inference_mode():
                    probabilities = self._torch.nn.functional.softmax(
                        self._model(tensor)[0], dim=0
                    )

                all_probabilities.append(probabilities)

            probabilities = self._torch.stack(all_probabilities).max(dim=0).values

            values, indices = self._torch.topk(
                probabilities, k=min(100, probabilities.numel())
            )
        except Exception as exc:
            raise RuntimeError("The uploaded image could not be classified.") from exc

        categories = self._weights.meta["categories"]
        print("\nTop 10 AI predictions:")
        for probability, index in zip(values.tolist()[:10], indices.tolist()[:10]):
            print(f"{categories[index]}: {probability:.4f}")
        predictions: list[Prediction] = []
        seen: set[str] = set()
        for probability, index in zip(values.tolist(), indices.tolist()):
            if probability < confidence_threshold:
                break
            label = categories[index]
            ingredient = ingredient_from_label(label)
            if ingredient and ingredient not in seen:
                predictions.append(Prediction(ingredient, probability, label))
                seen.add(ingredient)
            if len(predictions) >= top_k:
                break
        return predictions
