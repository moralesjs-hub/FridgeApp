"""Shared fine-tuned ingredient classifier checkpoint loading and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "ingredient_classifier.pt"
DEFAULT_DETECTOR_CHECKPOINT = ROOT / "models" / "food_detector.pt"


@dataclass(frozen=True)
class FineTunedClassifier:
    """A trained MobileNetV3-Small multi-label ingredient model."""

    model: Any
    ingredients: tuple[str, ...]
    transform: Any
    torch: Any


@dataclass(frozen=True)
class FineTunedDetector:
    """A trained Faster R-CNN ingredient detector."""

    model: Any
    ingredients: tuple[str, ...]
    transform: Any
    torch: Any


def checkpoint_path() -> Path:
    """Return the default on-disk path for a fine-tuned classifier."""
    return DEFAULT_CHECKPOINT


def detector_checkpoint_path() -> Path:
    """Return the default on-disk path for a fine-tuned Faster R-CNN."""
    return DEFAULT_DETECTOR_CHECKPOINT


def try_load_fine_tuned(path: Path | None = None) -> FineTunedClassifier | None:
    """Load a fine-tuned classifier if the checkpoint file exists."""
    resolved = path or checkpoint_path()
    if not resolved.is_file():
        return None

    try:
        import torch
        from torchvision.models import mobilenet_v3_small
    except Exception:
        return None

    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    ingredients = payload.get("ingredients")
    state_dict = payload.get("model_state")
    if not ingredients or state_dict is None:
        return None

    ingredient_list = tuple(str(item) for item in ingredients)
    num_classes = len(ingredient_list)

    try:
        from torchvision.models import MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        transform = weights.transforms()
    except Exception:
        return None

    return FineTunedClassifier(
        model=model,
        ingredients=ingredient_list,
        transform=transform,
        torch=torch,
    )


def predict_multilabel(
    bundle: FineTunedClassifier,
    image: Image.Image,
    confidence_threshold: float = 0.15,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return ingredient names and confidences from a fine-tuned multi-label head."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    tensor = bundle.transform(image.convert("RGB")).unsqueeze(0)
    with bundle.torch.inference_mode():
        logits = bundle.model(tensor)
        probs = bundle.torch.sigmoid(logits)[0]

    scored = [
        (bundle.ingredients[idx], float(probs[idx].item()))
        for idx in range(len(bundle.ingredients))
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results: list[tuple[str, float]] = []
    for ingredient, confidence in scored:
        if confidence < confidence_threshold:
            break
        results.append((ingredient, confidence))
        if len(results) >= top_k:
            break
    return results


def try_load_fine_tuned_detector(path: Path | None = None) -> FineTunedDetector | None:
    """Load a fine-tuned Faster R-CNN detector when its checkpoint exists."""
    resolved = path or detector_checkpoint_path()
    if not resolved.is_file():
        return None

    try:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except Exception:
        return None

    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    ingredients = payload.get("ingredients")
    state_dict = payload.get("model_state")
    if not ingredients or state_dict is None:
        return None

    ingredient_list = tuple(str(item) for item in ingredients)
    num_classes = len(ingredient_list) + 1

    try:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        transform = weights.transforms()
    except Exception:
        return None

    return FineTunedDetector(
        model=model,
        ingredients=ingredient_list,
        transform=transform,
        torch=torch,
    )