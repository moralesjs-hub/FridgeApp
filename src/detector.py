"""Multi-strategy local food object detector with spatial bounding-box grounding.

Architecture
------------
The detector uses a two-stage pipeline that combines spatial object detection
with single-label classification to maximise ingredient coverage:

Stage 1 – Spatial Object Detector (Faster R-CNN + MobileNetV3-Large FPN)
    Pretrained on COCO, which includes 12 food-relevant categories:
    banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
    donut, cake, bowl, bottle.

    Outputs bounding boxes with normalised coordinates [ymin, xmin, ymax, xmax]
    so the UI can draw visual grounding overlays on the uploaded photo.

Stage 2 – Whole-Image Classifier (MobileNetV3-Small, ImageNet-1K)
    Supplements the detector for food classes that COCO does not label
    (e.g. tomato, egg, cheese, chicken, mushroom, potato, yogurt, etc.).
    Only ImageNet labels with an explicit supported-ingredient mapping are used.

The two results are merged, deduplicated by canonical name, and filtered by
the user-selected confidence threshold before being returned to the caller.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.utils import ingredient_from_label, normalize_ingredient


# ---------------------------------------------------------------------------
# Colour palette for category-based visual grounding
# ---------------------------------------------------------------------------
_CATEGORY_COLORS: dict[str, str] = {
    "produce": "#2ecc71",    # emerald green
    "fruit": "#27ae60",      # dark green
    "protein": "#e74c3c",    # coral red
    "grain": "#f39c12",      # orange
    "dairy": "#3498db",      # blue
    "other": "#9b59b6",      # purple
}

# COCO food labels that the Faster R-CNN model directly detects
_COCO_FOOD_LABELS: set[str] = {
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
}

# COCO food label → canonical ingredient name
_COCO_TO_INGREDIENT: dict[str, str] = {
    "banana": "banana",
    "apple": "apple",
    "sandwich": "bread",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "sausage",
    "pizza": "cheese",
    "donut": "bread",
    "cake": "flour",
}

# Default ingredient category for display coloring
_INGREDIENT_CATEGORY: dict[str, str] = {
    "banana": "fruit",
    "apple": "fruit",
    "orange": "fruit",
    "strawberry": "fruit",
    "blueberry": "fruit",
    "avocado": "fruit",
    "pineapple": "fruit",
    "lemon": "fruit",
    "lime": "fruit",
    "pear": "fruit",
    "broccoli": "produce",
    "carrot": "produce",
    "tomato": "produce",
    "onion": "produce",
    "garlic": "produce",
    "bell pepper": "produce",
    "spinach": "produce",
    "potato": "produce",
    "mushroom": "produce",
    "cucumber": "produce",
    "lettuce": "produce",
    "zucchini": "produce",
    "cauliflower": "produce",
    "cabbage": "produce",
    "corn": "produce",
    "eggplant": "produce",
    "asparagus": "produce",
    "pea": "produce",
    "squash": "produce",
    "chicken": "protein",
    "beef": "protein",
    "pork": "protein",
    "salmon": "protein",
    "tuna": "protein",
    "shrimp": "protein",
    "sausage": "protein",
    "bacon": "protein",
    "turkey": "protein",
    "tofu": "protein",
    "egg": "protein",
    "cheese": "dairy",
    "milk": "dairy",
    "yogurt": "dairy",
    "butter": "dairy",
    "cream": "dairy",
    "pasta": "grain",
    "rice": "grain",
    "bread": "grain",
    "oats": "grain",
    "flour": "grain",
    "tortilla": "grain",
    "quinoa": "grain",
}


@dataclass
class BoundingBox:
    """Normalised bounding box in [0, 1] range."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float

    def to_pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates (left, top, right, bottom)."""
        left = int(self.xmin * width)
        top = int(self.ymin * height)
        right = max(int(self.xmax * width), left + 1)
        bottom = max(int(self.ymax * height), top + 1)
        return left, top, right, bottom


@dataclass
class DetectedEntity:
    """A food item detected in the image."""
    ingredient: str          # canonical normalised name, e.g. "tomato"
    confidence: float        # 0.0–1.0
    source: str              # "coco_detector" | "imagenet_classifier"
    category: str = "other"
    box: BoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingredient": self.ingredient,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "category": self.category,
            "box": vars(self.box) if self.box else None,
        }


@dataclass
class DetectionResult:
    """Full result of one detector call."""
    entities: list[DetectedEntity] = field(default_factory=list)
    unique_ingredients: list[str] = field(default_factory=list)
    annotated_image: Image.Image | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class FoodDetector:
    """Local CPU food detector combining spatial detection and image classification.

    The model weights are loaded lazily on first inference so that the app
    starts immediately and the sample demo path never triggers a download.
    """

    def __init__(self) -> None:
        self._rcnn: Any = None
        self._rcnn_weights: Any = None
        self._rcnn_transform: Any = None
        self._classifier: Any = None
        self._cls_weights: Any = None
        self._cls_transform: Any = None
        self._torch: Any = None
        self._rcnn_error: str | None = None
        self._cls_error: str | None = None

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _load_rcnn(self) -> None:
        """Load Faster R-CNN + MobileNetV3 FPN (COCO pretrained) lazily."""
        if self._rcnn is not None or self._rcnn_error:
            return
        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_fpn,
            )
            self._torch = torch
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            model.eval()
            self._rcnn = model
            self._rcnn_weights = weights
            self._rcnn_transform = weights.transforms()
        except Exception as exc:
            self._rcnn_error = (
                f"Object detection model unavailable: {exc}. "
                "Falling back to image classifier."
            )

    def _load_classifier(self) -> None:
        """Load MobileNetV3-Small (ImageNet) for classifier fallback."""
        if self._classifier is not None or self._cls_error:
            return
        try:
            import torch
            from torchvision.models import (
                MobileNet_V3_Small_Weights,
                mobilenet_v3_small,
            )
            self._torch = self._torch or torch
            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.eval()
            self._classifier = model
            self._cls_weights = weights
            self._cls_transform = weights.transforms()
        except Exception as exc:
            self._cls_error = (
                f"Image classifier unavailable: {exc}. "
                "Add ingredients manually."
            )

    # ------------------------------------------------------------------
    # Detection stages
    # ------------------------------------------------------------------

    def _run_rcnn(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[DetectedEntity]:
        """Stage 1: Faster R-CNN COCO spatial object detection."""
        self._load_rcnn()
        if self._rcnn_error:
            return []

        import torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        tensor = self._rcnn_transform(rgb)
        with torch.no_grad():
            outputs = self._rcnn([tensor])[0]

        categories = self._rcnn_weights.meta["categories"]
        entities: list[DetectedEntity] = []

        for box, label_idx, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            conf = float(score)
            if conf < confidence_threshold:
                continue
            label = categories[int(label_idx)]
            if label not in _COCO_FOOD_LABELS:
                continue

            ingredient = _COCO_TO_INGREDIENT.get(label)
            if not ingredient:
                continue
            ingredient = normalize_ingredient(ingredient)

            x1, y1, x2, y2 = box.tolist()
            bbox = BoundingBox(
                ymin=max(0.0, y1 / height),
                xmin=max(0.0, x1 / width),
                ymax=min(1.0, y2 / height),
                xmax=min(1.0, x2 / width),
            )
            category = _INGREDIENT_CATEGORY.get(ingredient, "other")
            entities.append(
                DetectedEntity(
                    ingredient=ingredient,
                    confidence=conf,
                    source="coco_detector",
                    category=category,
                    box=bbox,
                )
            )

        return entities

    def _run_classifier(
        self,
        image: Image.Image,
        confidence_threshold: float,
        top_k: int = 10,
    ) -> list[DetectedEntity]:
        """Stage 2: MobileNetV3-Small ImageNet whole-image classification."""
        self._load_classifier()
        if self._cls_error:
            return []

        import torch

        rgb = image.convert("RGB")
        tensor = self._cls_transform(rgb)
        with torch.no_grad():
            logits = self._classifier(tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]

        top_scores, top_indices = torch.topk(probs, min(top_k, len(probs)))
        categories = self._cls_weights.meta["categories"]

        entities: list[DetectedEntity] = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if score < confidence_threshold:
                continue
            label = categories[idx]
            ingredient = ingredient_from_label(label)
            if not ingredient:
                continue
            category = _INGREDIENT_CATEGORY.get(ingredient, "other")
            entities.append(
                DetectedEntity(
                    ingredient=ingredient,
                    confidence=float(score),
                    source="imagenet_classifier",
                    category=category,
                    box=None,  # whole-image classifier has no spatial coordinates
                )
            )

        return entities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
        annotate: bool = True,
    ) -> DetectionResult:
        """Detect food ingredients in *image* and optionally draw overlays.

        Returns a :class:`DetectionResult` with entities sorted by descending
        confidence.  Duplicate canonical ingredient names are collapsed — the
        highest-confidence detection wins and inherits its bounding box.
        """
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold!r}"
            )

        # Stage 1 — COCO spatial detector
        rcnn_entities = self._run_rcnn(image, confidence_threshold)

        # Stage 2 — ImageNet classifier (supplement)
        cls_entities = self._run_classifier(image, confidence_threshold)

        # Merge: prefer detector results (they have bounding boxes), then
        # fill in classifier results for ingredients not already detected.
        all_entities = rcnn_entities + cls_entities
        all_entities.sort(key=lambda e: -e.confidence)

        seen: set[str] = set()
        merged: list[DetectedEntity] = []
        for entity in all_entities:
            if entity.ingredient not in seen:
                seen.add(entity.ingredient)
                merged.append(entity)

        unique_ingredients = [e.ingredient for e in merged]

        # Build annotated image if requested
        annotated: Image.Image | None = None
        if annotate and merged:
            annotated = self._annotate(image, merged)

        # Compose human-readable error note when both models failed
        error: str | None = None
        if self._rcnn_error and self._cls_error:
            error = f"{self._rcnn_error} | {self._cls_error}"

        return DetectionResult(
            entities=merged,
            unique_ingredients=unique_ingredients,
            annotated_image=annotated,
            error=error if not merged else None,
        )

    @property
    def model_errors(self) -> dict[str, str | None]:
        """Return any lazy-load errors from either model stage."""
        return {
            "detector": self._rcnn_error,
            "classifier": self._cls_error,
        }

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate(image: Image.Image, entities: list[DetectedEntity]) -> Image.Image:
        """Draw colour-coded bounding boxes and label badges on a copy of *image*."""
        annotated = image.convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(annotated)

        width, height = annotated.size

        try:
            font_size = max(12, int(height * 0.022))
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()

        for entity in entities:
            if entity.box is None:
                continue

            color_hex = _CATEGORY_COLORS.get(entity.category, _CATEGORY_COLORS["other"])
            color_rgb = ImageColor.getrgb(color_hex)
            left, top, right, bottom = entity.box.to_pixel_box(width, height)

            # Semi-transparent fill
            draw_overlay.rectangle(
                [left, top, right, bottom],
                fill=(*color_rgb, 45),
            )
            # Border
            draw.rectangle(
                [left, top, right, bottom],
                outline=color_hex,
                width=3,
            )

            # Badge label
            label_text = f"{entity.ingredient} {entity.confidence:.0%}"
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(label_text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw, th = 80, 16

            pad = 4
            bx0 = left
            by0 = max(0, top - th - pad * 2)
            bx1 = bx0 + tw + pad * 2
            by1 = by0 + th + pad * 2

            draw.rectangle([bx0, by0, bx1, by1], fill=color_hex)
            draw.text((bx0 + pad, by0 + pad), label_text, fill="white", font=font)

        final = Image.alpha_composite(annotated, overlay).convert("RGB")
        return final

