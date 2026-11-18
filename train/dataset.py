"""Load FridgeApp JSON training manifests and detection records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.utils import normalize_ingredient


@dataclass(frozen=True)
class BoxAnnotation:
    """Normalized bounding box in ymin, xmin, ymax, xmax order (0–1)."""

    ingredient: str
    ymin: float
    xmin: float
    ymax: float
    xmax: float

    def to_pixel_xyxy(self, width: int, height: int) -> list[float]:
        """Convert to torchvision detection format [xmin, ymin, xmax, ymax] in pixels."""
        left = self.xmin * width
        top = self.ymin * height
        right = self.xmax * width
        bottom = self.ymax * height
        return [left, top, right, bottom]


@dataclass
class TrainingRecord:
    """One labeled image used for classifier and/or detector training."""

    image_path: Path
    ingredients: list[str] = field(default_factory=list)
    boxes: list[BoxAnnotation] = field(default_factory=list)
    pil_image: Any = None


def _resolve_image_path(manifest_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (manifest_dir / path).resolve()


def _parse_record(entry: dict[str, Any], manifest_dir: Path, dataset_root: Path | None) -> TrainingRecord:
    image_value = entry.get("image") or entry.get("file_name")
    if not image_value:
        raise ValueError("Each record needs an 'image' or 'file_name' field.")

    root = dataset_root or manifest_dir
    image_path = _resolve_image_path(root, str(image_value))

    raw_ingredients = entry.get("ingredients") or []
    ingredients = []
    for item in raw_ingredients:
        normalized = normalize_ingredient(str(item))
        if normalized and normalized not in ingredients:
            ingredients.append(normalized)

    boxes: list[BoxAnnotation] = []
    for box_entry in entry.get("boxes", []):
        ingredient = normalize_ingredient(str(box_entry.get("ingredient", "")))
        if not ingredient:
            continue
        bbox = box_entry.get("bbox") or box_entry.get("bbox_norm")
        if not bbox or len(bbox) != 4:
            continue
        ymin, xmin, ymax, xmax = (float(value) for value in bbox)
        boxes.append(
            BoxAnnotation(
                ingredient=ingredient,
                ymin=ymin,
                xmin=xmin,
                ymax=ymax,
                xmax=xmax,
            )
        )
        if ingredient not in ingredients:
            ingredients.append(ingredient)

    return TrainingRecord(image_path=image_path, ingredients=ingredients, boxes=boxes)


def _load_coco_manifest(data: dict[str, Any], manifest_dir: Path) -> list[TrainingRecord]:
    images = {item["id"]: item for item in data.get("images", [])}
    categories = {item["id"]: normalize_ingredient(str(item["name"])) for item in data.get("categories", [])}
    by_image: dict[int, TrainingRecord] = {}

    for image_id, meta in images.items():
        file_name = meta.get("file_name")
        if not file_name:
            continue
        image_path = _resolve_image_path(manifest_dir, str(file_name))
        by_image[image_id] = TrainingRecord(image_path=image_path)

    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if image_id not in by_image or category_id not in categories or not bbox:
            continue
        if len(bbox) != 4:
            continue
        ingredient = categories[category_id]
        if not ingredient:
            continue
        xmin, ymin, w, h = (float(value) for value in bbox)
        image_meta = images[image_id]
        width = float(image_meta.get("width") or 1)
        height = float(image_meta.get("height") or 1)
        xmax = xmin + w
        ymax = ymin + h
        record = by_image[image_id]
        record.boxes.append(
            BoxAnnotation(
                ingredient=ingredient,
                ymin=ymin / height,
                xmin=xmin / width,
                ymax=ymax / height,
                xmax=xmax / width,
            )
        )
        if ingredient not in record.ingredients:
            record.ingredients.append(ingredient)

    return list(by_image.values())


def load_training_manifest(manifest_path: Path) -> list[TrainingRecord]:
    """Load labeled images from a FridgeApp JSON manifest or a COCO-style export."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        if (
            data
            and isinstance(data[0], dict)
            and "ingredients" in data[0]
            and "image" not in data[0]
            and "file_name" not in data[0]
        ):
            raise ValueError(
                "This JSON is a text recipe list (ingredients only), not a vision training "
                "manifest. FridgeApp vision training needs image paths plus labels — see "
                "data/raw/annotations.example.json. Use train.json to expand recipes "
                "(data/processed/recipes.csv), not to train the image classifier."
            )
        raise ValueError(
            "Training manifest must be a JSON object with a 'records' array, not a bare list."
        )

    if not isinstance(data, dict):
        raise ValueError("Training manifest must be a JSON object.")

    manifest_dir = manifest_path.parent
    dataset_root_raw = data.get("dataset_root")
    dataset_root = Path(dataset_root_raw).resolve() if dataset_root_raw else None

    if "images" in data and "annotations" in data:
        return _load_coco_manifest(data, dataset_root or manifest_dir)

    records_raw = data.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("Manifest must contain a non-empty 'records' array or COCO keys.")

    records: list[TrainingRecord] = []
    for entry in records_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each record must be a JSON object.")
        records.append(_parse_record(entry, manifest_dir, dataset_root))
    return records


def ingredient_vocabulary(records: Iterable[TrainingRecord]) -> list[str]:
    """Collect a stable sorted list of ingredient labels seen in the dataset."""
    labels: set[str] = set()
    for record in records:
        for ingredient in record.ingredients:
            labels.add(ingredient)
        for box in record.boxes:
            labels.add(box.ingredient)
    return sorted(labels)


def train_val_split(
    records: list[TrainingRecord],
    val_fraction: float = 0.2,
    seed: int = 270,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Shuffle and split records for training and validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if len(records) < 2:
        return records, []

    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    if val_size >= len(shuffled):
        val_size = len(shuffled) - 1
    return shuffled[val_size:], shuffled[:val_size]