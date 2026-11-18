"""Load image+ingredient rows from Hugging Face datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from src.utils import normalize_ingredient
from train.dataset import TrainingRecord

DEFAULT_FOOD_INGREDIENTS_DATASET = "Scuccorese/food-ingredients-dataset"


def load_hf_training_records(
    dataset_id: str = DEFAULT_FOOD_INGREDIENTS_DATASET,
    *,
    split: str = "train",
    max_samples: int | None = None,
) -> list[TrainingRecord]:
    """Load labeled images from a Hugging Face dataset (Scuccorese-style schema)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the Hugging Face datasets package: pip install datasets"
        ) from exc

    table = load_dataset(dataset_id, split=split)
    records: list[TrainingRecord] = []

    for index, row in enumerate(table):
        if max_samples is not None and index >= max_samples:
            break

        label = normalize_ingredient(str(row.get("ingredient", "")))
        if not label:
            continue

        image_obj = row.get("image")
        if image_obj is None:
            continue
        pil_image = _coerce_pil_image(image_obj)

        records.append(
            TrainingRecord(
                image_path=Path(f"hf://{dataset_id}/{split}/{index}"),
                ingredients=[label],
                pil_image=pil_image,
            )
        )

    if not records:
        raise ValueError(
            f"No usable rows loaded from {dataset_id!r} (split={split!r}). "
            "Expected columns: ingredient, image."
        )
    return records


def _coerce_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    raise TypeError(f"Unsupported image type from Hugging Face dataset: {type(image_obj)!r}")