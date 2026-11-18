"""Fine-tune on Hugging Face image datasets without loading all images into RAM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image

from src.ingredient_classifier import checkpoint_path
from src.utils import normalize_ingredient


def _coerce_rgb(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    return image_obj.convert("RGB")


def train_hf_classifier(
    dataset_id: str,
    *,
    split: str = "train",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 270,
    max_samples: int | None = None,
    resume: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Train MobileNetV3 on a lazy Hugging Face split with per-epoch checkpoints."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    table = load_dataset(dataset_id, split=split)
    if max_samples is not None:
        table = table.select(range(min(max_samples, len(table))))

    raw_labels = [normalize_ingredient(str(value)) for value in table["ingredient"]]
    ingredients = sorted({label for label in raw_labels if label})
    if not ingredients:
        raise ValueError("No ingredient labels found in the Hugging Face dataset.")

    label_to_index = {name: idx for idx, name in enumerate(ingredients)}
    valid_indices = [idx for idx, label in enumerate(raw_labels) if label in label_to_index]

    rng = random.Random(seed)
    shuffled = list(valid_indices)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:] if val_size else shuffled

    weights_meta = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights_meta.transforms()
    device = torch.device("cpu")
    destination = (output_path or checkpoint_path()).resolve()

    class LazyIngredientDataset(Dataset):
        def __init__(self, indices: list[int]) -> None:
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
            row = table[self.indices[position]]
            image = _coerce_rgb(row["image"])
            label = normalize_ingredient(str(row["ingredient"]))
            tensor = transform(image)
            target = torch.zeros(len(ingredients), dtype=torch.float32)
            target[label_to_index[label]] = 1.0
            return tensor, target

    def _build_model() -> Any:
        model = mobilenet_v3_small(weights=weights_meta)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, len(ingredients))
        return model.to(device)

    start_epoch = 0
    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if resume and destination.is_file():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        saved_labels = payload.get("ingredients")
        if saved_labels == ingredients:
            model = _build_model()
            model.load_state_dict(payload["model_state"])
            start_epoch = int(payload.get("completed_epochs", 0))
            print(f"Resuming from epoch {start_epoch} ({destination})")
        else:
            print(
                "Checkpoint label vocabulary differs from this dataset; starting from scratch."
            )

    train_loader = DataLoader(
        LazyIngredientDataset(train_indices),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(LazyIngredientDataset(val_indices), batch_size=batch_size, shuffle=False)
        if val_indices
        else None
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1

        train_loss = running / max(batches, 1)
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.inference_mode():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    val_total += float(loss_fn(logits, batch_y).item())
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)

        completed = epoch + 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": "mobilenet_v3_small",
                "ingredients": ingredients,
                "model_state": model.state_dict(),
                "completed_epochs": completed,
                "dataset_id": dataset_id,
            },
            destination,
        )
        if val_loss is not None:
            print(
                f"Epoch {completed}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  -> {destination}"
            )
        else:
            print(
                f"Epoch {completed}/{epochs}  train_loss={train_loss:.4f}  -> {destination}"
            )

    return destination