"""Fine-tune MobileNetV3-Small for multi-label ingredient recognition."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.ingredient_classifier import checkpoint_path
from train.dataset import TrainingRecord, ingredient_vocabulary


def _open_record_image(record: TrainingRecord) -> Image.Image:
    if record.pil_image is not None:
        return record.pil_image.convert("RGB")
    return Image.open(record.image_path).convert("RGB")


def _missing_paths(records: list[TrainingRecord]) -> list[Path]:
    return [
        record.image_path
        for record in records
        if record.pil_image is None and not record.image_path.is_file()
    ]


def train_ingredient_classifier(
    train_records: list[TrainingRecord],
    val_records: list[TrainingRecord],
    *,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    output_path: Path | None = None,
) -> Path:
    """Train and persist a multi-label ingredient classifier checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    missing = _missing_paths(train_records + val_records)
    if missing:
        sample = "\n".join(str(path) for path in missing[:5])
        extra = f"\n... and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise FileNotFoundError(
            "Some training images are missing on disk. "
            f"Place files under data/raw/ (see data/raw/annotations.example.json).\n{sample}{extra}"
        )

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    ingredients = ingredient_vocabulary(train_records + val_records)
    if not ingredients:
        raise ValueError("No ingredient labels found in the training manifest.")

    label_to_index = {name: idx for idx, name in enumerate(ingredients)}
    weights_meta = MobileNet_V3_Small_Weights.DEFAULT
    train_transform = weights_meta.transforms()
    device = torch.device("cpu")

    class IngredientDataset(Dataset):
        def __init__(self, rows: list[TrainingRecord]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            record = self.rows[index]
            image = _open_record_image(record)
            tensor = train_transform(image)
            target = torch.zeros(len(ingredients), dtype=torch.float32)
            for name in record.ingredients:
                target[label_to_index[name]] = 1.0
            return tensor, target

    model = mobilenet_v3_small(weights=weights_meta)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(ingredients))
    model.to(device)

    train_loader = DataLoader(
        IngredientDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(IngredientDataset(val_records), batch_size=batch_size, shuffle=False)
        if val_records
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            count = 0
            with torch.inference_mode():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    val_loss += float(loss_fn(logits, batch_y).item())
                    count += 1
            _ = val_loss / max(count, 1)

    destination = (output_path or checkpoint_path()).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": "mobilenet_v3_small",
        "ingredients": ingredients,
        "model_state": model.state_dict(),
    }
    torch.save(payload, destination)
    return destination