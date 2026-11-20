"""Fine-tune Faster R-CNN on ingredient bounding boxes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.ingredient_classifier import detector_checkpoint_path
from train.dataset import TrainingRecord, ingredient_vocabulary


def records_with_boxes(records: list[TrainingRecord]) -> list[TrainingRecord]:
    return [record for record in records if record.boxes]


def train_food_detector(
    train_records: list[TrainingRecord],
    val_records: list[TrainingRecord],
    *,
    epochs: int = 3,
    learning_rate: float = 5e-4,
    output_path: Path | None = None,
) -> Path:
    """Fine-tune Faster R-CNN when the manifest includes bounding boxes."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    train_records = records_with_boxes(train_records)
    val_records = records_with_boxes(val_records)
    if not train_records:
        raise ValueError(
            "Detector training requires bounding boxes in the manifest "
            "(see 'boxes' in data/raw/annotations.example.json)."
        )

    for record in train_records + val_records:
        if not record.image_path.is_file():
            raise FileNotFoundError(f"Missing training image: {record.image_path}")

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.transforms import functional as F

    ingredients = ingredient_vocabulary(train_records + val_records)
    # Faster R-CNN reserves label 0 for background.
    label_to_index = {name: idx + 1 for idx, name in enumerate(ingredients)}
    device = torch.device("cpu")

    class DetectionDataset(Dataset):
        def __init__(self, rows: list[TrainingRecord]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            record = self.rows[index]
            image = Image.open(record.image_path).convert("RGB")
            width, height = image.size
            tensor = F.to_tensor(image)
            boxes = []
            labels = []
            for box in record.boxes:
                boxes.append(box.to_pixel_xyxy(width, height))
                labels.append(label_to_index[box.ingredient])
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
            return tensor, target

    def collate(
        batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        images, targets = zip(*batch)
        return list(images), list(targets)

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = len(ingredients) + 1
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_loader = DataLoader(
        DetectionDataset(train_records),
        batch_size=2,
        shuffle=True,
        collate_fn=collate,
    )

    for _epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    destination = (output_path or detector_checkpoint_path()).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": "fasterrcnn_mobilenet_v3_large_fpn",
        "ingredients": ingredients,
        "model_state": model.state_dict(),
    }
    torch.save(payload, destination)
    return destination