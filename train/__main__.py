"""CLI: python -m train --manifest data/raw/your_dataset.json"""

from __future__ import annotations

import argparse
from pathlib import Path

from train.dataset import load_training_manifest, train_val_split
from train.huggingface import DEFAULT_FOOD_INGREDIENTS_DATASET
from train.train_classifier import train_ingredient_classifier
from train.train_detector import records_with_boxes, train_food_detector
from train.train_hf_classifier import train_hf_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FridgeApp vision models from a JSON manifest in data/raw/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/annotations.json"),
        help="Path to the training JSON (default: data/raw/annotations.json).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        metavar="ID",
        help=(
            "Hugging Face dataset id (e.g. "
            f"{DEFAULT_FOOD_INGREDIENTS_DATASET}) instead of --manifest."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap when loading a Hugging Face dataset (debug/smoke runs).",
    )
    parser.add_argument(
        "--task",
        choices=("classifier", "detector", "both"),
        default="classifier",
        help="Which model to train (detector needs box annotations).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Classifier batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Classifier learning rate.")
    parser.add_argument(
        "--detector-lr",
        type=float,
        default=5e-4,
        help="Detector learning rate.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--seed", type=int, default=270, help="Shuffle seed for splitting.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue classifier training from models/ingredient_classifier.pt if labels match.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.hf_dataset:
        if args.task in ("detector", "both"):
            if args.task == "both":
                print("Skipping detector: Hugging Face ingredient images have no bounding boxes.")
            else:
                raise SystemExit(
                    "Detector training needs local box labels in a JSON manifest, not a HF image dataset."
                )
        path = train_hf_classifier(
            args.hf_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_samples=args.max_samples,
            resume=args.resume,
        )
        print(f"Finished. Checkpoint: {path}")
        return

    records = load_training_manifest(args.manifest)
    train_records, val_records = train_val_split(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if args.task in ("classifier", "both"):
        path = train_ingredient_classifier(
            train_records,
            val_records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        print(f"Saved ingredient classifier to {path}")

    if args.task in ("detector", "both"):
        if not records_with_boxes(records):
            raise SystemExit(
                "No bounding boxes found in the manifest. "
                "Add 'boxes' to records or use a COCO export for detector training."
            )
        path = train_food_detector(
            train_records,
            val_records,
            epochs=max(1, args.epochs // 2) or 1,
            learning_rate=args.detector_lr,
        )
        print(f"Saved food detector to {path}")


if __name__ == "__main__":
    main()