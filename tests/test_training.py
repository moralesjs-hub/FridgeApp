"""Tests for training manifest parsing and optional one-epoch smoke training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.dataset import (
    ingredient_vocabulary,
    load_training_manifest,
    train_val_split,
)
from train.train_classifier import train_ingredient_classifier

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MANIFEST = ROOT / "data" / "raw" / "annotations.example.json"


def test_example_manifest_loads_and_resolves_images() -> None:
    records = load_training_manifest(EXAMPLE_MANIFEST)
    assert len(records) == 3
    assert all(record.image_path.is_file() for record in records)
    labels = ingredient_vocabulary(records)
    assert "tomato" in labels
    assert "banana" in labels


def test_manifest_requires_records_key(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="records"):
        load_training_manifest(manifest)


def test_classifier_smoke_train_writes_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    records = load_training_manifest(EXAMPLE_MANIFEST)
    train_records, val_records = train_val_split(records, val_fraction=0.34, seed=1)
    output = tmp_path / "ingredient_classifier.pt"
    saved = train_ingredient_classifier(
        train_records,
        val_records,
        epochs=1,
        batch_size=2,
        output_path=output,
    )
    assert saved == output
    assert output.is_file()
    assert output.stat().st_size > 0