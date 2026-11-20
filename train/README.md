# Training FridgeApp vision models

## 1. Place your dataset

Copy your labeled JSON manifest to **`data/raw/annotations.json`** (or pass `--manifest`).

Use the team format in `data/raw/annotations.example.json`:

- **`image`**: path to a JPG/PNG, relative to `dataset_root` (default: manifest folder)
- **`ingredients`**: canonical ingredient names (normalized with `src.utils.normalize_ingredient`)
- **`boxes`** (optional): `[ymin, xmin, ymax, xmax]` in **0–1** normalized coordinates for detector training

Image files live under `data/raw/` (or a subfolder). Document license and source in `data/raw/README.md`.

COCO-style exports with top-level `images`, `annotations`, and `categories` are also accepted.

## 2. Run training

### Hugging Face (recommended image dataset)

```powershell
python -m pip install datasets
python -m train --hf-dataset Scuccorese/food-ingredients-dataset --task classifier --epochs 5
```

This downloads ~1.3 GB on first run (6,676 ingredient photos). Use `--max-samples 500` for a quicker smoke run.

### Local JSON manifest

From the repo root with the project venv active:

```powershell
python -m train --manifest data/raw/annotations.json --task classifier --epochs 5
```

Options:

| Flag | Meaning |
|------|---------|
| `--task classifier` | Multi-label MobileNetV3-Small (default) |
| `--task detector` | Faster R-CNN when records include `boxes` |
| `--task both` | Train both checkpoints |
| `--epochs` | Training epochs (default 5) |
| `--batch-size` | Classifier batch size (default 8) |
| `--val-fraction` | Holdout split (default 0.2) |

Outputs:

- `models/ingredient_classifier.pt`
- `models/food_detector.pt` (detector task only)

## 3. Use trained weights in the app

Restart Streamlit. If `models/ingredient_classifier.pt` exists, inference uses your fine-tuned labels instead of ImageNet mapping.

```powershell
streamlit run src/app.py
```

## Notes

- Training defaults to **CPU** to match the project quickstart.
- Small datasets are fine for coursework; expect overfitting below ~100 images per class.
- The app still works without checkpoints (pretrained COCO + ImageNet).