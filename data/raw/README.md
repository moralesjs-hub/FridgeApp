# Raw data

The three `sample_*.png` files are small, locally generated illustrative images used for the app's deterministic demo path. Their filenames and seeded demo ingredient lists are documented in `src/app.py`; they are not training data and are not classifier predictions.

No third-party dataset is included. If adding a Kaggle recipe or image dataset, download it through Kaggle's official tools, record its dataset URL, version, license, and acquisition date here, and keep the untouched files in this directory. Put only documented, cleaned derivatives in `data/processed/`. Do not add private refrigerator photos.

## Training images and labels

For vision model training:

1. Put image files in this folder (or a subfolder).
2. Copy `annotations.example.json` to **`annotations.json`** and point each `image` field at your files.
3. Run `python -m train --manifest data/raw/annotations.json` from the repo root.

See `train/README.md` for the full schema and command-line options. Trained weights are written to `models/` (gitignored).
