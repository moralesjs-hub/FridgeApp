# FridgeApp: AI Food Recognition and Recipe Suggestions

FridgeApp is a CSCI 270 team project that uses a pretrained computer-vision classifier to suggest recipes from food in a refrigerator photo. Recipe ranking is a transparent ingredient-overlap heuristic, not a vision-language API. The included sample images work without downloading model weights; uploads are classified locally after ImageNet weights are available.

## Team

Philip Pagana, Snovi Patel, Jaden Morales, and Amelia West
College of Charleston · CSCI 270 — Applied AI

## Quickstart

Use Python 3.10 or 3.11. From the repository root:

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
streamlit run src/app.py
```

Open the local URL printed by Streamlit. Select one of the bundled examples to try the full workflow immediately, or upload a JPG/PNG (up to 10 MB). The first uploaded-image classification may download the pretrained MobileNetV3-Small ImageNet weights; allow that initial network access. If model loading or weight download fails, the app reports the problem and remains usable with the bundled sample images and manual ingredient entry.

The extra package index selects official CPU-only PyTorch wheels where available, avoiding CUDA dependencies for this CPU-oriented project.

## What the app does

1. Runs a two-stage local computer-vision detection pipeline on an uploaded image:
   - **Stage 1 (Object Detection):** Pretrained torchvision Faster R-CNN (MobileNetV3-Large FPN) outputs spatial bounding boxes and identifies key COCO food items.
   - **Stage 2 (Classification):** Pretrained MobileNetV3-Small (ImageNet) whole-image classifier supplements with a wide ingredient vocabulary (vegetables, proteins, dairy, alliums, nightshades).
2. Maps detections to canonical ingredients using a 200+ culinary alias registry.
3. Lets the user review detected items, adjust confidence thresholds, toggle bounding box visualization, and add manual or pantry items.
4. Ranks a 120-recipe catalog using a staple-aware, substitution-credited ranking heuristic:
   - Built-in in-memory inverted index (`ingredient -> set(recipe_ids)`) provides sub-millisecond retrieval.
   - Non-perishable pantry staples (salt, pepper, oil, flour, etc.) are assumed present by default so recipes aren't unfairly penalized.
   - Core ingredients missing from user inventory are checked against known culinary substitutions (e.g. lime for lemon) for 80% partial credit.
   - Renders missing items, substitutions used, estimated serving-scaled nutrition, and step-by-step instructions.

Bundled sample images are explicitly marked demonstrations: their ingredient detections are seeded for a predictable offline walkthrough, not represented as classifier predictions. Uploaded images use the actual local vision models without remote APIs.

## Repository map

```text
.ai/codingrules.md                 AI contribution and data rules
data/raw/                          Bundled sample images and raw-data guidance
data/processed/recipes.csv         Curated, normalized 120-recipe catalog
docs/project_proposal.md           Team proposal and timeline
docs/project_design_doc.md         Architecture, data pipeline, and trade-offs
src/app.py                         Streamlit interface
src/detector.py                    Two-stage local food detector & box visualizer
src/vision_model.py                Local torchvision food-label recognition
src/recipe_engine.py               Staple-aware recipe scoring & ranking
src/nutrition.py                   Serving-aware nutrition estimates
src/utils.py                       Normalization, aliases, pantry staples, & substitutions
requirements.txt                   Pinned Python dependencies
```

## Data and model

The starter recipe catalog is curated project data and is stored in `data/processed/recipes.csv`. `data/raw/README.md` documents the raw-data boundary and optional external data acquisition. No Kaggle dataset or personal image is bundled. If the team later uses a Kaggle recipe dataset, acquire it through Kaggle's official download flow, retain its license and source metadata, and keep the original files in `data/raw/`; write cleaned project-specific data to `data/processed/`.

The classifier uses torchvision's pretrained MobileNetV3-Small weights. It is not fine-tuned on a food dataset; supported ImageNet labels are mapped into a small ingredient vocabulary. A food-specific model or transfer-learning experiment can replace this component without changing the recipe-ranking module. We do not send images to third-party vision APIs.

### Optional fine-tuning

On branch `train/vision-model`, add a JSON manifest at `data/raw/annotations.json` (see `data/raw/annotations.example.json`), then run:

```powershell
python -m train --manifest data/raw/annotations.json --task classifier --epochs 5
```

Checkpoints land in `models/` and are picked up automatically by the app. Full steps: `train/README.md`.

## Rubric-oriented checklist

- **Easy execution:** pinned requirements, Python 3.10/3.11 quickstart, one-command Streamlit launch, and offline sample walkthrough.
- **Modularity:** separate UI, vision, normalization, recipe-ranking, and nutrition modules.
- **Traditional AI:** local pretrained vision classification plus explainable ingredient-overlap ranking; no vision-LLM wrapper.
- **Data handling:** raw/processed separation, source and licensing guidance, and curated processed starter data.
- **Documentation:** proposal, design document, limitations, and assistant coding rules.

## Limitations

Lighting, occlusion, camera angle, packaging, and ImageNet's generic class labels can reduce recognition quality. The vocabulary intentionally excludes unsupported foods. Recipe scores measure ingredient coverage, not taste, dietary suitability, or cooking quality. Calories and macros are approximate per-serving estimates and are not medical or dietary advice. Verify ingredients, allergens, food safety, and nutrition independently.

## Run the regression checks

Install pytest in the active environment, then run:

```powershell
python -m pip install pytest
python -m pytest -q tests
```
