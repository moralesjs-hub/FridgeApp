# Ingredia

*Good food starts with what you have.*

Ingredia is an AI-powered recipe recommendation application that helps users discover meals based on ingredients they already have at home. Users can upload a photo of their fridge, review detected ingredients, and explore recipes with estimated calories and nutritional information.

The application uses Google Gemini for ingredient recognition, with a local computer vision model as a backup. It also includes dietary preferences, customizable serving sizes, shopping lists, and favorite recipes.

Developed as a CSCI 270 Applied AI team project at the College of Charleston.

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

## Gemini API Setup (Optional)

Ingredia uses Google Gemini to improve ingredient recognition
and identify multiple food items in an image.

To enable Gemini:

1. Visit https://aistudio.google.com/apikey
2. Create your own Gemini API key.
3. Create a `.env` file in the root project folder.
4. Add the following line to the file:

```env
GEMINI_API_KEY=your_api_key_here
```

5. Save the file and restart the application.

The API key is private and should never be uploaded to GitHub.
The `.env` file is excluded through `.gitignore`.

When Gemini is enabled, uploaded images are sent to Google's
Gemini API for ingredient recognition. An internet connection
is required.

If an API key is not provided or the Gemini service fails,
the application automatically uses the local computer vision
models instead.

## What the app does

1. Identifies ingredients from uploaded food or refrigerator images:
   - Google Gemini: Recognizes multiple food items and displays bounding boxes and confidence scores.
   - Local object detection: Uses pretrained Faster R-CNN (MobileNetV3-Large FPN) when Gemini is unavailable.
   - Local classification: Uses pretrained MobileNetV3-Small (ImageNet) to identify additional ingredients.

2. Maps detected food items to standardized ingredient names using a 200+ culinary alias registry.

3. Allows users to review, remove, and manually add ingredients, adjust confidence thresholds, and display bounding boxes.

4. Recommends recipes from a catalog of 120 recipes based on available ingredients, pantry staples, and ingredient substitutions.

5. Allows users to filter recipes based on dietary preferences.

6. Displays estimated calories, nutritional information, adjustable serving sizes, and step-by-step cooking instructions.

7. Allows users to create, edit, and download shopping lists for missing ingredients.

8. Allows users to save their favorite recipes.

Bundled sample images use predefined ingredients for offline
demonstrations. Uploaded images use Gemini when configured
or the local computer vision models as a fallback.

## Repository map

```text
.ai/codingrules.md                 AI contribution and data rules
data/raw/                          Bundled sample images and raw-data guidance
data/processed/recipes.csv         Curated, normalized 120-recipe catalog
docs/project_proposal.md           Team proposal and timeline
docs/project_design_doc.md         Architecture, data pipeline, and trade-offs
src/app.py                         Streamlit interface
src/cloud_vision.py                Gemini AI ingredient recognition
src/detector.py                    Two-stage local food detector & box visualizer
src/vision_model.py                Local torchvision food-label recognition
src/recipe_engine.py               Staple-aware recipe scoring & ranking
src/nutrition.py                   Serving-aware nutrition estimates
src/utils.py                       Normalization, aliases, pantry staples, & substitutions
requirements.txt                   Pinned Python dependencies
```

## Data and model

The starter recipe catalog is curated project data and is stored in `data/processed/recipes.csv`. `data/raw/README.md` documents the raw-data boundary and optional external data acquisition. No Kaggle dataset or personal image is bundled. If the team later uses a Kaggle recipe dataset, acquire it through Kaggle's official download flow, retain its license and source metadata, and keep the original files in `data/raw/`; write cleaned project-specific data to `data/processed/`.

Ingredia uses Google Gemini for cloud-based ingredient
recognition when an API key is configured.

The application also supports local computer vision using
pretrained Faster R-CNN and MobileNetV3-Small models from
torchvision. These models are not fine-tuned on a custom
food dataset.

If Gemini is unavailable, the local models are used for
ingredient recognition. Recipe recommendations are generated
using the existing ingredient-matching and ranking algorithm,
independently of Gemini.

Uploaded images are sent to Google's Gemini API only when
cloud-based recognition is enabled.

## Rubric-oriented checklist

- **Easy execution:** pinned requirements, Python 3.10/3.11 quickstart, one-command Streamlit launch, and offline sample walkthrough.
- **Modularity:** separate UI, vision, normalization, recipe-ranking, and nutrition modules.
- **Traditional AI:** local pretrained vision classification and explainable ingredient-overlap ranking, with optional Gemini integration for ingredient recognition.
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
