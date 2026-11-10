# FridgeApp: AI Food Recognition and Recipe Suggestions

FridgeApp is a CSCI 270 Applied AI team project at the College of Charleston. It combines production-grade multimodal cloud vision with 2D spatial bounding-box grounding and explainable ingredient-overlap recipe ranking to help users turn the contents of their refrigerator into healthy, customized meals.

## Team

Philip Pagana, Snovi Patel, Jaden Morales, and Amelia West  
College of Charleston · CSCI 270 — Applied AI

---

## Architecture & Features

1. **Multimodal Cloud Vision AI (`src/cloud_vision.py`)**:
   - Leverages Google Gemini multimodal vision endpoints for dense object detection and 2D coordinate grounding (`box_2d`).
   - Automatically draws sharp, color-coded bounding boxes and confidence badges for individual clustered produce items and containers.
   - Built-in multi-model failover (`gemini-3-flash-preview` ➔ `gemini-3.1-flash-lite` ➔ `gemini-flash-latest`), exponential backoff, and in-memory digest caching.
   - Comparative evaluation of Google Cloud Vision, AWS Rekognition, Azure, and MLLMs documented in [`docs/cloud_vision_evaluation.md`](file:///c:/Users/Jaden/FridgeApp/docs/cloud_vision_evaluation.md).

2. **FastAPI Cloud Backend Service (`main.py`)**:
   - High-throughput REST API with payload validation, MIME verification, and max upload limits.
   - Endpoints:
     - `GET /`: Health check and catalog status.
     - `POST /api/detect`: Multimodal food detection with spatial bounding coordinates and categories.
     - `POST /api/analyze-and-match`: End-to-end pipeline returning detected ingredients and ranked recipe recommendations with scaled nutrition.
   - Interactive Swagger API docs available at `http://localhost:8000/docs`.

3. **Streamlit Interactive Frontend (`src/app.py`)**:
   - Allows users to upload real fridge and produce drawer photos or choose offline demo samples.
   - Displays real-time visual grounding overlays and latency statistics.
   - Provides editable ingredient tag pills allowing users to review, add, or delete pantry items.
   - Computes transparent ingredient-coverage ranking (`matched ingredients / total ingredients`).
   - Scales calorie, carbohydrate, protein, and fat estimates relative to daily reference benchmarks.

---

## Quickstart

### 1. Environment Setup
Add your Google Gemini API key to `.env` in the repository root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Launch the Streamlit Frontend
From the repository root:
```powershell
streamlit run src/app.py
```

### 3. Launch the FastAPI Backend Service
```powershell
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to test the API endpoints interactively.

---

## Running Automated Tests

Run the full test suite with regression checks:
```powershell
pytest
```
Or with verbose output:
```powershell
python -m pytest -v
```

---

## Repository Map

```text
.ai/codingrules.md                 AI engineering guidelines
data/raw/                          Bundled sample images and documentation
data/processed/recipes.csv         Curated, normalized starter recipe catalog
docs/cloud_vision_evaluation.md    Comprehensive Cloud Vision API comparison & evaluation
docs/project_proposal.md           Team proposal and timeline
docs/project_design_doc.md         Architecture and data flow specification
main.py                            Production FastAPI backend service
gemini_service.py                  Gemini multimodal service adapter
src/app.py                         Interactive Streamlit frontend interface
src/cloud_vision.py                Cloud multimodal vision, grounding & Pillow annotation
src/vision_model.py                Local fallback image classifier (MobileNetV3)
src/recipe_engine.py               Catalog loading, validation, and overlap ranking
src/nutrition.py                   Serving-aware nutritional estimates
src/utils.py                       Ingredient normalization, aliases, and helpers
tests/test_cloud_vision.py         Tests for Cloud Vision service and FastAPI endpoints
tests/test_fridgeapp.py            Regression tests for recipes, matching, and UI
pytest.ini                         Pytest configuration
requirements.txt                   Pinned Python dependencies
```

## Data and model

The starter recipe catalog is curated project data and is stored in `data/processed/recipes.csv`. `data/raw/README.md` documents the raw-data boundary and optional external data acquisition. No Kaggle dataset or personal image is bundled. If the team later uses a Kaggle recipe dataset, acquire it through Kaggle's official download flow, retain its license and source metadata, and keep the original files in `data/raw/`; write cleaned project-specific data to `data/processed/`.

The classifier uses torchvision's pretrained MobileNetV3-Small weights. It is not fine-tuned on a food dataset; supported ImageNet labels are mapped into a small ingredient vocabulary. A food-specific model or transfer-learning experiment can replace this component without changing the recipe-ranking module. We do not send images to third-party vision APIs.

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
