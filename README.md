# Ingredia

<div align="center">
  <img src="assets/ingredia_logo.png" width="140" alt="Ingredia Logo">
  <p><strong>Good food starts with what you have.</strong></p>
</div>

**Ingredia** is an AI-powered recipe recommendation and pantry intelligence application that helps users discover meals using ingredients they already have at home.

Users can upload photos of food or refrigerator contents, review AI-detected ingredients, explore recipe recommendations, view estimated nutrition information, save favorite recipes, and organize shopping lists.

Developed for **CSCI 270 — Applied AI** at the **College of Charleston**.

---

## Team Members

- **Snovi Patel**
- **Jaden Morales**
- **Philip Pagana**
- **Amelia West**

## Key Features

### 1. AI-Powered Ingredient Recognition

Ingredia supports two image-recognition approaches:

**Google Gemini Cloud Vision**
- Uses Gemini to identify ingredients in uploaded images when an API key is configured.
- Supports identifying multiple ingredients from a single image.
- Can provide annotated detection results.

**Local Computer Vision Fallback**
- Uses pretrained Faster R-CNN with MobileNetV3-Large FPN and MobileNetV3-Small models.
- Supports ingredient classification and object detection.
- Allows image recognition without a Gemini API key.
- Uses confidence thresholds and ingredient normalization to process predictions.

Recognition accuracy depends on the image and the models being used.

### 2. Recipe Recommendations

Ingredia matches detected or selected ingredients against a recipe catalog.

Features include:
- Ingredient-based recipe matching.
- Ingredient normalization and culinary substitutions.
- Pantry-staple-aware recipe ranking.
- Recipe instructions and preparation times.
- Recipe browsing by category and keyword.
- Recipe recommendations based on available ingredients.

**Current dataset:** 120 recipes in `data/processed/recipes.csv`.

The catalog can be expanded by adding valid recipe records while preserving the required CSV structure.

### 3. Nutrition Information

Ingredia provides estimated nutritional information for recipes, including:

- Calories per serving.
- Protein.
- Carbohydrates.
- Fat.
- Interactive nutrition visualizations.
- Serving-size adjustments.

Nutrition values are estimates and should not be treated as verified dietary measurements.

### 4. Interactive Application Features

Ingredia includes a multipage Streamlit interface with:

- **My Fridge:** Upload food images and receive ingredient-based recommendations.
- **Explore Recipe Catalog:** Browse recipes using category filters and keyword search.
- **Saved Recipes:** Access bookmarked recipes.
- **Shopping List:** Organize ingredients needed for meals.
- **Settings:** Adjust detection confidence, recipe matching preferences, and serving sizes.

### 5. Custom Interface and Branding

The application includes:
- Ingredia branding and logo.
- Custom Streamlit styling.
- Cream-colored interface.
- Recipe cards and navigation.
- Interactive controls and visualizations.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Cloud ingredient recognition | Google Gemini |
| Local image classification | PyTorch, Torchvision, MobileNetV3-Small |
| Local object detection | Faster R-CNN with MobileNetV3-Large FPN |
| Recipe recommendation | Python-based ingredient matching |
| Recipe storage | CSV |
| Data processing | Pandas, NumPy |
| Nutrition visualization | Altair |
| Model training | PyTorch and optional Hugging Face datasets |
| Testing | Pytest |

---

## Installation and Setup

### Requirements

- Python 3.10 or 3.11 recommended.
- Internet access for initial dependency and pretrained-model downloads.
- A Google Gemini API key is optional.

### 1. Clone the Repository

```bash
git clone https://github.com/moralesjs-hub/FridgeApp.git
cd FridgeApp
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, consult the official PyTorch installation instructions for your operating system.

### 4. Run Ingredia

```bash
python -m streamlit run src/app.py
```

Open the local URL shown in the terminal, typically:

`http://localhost:8501`

You can upload a JPG or PNG image or use the sample images provided by the application.

---

## Optional Gemini API Configuration

Ingredia supports Gemini-based ingredient recognition.

### Setup

1. Visit [Google AI Studio](https://aistudio.google.com/apikey).
2. Generate a Gemini API key.
3. Create a `.env` file in the repository root.
4. Add the following:

```env
GEMINI_API_KEY=your_api_key_here
```

5. Save the file and restart Streamlit.

**Security:** Do not commit your actual API key to GitHub. The `.env` file is excluded through `.gitignore`.

### Recognition Fallback

When Gemini is unavailable, not configured, or does not return usable detections, the application can use its local computer vision pipeline.

Uploaded images may be sent to Google's Gemini service when cloud recognition is enabled. Users should avoid uploading sensitive images.

---

## AI Model Architecture

Ingredia combines cloud-based recognition with local computer vision.

### Image Recognition Workflow

1. The user uploads an image.
2. The application attempts Gemini-based ingredient recognition when configured.
3. If cloud recognition is unavailable or unsuccessful, the local recognition pipeline is used.
4. Detected ingredient names are normalized.
5. Ingredients are matched against the recipe catalog.
6. The application displays recommendations, recipe information, and estimated nutrition.

### Local Models

The local pipeline uses:

- **MobileNetV3-Small:** Image classification and ingredient-label mapping.
- **Faster R-CNN with MobileNetV3-Large FPN:** Object detection and bounding-box predictions.
- Additional ingredient filtering and normalization logic.

The application uses pretrained model weights when compatible fine-tuned checkpoints are unavailable.

---

## Optional Model Fine-Tuning

The repository includes training code contributed for ingredient classification and object detection.

Training is optional. The application can run using pretrained models without completing custom training.

### Training Dataset

Training data can be provided through a JSON manifest:

`data/raw/annotations.json`

An example format is available at:

`data/raw/annotations.example.json`

Training records can include:
- Image paths.
- Ingredient labels.
- Optional normalized bounding boxes.

The training system also supports a Hugging Face ingredient-image dataset.

### Train an Ingredient Classifier

Using a local manifest:

```bash
python -m train --manifest data/raw/annotations.json --task classifier --epochs 5
```

Using the supported Hugging Face dataset:

```bash
python -m pip install datasets
python -m train --hf-dataset Scuccorese/food-ingredients-dataset --task classifier --epochs 5
```

### Train an Object Detector

For annotated datasets containing bounding boxes:

```bash
python -m train --manifest data/raw/annotations.json --task detector --epochs 5
```

### Model Checkpoints

Training can produce:

- `models/ingredient_classifier.pt`
- `models/food_detector.pt`

The application supports loading compatible fine-tuned checkpoints when they are present.

**Note:** The repository includes the training implementation, but custom-trained checkpoint files are not bundled in the reviewed version. The default application uses pretrained models.

For complete training instructions, see [`train/README.md`](train/README.md) and [`models/README.md`](models/README.md).

---

## Recipe Dataset

The recipe catalog is stored in:

`data/processed/recipes.csv`

The current reviewed version contains **120 recipes**.

The dataset includes fields for:
- Recipe ID.
- Recipe name.
- Ingredients.
- Calories per serving.
- Preparation time.
- Cooking instructions.
- Protein.
- Carbohydrates.
- Fat.

The recipe engine validates required fields, normalizes ingredient names, and derives recipe categories for browsing and matching.

The catalog may be expanded without changing the overall application architecture, provided new records follow the expected schema.

---

## Repository Structure

```text
FridgeApp/
├── .ai/                     # Project coding guidance
├── .streamlit/              # Streamlit theme and configuration
├── assets/                  # Ingredia branding and images
├── data/
│   ├── raw/                 # Sample images and dataset documentation
│   └── processed/
│       └── recipes.csv      # Recipe catalog
├── docs/                    # Project proposal and design documents
├── models/                  # Optional trained model checkpoints
├── scratch/                 # Development utilities
├── src/
│   ├── app.py               # Streamlit application entry point
│   ├── backend.py           # Shared UI and application logic
│   ├── cloud_vision.py      # Gemini ingredient recognition
│   ├── detector.py          # Local object detection
│   ├── ingredient_classifier.py
│   ├── vision_model.py      # Local image recognition
│   ├── recipe_engine.py     # Recipe loading and matching
│   ├── validate_recipes.py  # Recipe dataset validation & quality checks
│   ├── nutrition.py         # Nutrition calculations
│   ├── utils.py             # Ingredient normalization
│   └── views/
│       ├── find_recipes.py
│       ├── explore_catalog.py
│       ├── saved_recipes.py
│       ├── shopping_list.py
│       └── settings.py
├── tests/                   # Automated tests
├── train/                   # Model training pipeline
├── requirements.txt         # Python dependencies
└── README.md                # Main project documentation
```

---

## Testing

The repository includes automated tests for application components, recipe matching, image detection, and training functionality.

Run the tests from the repository root:

```bash
python -m pytest -q tests
```

For manual testing:

1. Launch the Streamlit application.
2. Upload a food image.
3. Review detected ingredients.
4. Check recipe recommendations and nutrition information.
5. Browse the recipe catalog.
6. Save a recipe.
7. Check the shopping list.
8. Verify settings and navigation.
9. Test Gemini recognition when configured.
10. Verify local recognition without Gemini.

The presence of automated tests does not guarantee that every feature has passed testing in every environment.

---

## Project Documentation

Additional project materials are available in:

- `docs/` — project proposal, design documentation, and supporting specifications.
- `train/README.md` — model training instructions.
- `models/README.md` — checkpoint information.
- `data/raw/README.md` — dataset information.

---

## Limitations and Future Improvements

Current limitations include:

- Ingredient recognition accuracy varies with image quality, lighting, and overlapping objects.
- Pretrained models may misidentify unfamiliar ingredients.
- Recipe recommendations depend on the available catalog.
- Nutritional information is estimated.
- Fine-tuned model performance depends on the quality and size of the training dataset.
- Gemini recognition requires an API key and internet connectivity.

Potential improvements include expanding the recipe catalog, improving ingredient detection with additional labeled images, and refining recommendation filters.

---

**Ingredia — Good food starts with what you have.**

*CSCI 270 Applied AI | College of Charleston*
