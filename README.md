<div align="center">
  <img src="assets/ingredia_logo.png" width="140" alt="Ingredia Logo">
  <h1>Ingredia</h1>
  <p><b>Good food starts with what you have.</b></p>
</div>

---

**Ingredia** is an AI-powered recipe recommendation and pantry intelligence application that helps users discover meals based on ingredients they already have at home. Users can upload a refrigerator or pantry photo, review AI-detected items, and explore a rich catalog of categorized recipes complete with estimated calories, interactive macronutrient charts, and nutritional breakdowns.

Developed as a **CSCI 270 Applied AI** team project at the **College of Charleston**.

---

## 👥 Team
* **Philip Pagana**
* **Snovi Patel**
* **Jaden Morales**
* **Amelia West**  
* *College of Charleston · CSCI 270 — Applied AI*

---

## ✨ Key Features & Architecture
1. **Intelligent Computer Vision & Object Detection**:
   - **Google Gemini Cloud Vision**: Automatically recognizes multiple food items and renders bounding boxes with confidence scores.
   - **Two-Stage Local Fallback**: Uses pretrained Faster R-CNN (MobileNetV3-Large FPN) and MobileNetV3-Small (ImageNet) models when offline or unconfigured.
   - **Container Filtering**: Automatically excludes non-food storage containers (bottles, jars, jugs) from pantry inventory.
2. **Dynamic Recipe Synthesis & Catalog**: Matches available inventory against a curated catalog of **500+ recipes** featuring scrollable category filter pills, dynamic pagination, and staple-aware scoring.
3. **Custom UI & Branding**: Styled with a seamless, custom cream background (`#fdfaec`), custom typography (`Fredoka` & `Plus Jakarta Sans`), and the official brand logo.
4. **Nutrition & Macro Tracking**: Interactive Altair donut charts illustrating protein, carbohydrate, and fat distributions alongside benchmark percentages.
5. **Interactive Tools**: Dynamic shopping list exporter and favorited recipe bookmarking.

---

## 🚀 Quickstart

Use **Python 3.10** or **3.11**. From the repository root:

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 3. Launch the Streamlit application
streamlit run src/app.py

```

Open the local URL printed by Streamlit (`http://localhost:8501`). Select one of the bundled preset samples (`Produce drawer`, `Breakfast basics`, `Fruit and yogurt`) or upload your own JPG/PNG image (up to 10 MB).

---

## 🔑 Gemini API Setup (Optional)

Ingredia uses Google Gemini to improve ingredient recognition and identify multiple food items in a photo.

1. Visit [Google AI Studio](https://aistudio.google.com/apikey) to create your API key.
2. Create a `.env` file in the root project folder.
3. Add your key to the file:
```env
GEMINI_API_KEY=your_api_key_here

```


4. Save the file and restart Streamlit. *(The `.env` file is excluded via `.gitignore` for security).* If unconfigured, Ingredia seamlessly falls back to local computer vision models.

---

## 📂 Repository Map

```text
├── .streamlit/                # Streamlit configuration & custom theme styling
├── assets/                    # Official brand logo (ingredia_logo.png)
├── data/
│   ├── raw/                   # Bundled sample images & raw data documentation
│   └── processed/             # Curated, normalized 500-recipe catalog (recipes.csv)
├── docs/                      # Project proposal, design document, and design specs
├── src/
│   ├── app.py                 # Streamlit web application interface
│   ├── cloud_vision.py        # Google Gemini AI ingredient recognition
│   ├── detector.py            # Two-stage local food detector & box visualizer
│   ├── vision_model.py        # Local torchvision food-label recognition
│   ├── recipe_engine.py       # Staple-aware recipe scoring & ranking
│   ├── nutrition.py           # Serving-aware nutrition estimates & macros
│   └── utils.py               # Normalization, culinary aliases, & pantry staples
├── tests/                     # Pytest regression and integration test suite
├── requirements.txt           # Pinned Python dependencies
└── README.md                  # Project documentation

```

---

## 🧪 Run the Regression Tests

To run the test suite locally using `pytest`:

```powershell
python -m pip install pytest
python -m pytest -q tests

```

```

```
