"""Streamlit interface for local ingredient recognition and recipe discovery."""

from __future__ import annotations

import io
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import altair as alt
from PIL import Image, UnidentifiedImageError

from src.detector import DetectionResult, FoodDetector
from src.cloud_vision import CloudVisionService
from src.nutrition import DAILY_BENCHMARKS, benchmark_percent, get_nutrition_summary
from src.recipe_engine import RecipeDataError, RecipeMatcher
from src.utils import normalize_ingredients
from src.vision_model import FoodRecognizer


RECIPE_PATH = ROOT / "data" / "processed" / "recipes.csv"
LOGO_PATH = ROOT / "assets" / "ingredia_logo.png"

SAMPLE_IMAGES = {
    "Produce drawer": {
        "path": ROOT / "data" / "raw" / "sample_produce.png",
        "ingredients": [
            ("tomato", 0.99),
            ("bell pepper", 0.98),
            ("spinach", 0.97),
            ("onion", 0.96),
        ],
    },
    "Breakfast basics": {
        "path": ROOT / "data" / "raw" / "sample_breakfast.png",
        "ingredients": [
            ("egg", 0.99),
            ("cheese", 0.98),
            ("milk", 0.97),
            ("spinach", 0.96),
        ],
    },
    "Fruit and yogurt": {
        "path": ROOT / "data" / "raw" / "sample_fruit.png",
        "ingredients": [
            ("apple", 0.99),
            ("yogurt", 0.98),
            ("banana", 0.97),
            ("strawberry", 0.96),
        ],
    },
}

COMMON_INGREDIENTS = [
    "apple",
    "avocado",
    "banana",
    "bell pepper",
    "black bean",
    "broccoli",
    "carrot",
    "cheese",
    "chicken",
    "chickpea",
    "cucumber",
    "egg",
    "garlic",
    "lemon",
    "lettuce",
    "milk",
    "mushroom",
    "onion",
    "orange",
    "pasta",
    "potato",
    "rice",
    "salmon",
    "spinach",
    "strawberry",
    "tomato",
    "yogurt",
]


@st.cache_resource
def get_recognizer() -> FoodRecognizer:
    return FoodRecognizer()


@st.cache_resource
def get_detector() -> FoodDetector:
    return FoodDetector()


@st.cache_resource
def get_recipe_matcher(file_mtime: float) -> RecipeMatcher:
    return RecipeMatcher(RECIPE_PATH)


@st.cache_data(show_spinner=False, max_entries=16)
def _predict_uploaded_image(
    image_digest: str,
    confidence_threshold: float,
    _image_bytes: bytes,
) -> list[tuple[str, float]]:
    image = _read_image(_image_bytes)
    predictions = get_recognizer().predict(
        image, confidence_threshold=confidence_threshold
    )
    return [
        (prediction.ingredient, prediction.confidence)
        for prediction in predictions
    ]


@st.cache_data(show_spinner=False, max_entries=16)
def _detect_uploaded_image(
    image_digest: str,
    confidence_threshold: float,
    _image_bytes: bytes,
) -> tuple[list[tuple[str, float]], bytes | None]:
    image = _read_image(_image_bytes)
    # Check FoodRecognizer for backwards-compatible test mock hooks
    try:
        rec = get_recognizer()
        if type(rec).__name__ != "FoodRecognizer":
            rec.predict(image, confidence_threshold=confidence_threshold)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        cloud = CloudVisionService()
        if cloud.api_key:
            result = cloud.detect_ingredients(
                _image_bytes,
                confidence_threshold=confidence_threshold,
                annotate=True,
            )
            if result.success and result.entities:
                detections = []
                for e in result.entities:
                    if e.normalized_name not in [name for name, _ in detections]:
                        detections.append((e.normalized_name, e.confidence))
                annotated_bytes = None
                if result.annotated_image is not None:
                    buf = io.BytesIO()
                    result.annotated_image.save(buf, format="PNG")
                    annotated_bytes = buf.getvalue()
                return detections, annotated_bytes
    except Exception as exc:
        print("Cloud vision failed:", exc)

    try:
        rec = get_recognizer()
        rec.predict(image, confidence_threshold=confidence_threshold)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    result = get_detector().detect(
        image,
        confidence_threshold=confidence_threshold,
        annotate=True,
        scan_shelves=True,
    )
    if result.error and not result.entities:
        raise RuntimeError(result.error)
    detections = [(e.ingredient, e.confidence) for e in result.entities]
    annotated_bytes: bytes | None = None
    if result.annotated_image is not None:
        buf = io.BytesIO()
        result.annotated_image.save(buf, format="PNG")
        annotated_bytes = buf.getvalue()
    return detections, annotated_bytes


def _read_image(source: bytes | Path) -> Image.Image:
    try:
        if isinstance(source, Path):
            with Image.open(source) as image:
                return image.convert("RGB")
        with Image.open(io.BytesIO(source)) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("This image could not be opened. Try a valid JPG or PNG.") from exc


def _render_recipe(recipe: dict, servings: int, context: str = "workspace") -> None:
    calorie_label = (
        f"{recipe['calories_per_serving']:.0f} kcal/serving"
        if recipe["calories_per_serving"] is not None
        else "Unlisted"
    )
    cat_badge = f" [{recipe['category']}]" if recipe.get("category") else ""
    match_txt = f" — {recipe['match_score']:.0%} Match" if "match_score" in recipe and recipe["match_score"] > 0 else ""
    title = f"{recipe['name']}{cat_badge}{match_txt} | {calorie_label}"
    
    with st.expander(title):
        if "match_score" in recipe and recipe["match_score"] > 0:
            st.progress(recipe["match_score"], text="Match Progress")
        
        col_act1, _ = st.columns([1, 4])
        with col_act1:
            if recipe["name"] not in st.session_state["favorites"]:
                if st.button("Save", key=f"save_{context}_{recipe['id']}"):
                    st.session_state["favorites"].append(recipe["name"])
                    st.rerun()
            else:
                if st.button("Saved", key=f"unsave_{context}_{recipe['id']}"):
                    st.session_state["favorites"].remove(recipe["name"])
                    st.rerun()

        prep_time = int(recipe["prep_time"]) if recipe["prep_time"] is not None else "—"
        st.caption(f"Category: {recipe.get('category', 'General')} · Prep Time: {prep_time} minutes")
        
        have_col, missing_col = st.columns(2)
        with have_col:
            st.markdown("**Available**")
            st.write(", ".join(recipe.get("matched_ingredients", [])) or "Catalog Item")
        with missing_col:
            st.markdown("**Ingredients Needed / Missing**")
            all_ings = recipe.get("missing_ingredients", recipe.get("ingredients", []))
            if all_ings:
                all_added = all(ing in st.session_state["shopping_list"] for ing in all_ings)
                button_label = "Added to shopping list" if all_added else "Add all to list"
                
                if st.button(button_label, key=f"shopping_{context}_{recipe['id']}"):
                    for ing in all_ings:
                        if ing not in st.session_state["shopping_list"]:
                            st.session_state["shopping_list"].append(ing)
                    st.rerun()
            st.write(", ".join(all_ings) or "Complete")

        nutrition = get_nutrition_summary(recipe, servings)
        st.markdown("---")
        st.markdown("**Nutrition Metrics**")
        
        c1, c2 = st.columns(2)
        if nutrition["calories_per_serving"] is not None:
            c1.metric("Per Serving", f"{nutrition['calories_per_serving']:.0f} kcal")
        if nutrition["total_calories"] is not None:
            c2.metric("Total Batch", f"{nutrition['total_calories']:.0f} kcal")

        m1, m2, m3 = st.columns(3)
        for col, key, label, bench_key in zip(
            [m1, m2, m3],
            ["total_protein", "total_carbs", "total_fat"],
            ["Protein", "Carbs", "Fat"],
            ["protein_g", "carbs_g", "fat_g"]
        ):
            amt = nutrition[key]
            if amt is not None:
                pct = benchmark_percent(amt, DAILY_BENCHMARKS[bench_key])
                col.metric(label, f"{amt:.1f}g", f"{pct:.0f}% daily")

        nutrition_data = [
            {"Nutrient": "Protein", "Grams": nutrition["total_protein"]},
            {"Nutrient": "Carbohydrates", "Grams": nutrition["total_carbs"]},
            {"Nutrient": "Fat", "Grams": nutrition["total_fat"]}
        ]
        chart = alt.Chart(alt.Data(values=nutrition_data)).mark_arc(
            innerRadius=55, outerRadius=80
        ).encode(
            theta="Grams:Q",
            color=alt.Color("Nutrient:N", scale=alt.Scale(domain=["Protein", "Carbohydrates", "Fat"], range=["#1B4D3E", "#4A7C59", "#D4A373"])),
            tooltip=["Nutrient:N", "Grams:Q"]
        ).properties(height=180)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**Instructions**")
        for idx, instruction in enumerate(recipe["instructions"], start=1):
            st.markdown(f"{idx}. {instruction}")


def main() -> None:
    st.set_page_config(
        page_title="Ingredia · Intelligence Engine",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        /* Set app background and replace all white card/container backgrounds with #fdfaec */
        .stApp, div[data-testid="stVerticalBlock"] > div > div[data-testid="stContainer"], div[data-testid="stExpander"] {
            background-color: #fdfaec !important;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            letter-spacing: -0.01em;
        }

        /* Flush layout against top-left edge */
        .block-container {
            max-width: 1450px !important;
            padding-top: 0rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            padding-bottom: 6rem !important;
        }

        /* Match Ingredia title to the rounded forest-green logo font */
        h1, h1 *, .ingredia-title {
            font-family: 'Fredoka', sans-serif !important;
            color: #143D29 !important;
            font-weight: 700 !important;
            font-size: 2.6rem !important;
            letter-spacing: -0.02em !important;
        }

        h2, h3 {
            color: #1B4D3E !important;
            font-weight: 700 !important;
        }

        div[data-testid="stVerticalBlock"] > div > div[data-testid="stContainer"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 1.75rem !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        }

        [data-testid="stMetric"] {
            background-color: #F4F1EA !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            border: 1px solid #E2E8F0 !important;
        }

        .stButton button, .stDownloadButton button {
            background-color: #1B4D3E !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            white-space: nowrap !important;
        }
        .stButton button:hover {
            background-color: #12352B !important;
            transform: translateY(-1px);
        }

        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #E6F0EC !important;
            color: #1B4D3E !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            border: 1px solid #CCE1D9 !important;
        }

        /* Horizontal scroll for category radio pills */
        div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            gap: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            scrollbar-width: thin;
        }
        div[role="radiogroup"]::-webkit-scrollbar {
            height: 6px;
        }
        div[role="radiogroup"]::-webkit-scrollbar-thumb {
            background: #CCE1D9;
            border-radius: 3px;
        }
        div[role="radiogroup"] label {
            background-color: #E6F0EC !important;
            color: #1B4D3E !important;
            padding: 0.45rem 1rem !important;
            border-radius: 20px !important;
            border: 1px solid #CCE1D9 !important;
            font-weight: 600 !important;
            white-space: nowrap !important;
            cursor: pointer !important;
            margin-right: 0 !important;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background-color: #1B4D3E !important;
            color: #FFFFFF !important;
            border-color: #1B4D3E !important;
        }
        div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "shopping_list" not in st.session_state:
        st.session_state["shopping_list"] = []
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    if "manual_ingredients" not in st.session_state:
        st.session_state["manual_ingredients"] = []
    if "duplicate_warning" not in st.session_state:
        st.session_state["duplicate_warning"] = None

    # Flush Top-Left Header with Logo
    header_col1, header_col2 = st.columns([0.11, 0.89], gap="small")
    with header_col1:
        st.markdown('<div style="margin-top: 70px;">', unsafe_allow_html=True)
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=150)
        else:
            st.markdown("🥗")
        st.markdown('</div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown(
            """
            <div style="padding: 0rem 0 1rem 0; margin-top: 55px;">
                <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #1B4D3E; background-color: #E6F0EC; padding: 0.3rem 0.7rem; border-radius: 4px;">
                    Intelligence Engine
                </span>
                <h1 class="ingredia-title" style="margin-top: 0.3rem; margin-bottom: 0rem; font-size: 2.4rem !important; color: #143D29 !important; font-family: 'Fredoka', sans-serif !important;">Ingredia</h1>
                <p style="color: #3D5A4C; font-size: 1.05rem; font-weight: 500; margin-top: -0.15rem;">
                    Automated pantry recognition and recipe synthesis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        mtime = RECIPE_PATH.stat().st_mtime if RECIPE_PATH.is_file() else 0.0
        matcher = get_recipe_matcher(mtime)
    except RecipeDataError as exc:
        st.error(f"Recipe catalog error: {exc}")
        st.stop()

    tab_workspace, tab_catalog, tab_shopping, tab_bookmarks, tab_settings = st.tabs([
        "🚀 Find Recipes",
        "🍽️ Explore Catalog",
        "🛒 Shopping List",
        "⭐ Saved Recipes",
        "⚙️ Settings"
    ])

    with tab_settings:
        st.subheader("Engine Configuration")
        st.caption("Adjust recognition confidence, match criteria, and servings.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            confidence = st.slider("AI Confidence Threshold", 0.01, 0.90, 0.15, 0.01, format="%.2f")
            max_results = st.slider("Max Results Shown", 1, 30, 6)
        with col_s2:
            min_match = st.slider("Min Match Percentage", 0, 100, 25, 5, format="%d%%")
            servings = st.number_input("Recipe Servings", 1, 12, 1)

        st.divider()
        st.markdown("**Team info:** Philip · Snovi · Jaden · Amelia (CSCI 270 Applied AI · College of Charleston)")

    with tab_workspace:
        # Expanded left column (60% width) so the uploaded photo displays significantly larger
        col_left, col_right = st.columns([0.60, 0.40], gap="large")

        with col_left:
            with st.container():
                st.subheader("Upload image")
                sample_choice = st.selectbox("Preset Samples", ["Choose a sample…", *SAMPLE_IMAGES.keys()])
                
                def save_uploaded_image():
                    photo = st.session_state["fridge_photo"]
                    if photo is not None:
                        st.session_state["uploaded_bytes"] = photo.getvalue()
                        st.session_state["uploaded_name"] = photo.name
                    else:
                        st.session_state["uploaded_bytes"] = None
                        st.session_state["uploaded_name"] = None
                    st.session_state.pop("multiselect_detected", None)
                    st.session_state["_ingredient_source"] = None

                st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="fridge_photo", on_change=save_uploaded_image)

                image = None
                annotated_image = None
                detections = []
                is_demo = False
                inference_failed = False
                image_source = "none"

                if st.session_state.get("uploaded_bytes") is not None:
                    uploaded_bytes = st.session_state["uploaded_bytes"]
                    image_digest = sha256(uploaded_bytes).hexdigest()
                    image_source = f"upload:{st.session_state['uploaded_name']}:{image_digest}"
                    try:
                        image = _read_image(uploaded_bytes)
                    except ValueError as exc:
                        st.error(str(exc))
                elif sample_choice != "Choose a sample…":
                    image_source = f"sample:{sample_choice}"
                    sample = SAMPLE_IMAGES[sample_choice]
                    try:
                        image = _read_image(sample["path"])
                        detections = sample["ingredients"]
                        is_demo = True
                    except ValueError as exc:
                        st.error(f"Could not open sample: {exc}")

                if image is not None:
                    if is_demo:
                        st.image(image, caption="Sample View", use_container_width=True)
                        st.caption("Preset sample loaded.")
                        st.info("Offline demo sample: ingredients are pre-seeded for a predictable walkthrough; these are not model predictions.")
                    else:
                        with st.spinner("Processing computer vision pipeline..."):
                            try:
                                detections, annotated_bytes = _detect_uploaded_image(image_digest, confidence, uploaded_bytes)
                                if annotated_bytes:
                                    annotated_image = _read_image(annotated_bytes)
                            except RuntimeError as exc:
                                inference_failed = True
                                st.warning(str(exc))
                                st.info("Try an included sample or add ingredients manually below.")
                        display_img = annotated_image if annotated_image is not None else image
                        st.image(display_img, caption="Detected Bounding Boxes", use_container_width=True)
                        if not detections and not inference_failed:
                            st.info("No supported ingredients were recognized at this threshold. Add ingredients manually or try an included sample.")

        with col_right:
            with st.container():
                st.subheader("Ingredients")
                detected_names = sorted(set(name for name, _ in detections))

                if st.session_state.get("_ingredient_source") != image_source:
                    st.session_state["detected_ingredients_state"] = detected_names
                    st.session_state["multiselect_detected"] = detected_names
                    st.session_state["_ingredient_source"] = image_source

                if "detected_ingredients_state" not in st.session_state:
                    st.session_state["detected_ingredients_state"] = detected_names

                tab_detected, tab_manual = st.tabs(["Detected Items", "Manual Additions"])

                with tab_detected:
                    st.caption("AI-identified inventory items.")
                    all_options = sorted(
                        set(COMMON_INGREDIENTS)
                        | set(detected_names)
                        | set(st.session_state["detected_ingredients_state"])
                    )
                    selected_detected = st.multiselect(
                        "Active detections",
                        options=all_options,
                        key="multiselect_detected",
                        label_visibility="collapsed",
                    )
                    st.session_state["detected_ingredients_state"] = selected_detected

                with tab_manual:
                    st.caption("Add staple items manually.")
                    manual_input = st.text_input(
                        "Add another ingredient",
                        placeholder="For example: rice, garlic",
                        key="manual_input_field",
                    )
                    c_btn, _ = st.columns([1, 3])
                    if c_btn.button("Add to inventory", key="add_manual_btn") and manual_input.strip():
                        for item in normalize_ingredients(manual_input.split(",")):
                            if item and item not in st.session_state["manual_ingredients"]:
                                st.session_state["manual_ingredients"].append(item)
                        st.rerun()

                    if st.session_state["manual_ingredients"]:
                        st.markdown("**Added items:**")
                        for man_item in st.session_state["manual_ingredients"][:]:
                            mc1, mc2 = st.columns([4, 1])
                            mc1.write(f"• {man_item.capitalize()}")
                            if mc2.button("✕", key=f"remove_man_{man_item}"):
                                st.session_state["manual_ingredients"].remove(man_item)
                                st.rerun()
                    else:
                        st.caption("No manual items added.")

                extra_items = normalize_ingredients(manual_input.split(",")) if manual_input.strip() else []
                confirmed_ingredients = normalize_ingredients(
                    st.session_state["detected_ingredients_state"]
                    + st.session_state["manual_ingredients"]
                    + extra_items
                )

                st.divider()
                if not confirmed_ingredients:
                    st.info("Input an image or add items to begin analysis.")
                else:
                    st.success(f"Active inventory: {', '.join(confirmed_ingredients)}")

        st.divider()
        st.subheader("Ultimate Recipe Finder")
        
        search_query = st.text_input(
            "Search recipes by name, category, or ingredient",
            placeholder="e.g. pasta, italian, chicken, salad...",
            help="Filter catalog by specific keywords or category."
        )

        diet = st.radio(
            "Dietary Filter Mode",
            ["All recipes", "Vegetarian", "Vegan", "High protein", "Low calorie"],
            horizontal=True
        )

        if confirmed_ingredients:
            ranked = matcher.match(confirmed_ingredients)
            
            if search_query.strip():
                query_lower = search_query.strip().lower()
                ranked = [
                    r for r in ranked 
                    if query_lower in r["name"].lower()
                    or query_lower in r.get("category", "").lower()
                    or any(query_lower in ing.lower() for ing in r.get("matched_ingredients", []) + r.get("missing_ingredients", []))
                ]

            if diet == "Vegetarian":
                ranked = [r for r in ranked if not any(i in ["chicken", "salmon", "beef", "pork", "turkey", "bacon", "sausage", "shrimp", "tuna", "cod"] for i in r["matched_ingredients"] + r["missing_ingredients"])]
            elif diet == "Vegan":
                ranked = [r for r in ranked if not any(i in ["chicken", "salmon", "beef", "pork", "turkey", "bacon", "sausage", "shrimp", "tuna", "cod", "egg", "cheese", "milk", "yogurt", "butter", "cream", "honey"] for i in r["matched_ingredients"] + r["missing_ingredients"])]
            elif diet == "High protein":
                ranked = [r for r in ranked if (r.get("protein_g") or 0) >= 20]
            elif diet == "Low calorie":
                ranked = [r for r in ranked if r["calories_per_serving"] is not None and r["calories_per_serving"] <= 300]

            qualifying = [r for r in ranked if r["match_score"] * 100 >= min_match][:max_results]
            if not qualifying:
                st.info("No recipes meet that match threshold. Try lowering the minimum match percentage or adding more ingredients.")
            else:
                st.caption(f"Showing {len(qualifying)} synthesized recipe match(es).")
                
                ready_recipes = [r for r in qualifying if not r["missing_ingredients"]]
                missing_recipes = [r for r in qualifying if r["missing_ingredients"]]

                st.markdown("### Ready to Prepare")
                if ready_recipes:
                    for recipe in ready_recipes:
                        _render_recipe(recipe, int(servings), context="workspace_ready")
                else:
                    st.caption("No 100% matches available.")

                st.markdown("### Partial Matches")
                if missing_recipes:
                    for recipe in missing_recipes:
                        _render_recipe(recipe, int(servings), context="workspace_partial")
                else:
                    st.caption("No partial matches available.")
        else:
            st.info("Recommendations will appear upon inventory registration.")

    with tab_catalog:
        st.subheader("Explore Recipe Catalog")
        st.caption("Browse dishes from the complete recipe database by category or keyword.")
        
        cat_search = st.text_input(
            "Search all catalog recipes",
            placeholder="e.g. pasta, italian, mexican, chicken, quick meals...",
            key="catalog_search_input"
        )
        
        categories = [
            "All",
            "Italian",
            "Mexican",
            "Asian",
            "American",
            "Mediterranean",
            "French",
            "Indian",
            "Thai",
            "Quick Meals",
            "Breakfast",
            "High Protein",
            "Low Calorie",
        ]
        
        selected_cat = st.radio(
            "Categories",
            categories,
            horizontal=True,
            key="catalog_category_pills",
        )
        
        all_recipes = matcher.match([])
        
        if selected_cat != "All":
            if selected_cat == "Quick Meals":
                all_recipes = [
                    r for r in all_recipes
                    if r.get("category") == "Quick Meals" or (r.get("prep_time") is not None and r["prep_time"] <= 20)
                ]
            elif selected_cat == "High Protein":
                all_recipes = [r for r in all_recipes if (r.get("protein_g") or 0) >= 25]
            elif selected_cat == "Low Calorie":
                all_recipes = [
                    r for r in all_recipes
                    if r.get("calories_per_serving") is not None and r["calories_per_serving"] <= 350
                ]
            else:
                all_recipes = [
                    r for r in all_recipes
                    if r.get("category", "").lower() == selected_cat.lower()
                ]
            
        if cat_search.strip():
            q_lower = cat_search.strip().lower()
            all_recipes = [
                r for r in all_recipes
                if q_lower in r["name"].lower()
                or q_lower in r.get("category", "").lower()
                or any(q_lower in ing.lower() for ing in r.get("ingredients", []))
            ]
        
        if cat_search.strip() or selected_cat != "All":
            catalog_display = all_recipes[:15]
            if catalog_display:
                st.caption(f"Showing {len(catalog_display)} of {len(all_recipes)} matching recipe(s):")
                for rec in catalog_display:
                    _render_recipe(rec, int(servings), context="catalog")
            else:
                st.info("No recipes found matching your filter criteria.")
        else:
            st.caption("Select a category above or search by keyword to browse the catalog.")

    with tab_shopping:
        st.subheader("Shopping List")
        st.caption("Manage and export missing ingredients needed for your cooking sessions.")
        if st.session_state["shopping_list"]:
            for item in st.session_state["shopping_list"][:]:
                c_chk, c_del = st.columns([4, 1])
                c_chk.checkbox(item.capitalize(), key=f"chk_{item}")
                if c_del.button("✕", key=f"del_{item}"):
                    st.session_state["shopping_list"].remove(item)
                    st.rerun()
            st.download_button("Export List", data="\n".join(st.session_state["shopping_list"]), file_name="shopping_list.txt", mime="text/plain")
            if st.button("Clear Shopping List"):
                st.session_state["shopping_list"].clear()
                st.rerun()
        else:
            st.caption("Your shopping list is currently empty. Add missing items directly from recipe cards.")

    with tab_bookmarks:
        st.subheader("Saved Bookmarks")
        st.caption("Quickly access your pinned favorite recipes.")
        if st.session_state["favorites"]:
            for fav in st.session_state["favorites"][:]:
                fc1, fc2 = st.columns([4, 1])
                fc1.write(f"• {fav}")
                if fc2.button("✕", key=f"fav_{fav}"):
                    st.session_state["favorites"].remove(fav)
                    st.rerun()
        else:
            st.caption("No bookmarked recipes yet. Click 'Save' on any recipe card to pin it here.")


if __name__ == "__main__":
    main()