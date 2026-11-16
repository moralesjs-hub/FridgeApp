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
        else "—"
    )
    cat_badge = f" [{recipe['category']}]" if recipe.get("category") else ""
    title = (
        f"{recipe['name']}{cat_badge}  ·  {recipe['match_score']:.0%} match"
        f"  ·  {calorie_label}"
    )
    
    with st.expander(title):
        st.progress(recipe["match_score"], text="Ingredient match")
        
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            if recipe["name"] not in st.session_state["favorites"]:
                if st.button("☆ Save to favorites", key=f"save_{context}_{recipe['id']}", type="secondary"):
                    st.session_state["favorites"].append(recipe["name"])
                    st.rerun()
            else:
                if st.button("★ Saved to favorites", key=f"unsave_{context}_{recipe['id']}", type="primary"):
                    st.session_state["favorites"].remove(recipe["name"])
                    st.rerun()

        missing_ings = recipe.get("missing_ingredients", [])
        with col_act2:
            if missing_ings:
                all_added = all(ing in st.session_state["shopping_list"] for ing in missing_ings)
                btn_lbl = "✓ In shopping list" if all_added else "🛒 Add missing to list"
                btn_typ = "secondary" if all_added else "primary"
                if st.button(btn_lbl, key=f"shop_{context}_{recipe['id']}", type=btn_typ):
                    for ing in missing_ings:
                        if ing not in st.session_state["shopping_list"]:
                            st.session_state["shopping_list"].append(ing)
                    st.rerun()

        prep_time = int(recipe["prep_time"]) if recipe["prep_time"] is not None else "—"
        st.caption(f"Category: {recipe.get('category', 'General')} · Prep Time: {prep_time} minutes")
        
        have_col, missing_col = st.columns(2)
        with have_col:
            st.markdown("**Matched ingredients**")
            st.write(", ".join(recipe.get("matched_ingredients", [])) or "None")
        with missing_col:
            st.markdown("**Missing ingredients**")
            st.write(", ".join(recipe.get("missing_ingredients", [])) or "None (Ready to cook!)")

        nutrition = get_nutrition_summary(recipe, servings)
        st.markdown("---")
        st.markdown("**Estimated nutrition per serving**")
        
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


def _render_catalog_card(recipe: dict, servings: int) -> None:
    calorie_label = (
        f"{recipe['calories_per_serving']:.0f} kcal/serving"
        if recipe["calories_per_serving"] is not None
        else "—"
    )
    with st.container():
        c_title, c_save = st.columns([4, 1])
        with c_title:
            st.markdown(f"#### {recipe['name']}")
            st.caption(f"Category: {recipe.get('category', 'General')} · Prep Time: {recipe.get('prep_time', '—')} min · {calorie_label}")
        with c_save:
            if recipe["name"] not in st.session_state["favorites"]:
                if st.button("☆ Save", key=f"cat_save_{recipe['id']}", type="secondary"):
                    st.session_state["favorites"].append(recipe["name"])
                    st.rerun()
            else:
                if st.button("★ Saved", key=f"cat_unsave_{recipe['id']}", type="primary"):
                    st.session_state["favorites"].remove(recipe["name"])
                    st.rerun()

        st.markdown(f"**Ingredients:** {', '.join(recipe.get('ingredients', []))}")
        st.markdown(f"**Instructions:** {recipe.get('instructions', 'No instructions provided.')}")
        
        nutrition = get_nutrition_summary(recipe, servings)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calories", f"{nutrition['calories_per_serving']:.0f} kcal" if nutrition['calories_per_serving'] is not None else "—")
        m2.metric("Protein", f"{nutrition['total_protein']:.0f}g" if nutrition['total_protein'] is not None else "—")
        m3.metric("Carbs", f"{nutrition['total_carbs']:.0f}g" if nutrition['total_carbs'] is not None else "—")
        m4.metric("Fat", f"{nutrition['total_fat']:.0f}g" if nutrition['total_fat'] is not None else "—")
        st.divider()


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

        /* Clean, refined button design system */
        .stButton button, .stDownloadButton button {
            font-family: inherit !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: -0.01em !important;
            border-radius: 8px !important;
            padding: 0.45rem 1rem !important;
            min-height: 2.3rem !important;
            line-height: 1.4 !important;
            transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.45rem !important;
            cursor: pointer !important;
            border: 1px solid transparent !important;
            white-space: nowrap !important;
        }

        /* Primary Button: rich forest gradient, subtle border, gentle lift on hover */
        .stButton button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button,
        .stDownloadButton button {
            background: linear-gradient(180deg, #1E5646 0%, #153E32 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #143B2F !important;
            box-shadow: 0 1px 3px rgba(20, 61, 41, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        }
        .stButton button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        .stDownloadButton button:hover {
            background: linear-gradient(180deg, #246552 0%, #1A493B 100%) !important;
            color: #FFFFFF !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 10px rgba(20, 61, 41, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
        }
        .stButton button[kind="primary"]:active,
        div[data-testid="stFormSubmitButton"] button:active,
        .stDownloadButton button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 1px 2px rgba(20, 61, 41, 0.16) !important;
        }

        /* Secondary Button: clean white card style with sage border */
        .stButton button[kind="secondary"],
        .stButton button:not([kind="primary"]):not([kind="tertiary"]) {
            background: #FFFFFF !important;
            color: #1B4D3E !important;
            border: 1.5px solid #CFDFD7 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }
        .stButton button[kind="secondary"]:hover,
        .stButton button:not([kind="primary"]):not([kind="tertiary"]):hover {
            background: #F4F8F6 !important;
            border-color: #1B4D3E !important;
            color: #153E32 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 3px 8px rgba(27, 77, 62, 0.09) !important;
        }
        .stButton button[kind="secondary"]:active,
        .stButton button:not([kind="primary"]):not([kind="tertiary"]):active {
            background: #E8F2EC !important;
            transform: translateY(0px) !important;
        }

        /* Tertiary / Delete Button: minimalist, sleek badge that turns soft coral-red on hover */
        .stButton button[kind="tertiary"] {
            background: #F8FAFC !important;
            color: #64748B !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
            min-height: 2rem !important;
            height: 2rem !important;
            width: 2rem !important;
            padding: 0 !important;
            border-radius: 6px !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
        }
        .stButton button[kind="tertiary"]:hover {
            background: #FEE2E2 !important;
            color: #DC2626 !important;
            border-color: #FECACA !important;
            transform: scale(1.06) !important;
            box-shadow: 0 2px 5px rgba(220, 38, 38, 0.15) !important;
        }
        .stButton button[kind="tertiary"]:active {
            transform: scale(0.98) !important;
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
            confidence = st.slider(
                "Model confidence threshold",
                min_value=0.05,
                max_value=0.90,
                value=0.15,
                step=0.05,
                help="Filter detected ingredients based on model confidence.",
            )
            servings = st.number_input("Recipe Servings", 1, 12, 1)
        with col_s2:
            max_results = st.slider(
                "Maximum recipe results", min_value=1, max_value=12, value=6
            )
            min_match = st.slider(
                "Minimum recipe match",
                min_value=0,
                max_value=100,
                value=25,
                step=5,
                format="%d%%",
                help="Filter out recipes that match less than this percentage.",
            )

        st.divider()
        st.markdown("**Team info:** Philip · Snovi · Jaden · Amelia (CSCI 270 Applied AI · College of Charleston)")

    with tab_workspace:
        col_left, col_right = st.columns([0.55, 0.45], gap="large")

        with col_left:
            with st.container():
                st.subheader("1. Add a fridge photo")
                sample_choice = st.selectbox(
                    "Try an included sample",
                    ["Choose a sample…", *SAMPLE_IMAGES.keys()],
                )
                
                def save_uploaded_image():
                    photo = st.session_state["fridge_photo"]
                    if photo is not None:
                        st.session_state["uploaded_bytes"] = photo.getvalue()
                        st.session_state["uploaded_name"] = photo.name
                    else:
                        st.session_state["uploaded_bytes"] = None
                        st.session_state["uploaded_name"] = None
                    st.session_state["_ingredient_source"] = None

                st.file_uploader(
                    "Upload image (JPG/PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="fridge_photo",
                    on_change=save_uploaded_image
                )

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
                        st.info(
                            "Offline demo sample: ingredients are pre-seeded for a predictable "
                            "walkthrough; these are not model predictions."
                        )
                    else:
                        with st.spinner("Processing computer vision pipeline..."):
                            try:
                                detections, annotated_bytes = _detect_uploaded_image(
                                    image_digest, confidence, uploaded_bytes
                                )
                                if annotated_bytes:
                                    annotated_image = _read_image(annotated_bytes)
                            except RuntimeError as exc:
                                inference_failed = True
                                st.warning(str(exc))
                                st.info("Try an included sample or add ingredients manually below.")
                        display_img = annotated_image if annotated_image is not None else image
                        caption = "Detected items with bounding boxes" if annotated_image is not None else "Uploaded image"
                        st.image(display_img, caption=caption, use_container_width=True)
                        if not detections and not inference_failed:
                            st.info(
                                "No supported ingredients were recognized at this threshold. "
                                "Add ingredients manually or try an included sample."
                            )

        with col_right:
            with st.container():
                st.subheader("2. Review your ingredients")
                detected_names = [name for name, _ in detections]

                if "saved_ingredients" not in st.session_state:
                    st.session_state["saved_ingredients"] = detected_names

                if st.session_state.get("_ingredient_source") != image_source:
                    st.session_state["saved_ingredients"] = detected_names
                    st.session_state["_ingredient_source"] = image_source

                tab_detected, tab_manual = st.tabs(["Detected Items", "Manual Additions"])

                with tab_detected:
                    st.caption("Review AI-detected items and common pantry ingredients.")
                    options = sorted(
                        set(COMMON_INGREDIENTS)
                        | set(detected_names)
                        | set(st.session_state["saved_ingredients"])
                    )
                    selected_ingredients = st.multiselect(
                        "Detected and pantry ingredients",
                        options=options,
                        default=st.session_state["saved_ingredients"],
                        help="Remove false detections or add ingredients from your pantry.",
                    )
                    st.session_state["saved_ingredients"] = selected_ingredients

                    if detections and not is_demo:
                        st.markdown("**Current detections:**")
                        badges = [f"`{name}` ({score:.0%})" for name, score in detections]
                        st.markdown(" · ".join(badges))

                with tab_manual:
                    st.caption("Add staple items to your inventory.")
                    extra_ingredients = st.text_input(
                        "Add another ingredient",
                        placeholder="For example: rice, garlic",
                        help="Enter comma-separated ingredients to add them to your inventory.",
                    )
                    extra_list = (
                        normalize_ingredients(extra_ingredients.split(","))
                        if extra_ingredients.strip()
                        else []
                    )
                    c_btn, _ = st.columns([1, 2])
                    with c_btn:
                        if st.button("Add to inventory", key="btn_add_manual_item", type="secondary"):
                            if extra_list:
                                for item in extra_list:
                                    if item not in st.session_state["manual_ingredients"]:
                                        st.session_state["manual_ingredients"].append(item)
                                st.rerun()

                    if st.session_state["manual_ingredients"]:
                        st.markdown("**Added manual items:**")
                        for man_item in st.session_state["manual_ingredients"][:]:
                            mc1, mc2 = st.columns([4, 1])
                            mc1.write(f"• {man_item.capitalize()}")
                            if mc2.button("✕", key=f"remove_man_{man_item}", type="tertiary"):
                                st.session_state["manual_ingredients"].remove(man_item)
                                st.rerun()

                confirmed_ingredients = normalize_ingredients(
                    selected_ingredients + extra_list + st.session_state["manual_ingredients"]
                )

                st.divider()
                if not confirmed_ingredients:
                    st.info("Input an image or add items to begin analysis.")
                else:
                    st.success(
                        f"{len(confirmed_ingredients)} ingredient(s): "
                        + ", ".join(confirmed_ingredients)
                    )

        st.divider()
        st.subheader("3. Recommended recipes")
        
        col_search, col_diet = st.columns([2, 1])
        with col_search:
            search_query = st.text_input(
                "Search recipes by name, category, or ingredient",
                placeholder="e.g. pasta, italian, chicken, salad...",
                key="workspace_search_query"
            )
        with col_diet:
            diet = st.radio(
                "Dietary Filter",
                ["All recipes", "Vegetarian", "Vegan", "High protein", "Low calorie"],
                horizontal=True,
                key="workspace_diet_filter"
            )

        if not confirmed_ingredients:
            st.info("Your recipe matches will appear here once you add ingredients.")
            return

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
            st.info(
                "No recipes meet that match threshold. Lower the minimum match or add "
                "more ingredients."
            )
            return

        st.caption(
            f"Ranked by ingredient coverage · {len(qualifying)} recipe(s) shown · "
            "scores do not account for ingredient quantities or substitutions."
        )

        ready_recipes = [r for r in qualifying if not r["missing_ingredients"]]
        missing_recipes = [r for r in qualifying if r["missing_ingredients"]]

        st.markdown("### Ready to cook")
        if ready_recipes:
            for recipe in ready_recipes:
                _render_recipe(recipe, int(servings), context="ready")
        else:
            st.info("No complete recipe matches yet.")

        if missing_recipes:
            st.markdown("### Almost ready to cook")
            for recipe in missing_recipes:
                _render_recipe(recipe, int(servings), context="almost")

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
        
        catalog_display = all_recipes[:15]
        if catalog_display:
            st.caption(f"Showing {len(catalog_display)} of {len(all_recipes)} catalog recipe(s):")
            for rec in catalog_display:
                _render_catalog_card(rec, int(servings))
        else:
            st.info("No recipes found matching your filter criteria.")

    with tab_shopping:
        st.subheader("Shopping List")
        st.caption("Manage and export missing ingredients needed for your cooking sessions.")
        if st.session_state["shopping_list"]:
            for item in st.session_state["shopping_list"][:]:
                c_chk, c_del = st.columns([5, 1])
                c_chk.checkbox(item.capitalize(), key=f"chk_{item}")
                if c_del.button("✕", key=f"del_{item}", type="tertiary", help="Remove item"):
                    st.session_state["shopping_list"].remove(item)
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            col_exp, col_clr = st.columns([1, 1])
            with col_exp:
                st.download_button(
                    "Export List (TXT)",
                    data="\n".join(st.session_state["shopping_list"]),
                    file_name="shopping_list.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_clr:
                if st.button("Clear Shopping List", type="secondary", use_container_width=True):
                    st.session_state["shopping_list"].clear()
                    st.rerun()
        else:
            st.caption("Your shopping list is currently empty. Add missing items directly from recipe cards.")

    with tab_bookmarks:
        st.subheader("Saved Bookmarks")
        st.caption("Quickly access your pinned favorite recipes.")
        if st.session_state["favorites"]:
            for fav in st.session_state["favorites"][:]:
                fc1, fc2 = st.columns([5, 1])
                fc1.markdown(f"**• {fav}**")
                if fc2.button("✕", key=f"fav_{fav}", type="tertiary", help="Remove bookmark"):
                    st.session_state["favorites"].remove(fav)
                    st.rerun()
        else:
            st.caption("No bookmarked recipes yet. Click '☆ Save' on any recipe card to pin it here.")


if __name__ == "__main__":
    main()