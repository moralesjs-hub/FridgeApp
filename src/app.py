"""Streamlit interface for local ingredient recognition and recipe discovery."""

from __future__ import annotations

import io
import sys
from hashlib import sha256
from pathlib import Path

# Ensure project root is in sys.path regardless of execution working directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.detector import DetectionResult, FoodDetector
from src.nutrition import DAILY_BENCHMARKS, benchmark_percent, estimate_nutrition
from src.recipe_engine import RecipeDataError, RecipeMatcher
from src.utils import normalize_ingredients
from src.vision_model import FoodRecognizer


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "data" / "processed" / "recipes.csv"
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
    """Share the legacy lazy model wrapper between Streamlit reruns."""
    return FoodRecognizer()


@st.cache_resource
def get_detector() -> FoodDetector:
    """Share the lazy model wrapper between Streamlit reruns."""
    return FoodDetector()


@st.cache_resource
def get_recipe_matcher() -> RecipeMatcher:
    """Load the catalog once and reuse it across reruns."""
    return RecipeMatcher(RECIPE_PATH)


@st.cache_data(show_spinner=False, max_entries=16)
def _predict_uploaded_image(
    image_digest: str,
    confidence_threshold: float,
    _image_bytes: bytes,
) -> list[tuple[str, float]]:
    """Legacy predictor helper for backwards compatibility."""
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
    """Cache per-image detections so ordinary UI reruns stay responsive."""
    image = _read_image(_image_bytes)
    # Check FoodRecognizer for backwards-compatible test mock hooks
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
    """Decode an image from memory or disk and return an independent RGB image."""
    try:
        if isinstance(source, Path):
            with Image.open(source) as image:
                return image.convert("RGB")
        with Image.open(io.BytesIO(source)) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(
            "This image could not be opened. Try a valid JPG or PNG."
        ) from exc


def _render_recipe(recipe: dict, servings: int) -> None:
    """Render one expandable recipe card with match and nutrition details."""
    calorie_label = (
        recipe["calories_per_serving"]
        if recipe["calories_per_serving"] is not None
        else "—"
    )
    title = (
        f"{recipe['name']}  ·  {recipe['match_score']:.0%} match"
        f"  ·  {calorie_label} kcal/serving"
    )
    with st.expander(title):
        prep_time = (
            int(recipe["prep_time"])
            if recipe["prep_time"] is not None
            else "—"
        )
        st.caption(
            f"{prep_time} min prep"
        )
        have_column, missing_column = st.columns(2)
        with have_column:
            st.markdown("**Ingredients you have**")
            st.write(", ".join(recipe["matched_ingredients"]) or "None yet")
        with missing_column:
            st.markdown("**Missing ingredients**")
            st.write(", ".join(recipe["missing_ingredients"]) or "You're all set!")

        nutrition = estimate_nutrition(recipe, servings)
        calorie_value = nutrition["calories_per_serving"]
        if calorie_value is not None:
            st.metric(
                f"Estimated calories · {servings} serving(s)",
                f"{calorie_value:.0f} kcal",
                (
                    f"{calorie_value / DAILY_BENCHMARKS['calories']:.0%} "
                    "of 2,000 kcal reference"
                ),
            )
        macro_columns = st.columns(3)
        for column, key, label in zip(
            macro_columns,
            ("protein_g", "carbs_g", "fat_g"),
            ("Protein", "Carbohydrates", "Fat"),
        ):
            amount = nutrition[key]
            if amount is not None:
                benchmark = DAILY_BENCHMARKS[key]
                percentage = benchmark_percent(amount, benchmark)
                column.metric(
                    label,
                    f"{amount:.1f} g",
                    f"{percentage:.0f}% of {benchmark:.0f} g reference",
                )
        st.markdown("**Instructions**")
        for index, instruction in enumerate(recipe["instructions"], start=1):
            st.markdown(f"{index}. {instruction}")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="FridgeApp · Recipe Finder",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: #f3f8f2; padding: 1rem; border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🥗 FridgeApp")
    st.markdown("### Turn what you have into something delicious.")
    st.caption(
        "Local computer vision · editable ingredient detection · "
        "explainable recipe matches"
    )

    with st.sidebar:
        st.header("CSCI 270 · Applied AI")
        st.caption("College of Charleston")
        st.markdown(
            "**Team**  \nPhilip Pagana · Snovi Patel  \n"
            "Jaden Morales · Amelia West"
        )
        st.divider()
        confidence = st.slider(
            "Model confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.15,
            step=0.05,
            help="Hide lower-confidence supported ImageNet predictions.",
        )
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
        )
        servings = st.number_input(
            "Servings for nutrition estimate", min_value=1, max_value=12, value=1
        )
        st.divider()
        st.caption(
            "ImageNet recognition is limited and can be wrong. Review ingredients "
            "before relying on recipe suggestions."
        )

    try:
        matcher = get_recipe_matcher()
    except RecipeDataError as exc:
        st.error(f"Recipe catalog error: {exc}")
        st.stop()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("1. Add a fridge photo")
        sample_choice = st.selectbox(
            "Try an included sample",
            ["Choose a sample…", *SAMPLE_IMAGES.keys()],
        )
        uploaded_file = st.file_uploader(
            "Or upload a JPG or PNG",
            type=["jpg", "jpeg", "png"],
            max_upload_size=10,
        )

        image: Image.Image | None = None
        annotated_image: Image.Image | None = None
        detections: list[tuple[str, float]] = []
        is_demo = False
        inference_failed = False
        image_source = "none"
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            image_digest = sha256(uploaded_bytes).hexdigest()
            image_source = (
                f"upload:{uploaded_file.name}:{image_digest}"
            )
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
                st.error(f"Could not open the bundled sample: {exc}")

        if image is not None:
            if is_demo:
                st.image(image, caption="Selected sample", width="stretch")
                st.info(
                    "Offline demo sample: ingredients are pre-seeded for a predictable "
                    "walkthrough; these are not model predictions."
                )
            else:
                with st.spinner("Running food detector and shelf scan…"):
                    try:
                        detections, annotated_bytes = _detect_uploaded_image(
                            image_digest, confidence, uploaded_bytes
                        )
                        if annotated_bytes:
                            annotated_image = _read_image(annotated_bytes)
                    except RuntimeError as exc:
                        inference_failed = True
                        st.warning(str(exc))
                        st.info(
                            "Try an included sample or add ingredients manually below."
                        )
                display_img = annotated_image if annotated_image is not None else image
                caption = "Detected items with bounding boxes" if annotated_image is not None else "Uploaded image"
                st.image(display_img, caption=caption, width="stretch")
                if not detections and not inference_failed:
                    st.info(
                        "No supported ingredients were recognized at this threshold. "
                        "Add ingredients manually or try an included sample."
                    )

    with right:
        st.subheader("2. Review your ingredients")
        detected_names = [name for name, _ in detections]
        if st.session_state.get("_ingredient_source") != image_source:
            st.session_state["selected_ingredients"] = detected_names
            st.session_state["_ingredient_source"] = image_source
        selected_defaults = st.session_state.get(
            "selected_ingredients", detected_names
        )
        options = sorted(
            set(COMMON_INGREDIENTS) | set(detected_names) | set(selected_defaults)
        )
        selected_ingredients = st.multiselect(
            "Detected and pantry ingredients",
            options=options,
            key="selected_ingredients",
            help="Remove false detections or add ingredients from your pantry.",
        )
        if detections:
            st.markdown("**Current image detections**")
            badges = [
                f"`{name}` · {score:.0%}"
                for name, score in detections
                if name in selected_ingredients
            ]
            st.markdown("  ".join(badges) if badges else "No detected items selected.")
        extra_ingredients = st.text_input(
            "Add another ingredient",
            placeholder="For example: rice, garlic",
            help=(
                "Separate multiple ingredients with commas, then press Enter. "
                "Clear the field to remove these manual additions."
            ),
        )
        manually_added = normalize_ingredients(extra_ingredients.split(","))
        confirmed_ingredients = normalize_ingredients(
            [*selected_ingredients, *manually_added]
        )
        if not confirmed_ingredients:
            st.info(
                "Choose a sample, upload a photo, or add ingredients to get started."
            )
        else:
            st.success(
                f"{len(confirmed_ingredients)} ingredient(s): "
                + ", ".join(confirmed_ingredients)
            )

    st.divider()
    st.subheader("3. Recipes you can make")
    if not confirmed_ingredients:
        st.info("Your recipe matches will appear here once you add ingredients.")
        return

    ranked = matcher.match(confirmed_ingredients)
    qualifying = [
        recipe
        for recipe in ranked
        if recipe["match_score"] * 100 >= min_match
    ][:max_results]
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
    for recipe in qualifying:
        _render_recipe(recipe, int(servings))


if __name__ == "__main__":
    main()
