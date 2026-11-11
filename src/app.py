"""Streamlit interface for local and cloud ingredient recognition and recipe discovery."""

from __future__ import annotations

import io
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloud_vision import CloudVisionService  # noqa: E402
from src.nutrition import DAILY_BENCHMARKS, benchmark_percent, estimate_nutrition  # noqa: E402
from src.recipe_engine import RecipeDataError, RecipeMatcher  # noqa: E402
from src.utils import normalize_ingredients  # noqa: E402
from src.vision_model import FoodRecognizer  # noqa: E402

RECIPE_PATH = ROOT / "data" / "processed" / "recipes.csv"
RAW = ROOT / "data" / "raw"

SAMPLE_IMAGES = {
    "Produce drawer": {
        "path": RAW / "sample_produce.png",
        "ingredients": [("tomato", 0.99), ("bell pepper", 0.98), ("spinach", 0.97), ("onion", 0.96)],
    },
    "Breakfast basics": {
        "path": RAW / "sample_breakfast.png",
        "ingredients": [("egg", 0.99), ("cheese", 0.98), ("milk", 0.97), ("spinach", 0.96)],
    },
    "Fruit and yogurt": {
        "path": RAW / "sample_fruit.png",
        "ingredients": [("apple", 0.99), ("yogurt", 0.98), ("banana", 0.97), ("strawberry", 0.96)],
    },
}

COMMON_INGREDIENTS = [
    "apple", "avocado", "banana", "bell pepper", "black bean", "broccoli",
    "carrot", "cheese", "chicken", "chickpea", "cucumber", "egg", "garlic",
    "lemon", "lettuce", "milk", "mushroom", "onion", "orange", "pasta",
    "potato", "rice", "salmon", "spinach", "strawberry", "tomato", "yogurt",
]


@st.cache_resource
def get_recognizer() -> FoodRecognizer:
    """Share the lazy model wrapper between Streamlit reruns."""
    return FoodRecognizer()


@st.cache_resource
def get_recipe_matcher() -> RecipeMatcher:
    """Load the catalog once and reuse it across reruns."""
    return RecipeMatcher(RECIPE_PATH)


@st.cache_resource
def get_cloud_vision_service() -> CloudVisionService:
    """Initialize and reuse CloudVisionService across reruns."""
    return CloudVisionService()


@st.cache_data(show_spinner=False, max_entries=16)
def _predict_uploaded_image(
    image_digest: str, confidence_threshold: float, _image_bytes: bytes
) -> list[tuple[str, float]]:
    """Cache per-image predictions so ordinary UI reruns stay responsive."""
    image = _read_image(_image_bytes)
    recognizer = get_recognizer()
    preds = recognizer.predict(image, confidence_threshold=confidence_threshold)
    return [(p.ingredient, p.confidence) for p in preds]


def _read_image(source: bytes | Path) -> Image.Image:
    """Decode an image from memory or disk and return an independent RGB image."""
    try:
        with Image.open(source if isinstance(source, Path) else io.BytesIO(source)) as img:
            return img.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("This image could not be opened. Try a valid JPG or PNG.") from exc


def _render_recipe(recipe: dict, servings: int) -> None:
    """Render one expandable recipe card with match and nutrition details."""
    cal = (
        f"{recipe['calories_per_serving']:.0f}"
        if recipe["calories_per_serving"] is not None
        else "—"
    )
    title = f"{recipe['name']}  ·  {recipe['match_score']:.0%} match  ·  {cal} kcal/serving"

    with st.expander(title):
        prep = int(recipe["prep_time"]) if recipe["prep_time"] is not None else "—"
        st.caption(f"{prep} min prep")
        col_have, col_miss = st.columns(2)
        col_have.markdown("**Ingredients you have**")
        col_have.write(", ".join(recipe["matched_ingredients"]) or "None yet")
        col_miss.markdown("**Missing ingredients**")
        col_miss.write(", ".join(recipe["missing_ingredients"]) or "You're all set!")

        nutr = estimate_nutrition(recipe, servings)
        cal_val = nutr["calories_per_serving"]
        if cal_val is not None:
            pct = cal_val / DAILY_BENCHMARKS["calories"]
            st.metric(
                f"Estimated calories · {servings} serving(s)",
                f"{cal_val:.0f} kcal",
                f"{pct:.0%} of 2,000 kcal reference",
            )

        macro_cols = st.columns(3)
        labels = ("Protein", "Carbohydrates", "Fat")
        for col, key, label in zip(macro_cols, ("protein_g", "carbs_g", "fat_g"), labels):
            amt = nutr[key]
            if amt is not None:
                bm = DAILY_BENCHMARKS[key]
                col.metric(label, f"{amt:.1f} g", f"{benchmark_percent(amt, bm):.0f}% of {bm:.0f} g reference")

        st.markdown("**Instructions**")
        for i, step in enumerate(recipe["instructions"], 1):
            st.markdown(f"{i}. {step}")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="FridgeApp · Recipe Finder",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Sleek, minimalist modern styling with no decorative icons
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3.5rem;
        }
        h1, h2, h3, h4 {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            letter-spacing: -0.02em;
            color: #111827;
        }
        [data-testid="stMetric"] {
            background: #f9fafb;
            padding: 0.85rem 1rem;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        .ingredient-tag {
            display: inline-block;
            background: #f3f4f6;
            color: #1f2937;
            padding: 3px 9px;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("FridgeApp")
    st.caption("Ingredient recognition and recipe discovery")

    try:
        matcher = get_recipe_matcher()
    except RecipeDataError as exc:
        st.error(f"Recipe catalog error: {exc}")
        st.stop()

    tab_recipes, tab_settings = st.tabs(["Recipe Finder", "Settings"])

    # Settings Tab (separate dedicated tab)
    with tab_settings:
        st.subheader("Settings & Preferences")
        col_set1, col_set2 = st.columns(2, gap="large")
        with col_set1:
            st.markdown("**Computer Vision Engine**")
            use_cloud = st.toggle(
                "Cloud Vision AI (Gemini)",
                value=bool(os.getenv("GEMINI_API_KEY") and "pytest" not in sys.modules),
                help="Multimodal AI detection with 2D spatial bounding coordinates.",
            )
            confidence = st.slider(
                "Model confidence threshold",
                0.05, 0.90, 0.15, 0.05,
                help="Hide lower-confidence predictions.",
            )
        with col_set2:
            st.markdown("**Recipe Matching & Nutrition**")
            max_results = st.slider("Maximum recipe results", 1, 12, 6)
            min_match = st.slider("Minimum recipe match", 0, 100, 25, 5, format="%d%%")
            servings = st.number_input("Servings for nutrition estimate", 1, 12, 1)

        st.divider()
        st.caption(
            "CSCI 270 Applied AI · College of Charleston\n\n"
            "Team: Philip Pagana · Snovi Patel · Jaden Morales · Amelia West\n\n"
            "Image recognition can make mistakes. Please verify ingredients before preparing recipes."
        )

    # Main Recipe Finder Tab
    with tab_recipes:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("1. Add a fridge photo")
            sample_choice = st.selectbox(
                "Try an included sample", ["Choose a sample…", *SAMPLE_IMAGES.keys()]
            )
            uploaded_file = st.file_uploader("Or upload a JPG or PNG", type=["jpg", "jpeg", "png"])

            image, detections, is_demo, failed, image_source = None, [], False, False, "none"
            annotated_image = None
            cloud_meta: dict[str, Any] = {}

            if uploaded_file is not None:
                data = uploaded_file.getvalue()
                digest = sha256(data).hexdigest()
                image_source = f"upload:{uploaded_file.name}:{digest}"
                try:
                    image = _read_image(data)
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
                    if use_cloud and os.getenv("GEMINI_API_KEY"):
                        with st.spinner("Analyzing image with Cloud Vision AI…"):
                            try:
                                cloud_service = get_cloud_vision_service()
                                cloud_res = cloud_service.detect_ingredients(
                                    data, confidence_threshold=confidence, annotate=True
                                )
                                if cloud_res.success and cloud_res.entities:
                                    detections = [(e.normalized_name, e.confidence) for e in cloud_res.entities]
                                    annotated_image = cloud_res.annotated_image
                                    cloud_meta = {
                                        "model": cloud_res.model,
                                        "latency": cloud_res.latency_ms,
                                        "count": len(cloud_res.entities),
                                    }
                                elif not cloud_res.success:
                                    st.warning(f"Cloud Vision API error: {cloud_res.error}")
                            except Exception as exc:
                                st.warning(f"Cloud Vision notice: {exc}")

                    if not detections and not failed:
                        with st.spinner("Running the local image classifier…"):
                            try:
                                detections = _predict_uploaded_image(digest, confidence, data)
                            except RuntimeError as exc:
                                failed = True
                                st.warning(str(exc))
                                st.info("Try an included sample or add ingredients manually below.")

                    display_img = annotated_image or image
                    caption = (
                        f"Detected Items ({cloud_meta.get('model', 'cloud')}) · {cloud_meta.get('latency', 0):.0f} ms"
                        if annotated_image
                        else "Selected image"
                    )
                    st.image(display_img, caption=caption, width="stretch")

                    if cloud_meta:
                        st.success(
                            f"Cloud Vision Grounding: {cloud_meta['count']} item(s) detected "
                            f"({cloud_meta['model']} · {cloud_meta['latency']:.0f} ms)"
                        )

                    if not detections and not failed:
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
            selected_defaults = st.session_state.get("selected_ingredients", detected_names)
            options = sorted(set(COMMON_INGREDIENTS) | set(detected_names) | set(selected_defaults))
            selected_ingredients = st.multiselect(
                "Detected and pantry ingredients",
                options=options,
                key="selected_ingredients",
                help="Remove false detections or add ingredients from your pantry.",
            )
            if detections:
                st.markdown("**Detected Items**")
                tags_html = "".join([
                    f'<span class="ingredient-tag">{n} · {s:.0%}</span>'
                    for n, s in detections
                    if n in selected_ingredients
                ])
                if tags_html:
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("No detected items currently selected.")

            extra = st.text_input(
                "Add another ingredient",
                placeholder="For example: rice, garlic",
                help=(
                    "Separate multiple ingredients with commas, then press Enter. "
                    "Clear the field to remove these manual additions."
                ),
            )
            confirmed = normalize_ingredients(
                [*selected_ingredients, *normalize_ingredients(extra.split(","))]
            )

            if not confirmed:
                st.info("Choose a sample, upload a photo, or add ingredients to get started.")
            else:
                st.success(f"{len(confirmed)} ingredient(s): " + ", ".join(confirmed))

        st.divider()
        st.subheader("3. Recipes you can make")
        if not confirmed:
            st.info("Your recipe matches will appear here once you add ingredients.")
            return

        qualifying = [
            r for r in matcher.match(confirmed) if r["match_score"] * 100 >= min_match
        ][:max_results]
        if not qualifying:
            st.info("No recipes meet that match threshold. Lower the minimum match or add more ingredients.")
            return

        st.caption(
            f"Ranked by ingredient coverage · {len(qualifying)} recipe(s) shown · "
            "scores do not account for ingredient quantities or substitutions."
        )
        for recipe in qualifying:
            _render_recipe(recipe, int(servings))


if __name__ == "__main__":
    main()
