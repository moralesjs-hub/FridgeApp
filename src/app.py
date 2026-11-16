"""Streamlit interface and multi-page router for Ingredia."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.detector import FoodDetector
from src.recipe_engine import RecipeMatcher
from src.vision_model import FoodRecognizer
from src.backend import (
    COMMON_INGREDIENTS,
    LOGO_PATH,
    RECIPE_PATH,
    SAMPLE_IMAGES,
    apply_custom_styles,
    detect_uploaded_image,
    init_session_state,
    read_image,
    render_catalog_card,
    render_header,
    render_recipe,
)

# Direct definitions in app.py for test patching compatibility
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
    image = read_image(_image_bytes)
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
    return detect_uploaded_image(image_digest, confidence_threshold, _image_bytes)


_read_image = read_image
_render_recipe = render_recipe
_render_catalog_card = render_catalog_card



def main() -> None:
    st.set_page_config(
        page_title="Ingredia · Intelligence Engine",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    init_session_state()
    render_header()

    # Define navigation pages using Streamlit's native st.navigation & st.Page
    pages = [
        st.Page("views/find_recipes.py", title="My Fridge", icon="🚀", default=True),
        st.Page("views/explore_catalog.py", title="Explore Recipes", icon="🍽️"),
        st.Page("views/saved_recipes.py", title="Saved Recipes", icon="⭐"),
        st.Page("views/shopping_list.py", title="Shopping List", icon="🛒"),
        st.Page("views/settings.py", title="Settings", icon="⚙️"),
    ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
