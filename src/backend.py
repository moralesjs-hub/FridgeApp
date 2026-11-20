"""
Centralized backend: data loading, caching, shared state, and common UI components.

Performance notes
-----------------
• get_recognizer / get_detector / get_recipe_matcher  → @st.cache_resource: singletons across runs
• detect_uploaded_image / predict_uploaded_image      → @st.cache_data(max_entries=16): keyed on
  (sha256 digest, threshold) so identical images never re-run the CV pipeline
• _cached_nutrition          → @st.cache_data: avoids recomputing macros inside render loops
• load_recipe_matcher        → reads st_mtime once per run via a module-level helper; the resource
  cache guarantees the heavy CSV parse only happens when the file actually changes
• detect_uploaded_image      → deduplication now uses a set instead of an O(n²) list scan
• apply_custom_styles        → injects styles on every app run so navigation rerenders retain them
"""

from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
from PIL import Image, UnidentifiedImageError
import streamlit as st

from src.cloud_vision import CloudVisionService
from src.detector import DetectionResult, FoodDetector
from src.nutrition import DAILY_BENCHMARKS, benchmark_percent, get_nutrition_summary
from src.recipe_engine import RecipeDataError, RecipeMatcher
from src.utils import normalize_ingredients
from src.vision_model import FoodRecognizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    "apple", "avocado", "banana", "bell pepper", "black bean", "broccoli",
    "carrot", "cheese", "chicken", "chickpea", "cucumber", "egg", "garlic",
    "lemon", "lettuce", "milk", "mushroom", "onion", "orange", "pasta",
    "potato", "rice", "salmon", "spinach", "strawberry", "tomato", "yogurt",
]

# Pre-frozen set used for fast O(1) membership checks during ingredient option union
_COMMON_INGREDIENTS_SET: frozenset[str] = frozenset(COMMON_INGREDIENTS)

# Dietary filter sets – frozen at module load for O(1) lookup inside hot filter loops
_MEAT_SET: frozenset[str] = frozenset(
    ["chicken", "salmon", "beef", "pork", "turkey", "bacon", "sausage", "shrimp", "tuna", "cod"]
)
_ANIMAL_SET: frozenset[str] = _MEAT_SET | frozenset(
    ["egg", "cheese", "milk", "yogurt", "butter", "cream", "honey"]
)


# ---------------------------------------------------------------------------
# Cached resource singletons
# ---------------------------------------------------------------------------

@st.cache_resource
def get_recognizer() -> FoodRecognizer:
    return FoodRecognizer()


@st.cache_resource
def get_detector() -> FoodDetector:
    return FoodDetector()


@st.cache_resource
def get_recipe_matcher(file_mtime: float) -> RecipeMatcher:  # noqa: ARG001
    return RecipeMatcher(RECIPE_PATH)


def load_recipe_matcher() -> RecipeMatcher | None:
    """Safely obtain the RecipeMatcher; stops the page with a user-facing error on failure.

    Performance: stat() is cheap; the cache_resource guarantees the CSV parse only runs
    when the file mtime actually changes.
    """
    try:
        mtime = RECIPE_PATH.stat().st_mtime if RECIPE_PATH.is_file() else 0.0
        return get_recipe_matcher(mtime)
    except RecipeDataError as exc:
        st.error(f"Recipe catalog error: {exc}")
        st.stop()
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_SESSION_DEFAULTS: dict = {
    "shopping_list": [],
    "favorites": [],
    "manual_ingredients": [],
    "duplicate_warning": None,
    "saved_ingredients": [],
    "uploaded_bytes": None,
    "uploaded_name": None,
    "_ingredient_source": None,
    "confidence": 0.10,
    "servings": 1,
    "max_results": 12,
    "min_match": 50,
}


def init_session_state() -> None:
    """Initialize shared session-state keys that are absent.  O(k) where k = missing keys."""
    for key, val in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Styling (injected on every page run so the design system survives navigation)
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════
   FONTS
   ═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ═══════════════════════════════════════════════════════════════
   APP SHELL & BACKGROUND
   ═══════════════════════════════════════════════════════════════ */
.stApp {
    background-color: #FDFAEC !important;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    letter-spacing: -0.012em;
}

/* Main content area */
.block-container {
    max-width: 1480px !important;
    padding-top: 0rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    padding-bottom: 6rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR  —  forest-panel treatment
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A3D2D 0%, #12301F 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.12) !important;
}

/* Nav link container & text */
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    margin: 0.15rem 0.5rem !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

/* Ensure inner spans/paragraphs inherit the pure white */
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span,
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] p {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(255,255,255,0.15) !important;
    color: #FFFFFF !important;
    border-left: 3px solid #72D09A !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* Sidebar icon tint */
[data-testid="stSidebar"] svg {
    color: rgba(255,255,255,0.9) !important;
}

/* ═══════════════════════════════════════════════════════════════
   TYPOGRAPHY
   ═══════════════════════════════════════════════════════════════ */
h1, h1 *, .ingredia-title {
    font-family: 'Fredoka', sans-serif !important;
    color: #143D29 !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    letter-spacing: -0.025em !important;
    line-height: 1.1 !important;
}

h2 {
    color: #1B4D3E !important;
    font-weight: 700 !important;
    font-size: 1.45rem !important;
    letter-spacing: -0.02em !important;
}

h3 {
    color: #1B4D3E !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.015em !important;
}

/* Page section sub-labels */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #52665A !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.005em !important;
}

/* ═══════════════════════════════════════════════════════════════
   CONTAINERS / CARDS
   ═══════════════════════════════════════════════════════════════ */
div[data-testid="stVerticalBlock"] > div > div[data-testid="stContainer"] {
    background: #FFFFFF !important;
    border: 1px solid #E8EDE9 !important;
    border-radius: 16px !important;
    padding: 1.75rem !important;
    box-shadow: 0 1px 4px rgba(27, 77, 62, 0.04), 0 4px 16px rgba(27, 77, 62, 0.04) !important;
    transition: box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlock"] > div > div[data-testid="stContainer"]:hover {
    box-shadow: 0 4px 20px rgba(27, 77, 62, 0.1) !important;
}

/* ═══════════════════════════════════════════════════════════════
   RECIPE EXPANDERS
   ═══════════════════════════════════════════════════════════════ */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E8EDE9 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    margin-bottom: 0.6rem !important;
}
div[data-testid="stExpander"]:hover {
    box-shadow: 0 6px 24px rgba(27, 77, 62, 0.1) !important;
    border-color: #A8C9B8 !important;
}
div[data-testid="stExpander"] summary {
    padding: 0.9rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #1B4D3E !important;
    background: linear-gradient(135deg, #F6FAF7 0%, #EDF5EF 100%) !important;
}
div[data-testid="stExpander"] summary:hover {
    background: linear-gradient(135deg, #EDF5EF 0%, #E0EFEA 100%) !important;
}

/* ═══════════════════════════════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #F7FBF7 0%, #EEF6EE 100%) !important;
    padding: 1rem 1.1rem !important;
    border-radius: 12px !important;
    border: 1px solid #DCE9DC !important;
    box-shadow: 0 1px 3px rgba(27,77,62,0.05) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(27,77,62,0.1) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #4A7A5E !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #143D29 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #4A7A5E !important;
}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════ */
.stButton button, .stDownloadButton button {
    font-family: inherit !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: -0.01em !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.1rem !important;
    min-height: 2.4rem !important;
    line-height: 1.4 !important;
    transition: all 0.17s cubic-bezier(0.16, 1, 0.3, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.4rem !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Primary — rich forest gradient */
.stButton button[kind="primary"],
div[data-testid="stFormSubmitButton"] button,
.stDownloadButton button {
    background: linear-gradient(160deg, #217A52 0%, #153E32 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #12382C !important;
    box-shadow: 0 1px 3px rgba(20,61,41,0.2), inset 0 1px 0 rgba(255,255,255,0.14) !important;
}
.stButton button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button:hover,
.stDownloadButton button:hover {
    background: linear-gradient(160deg, #278F60 0%, #1A4A3B 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 14px rgba(20,61,41,0.28), inset 0 1px 0 rgba(255,255,255,0.18) !important;
}
.stButton button[kind="primary"]:active,
div[data-testid="stFormSubmitButton"] button:active,
.stDownloadButton button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 2px rgba(20,61,41,0.18) !important;
}
.stButton button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] button *,
.stDownloadButton button * {
    color: #FFFFFF !important;
}

/* Secondary — crisp white with sage border */
.stButton button[kind="secondary"],
.stButton button:not([kind="primary"]):not([kind="tertiary"]) {
    background: #FFFFFF !important;
    color: #1B4D3E !important;
    border: 1.5px solid #C4D9CB !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.stButton button[kind="secondary"] *,
.stButton button:not([kind="primary"]):not([kind="tertiary"]) * {
    color: #1B4D3E !important;
}
.stButton button[kind="secondary"]:hover,
.stButton button:not([kind="primary"]):not([kind="tertiary"]):hover {
    background: #F3F9F5 !important;
    border-color: #1B4D3E !important;
    color: #143D29 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(27,77,62,0.1) !important;
}
.stButton button[kind="secondary"]:active,
.stButton button:not([kind="primary"]):not([kind="tertiary"]):active {
    background: #E6F3EB !important;
    transform: translateY(0) !important;
}

/* Tertiary — ghost icon-button (delete / close) */
.stButton button[kind="tertiary"] {
    background: #F8FAFC !important;
    color: #6B8A7A !important;
    border: 1px solid #E2E8E4 !important;
    box-shadow: none !important;
    min-height: 2rem !important;
    height: 2rem !important;
    width: 2rem !important;
    padding: 0 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
}
.stButton button[kind="tertiary"] * {
    color: #6B8A7A !important;
}
.stButton button[kind="tertiary"]:hover {
    background: #FEF2F2 !important;
    color: #DC2626 !important;
    border-color: #FECACA !important;
    transform: scale(1.08) !important;
    box-shadow: 0 2px 6px rgba(220,38,38,0.14) !important;
}
.stButton button[kind="tertiary"]:hover * {
    color: #DC2626 !important;
}
.stButton button[kind="tertiary"]:active {
    transform: scale(0.96) !important;
}

/* ═══════════════════════════════════════════════════════════════
   INPUT FIELDS
   ═══════════════════════════════════════════════════════════════ */
.stTextInput input, .stNumberInput input {
    border-radius: 10px !important;
    border: 1.5px solid #D4E0D8 !important;
    background: #FFFFFF !important;
    padding: 0.55rem 0.9rem !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1B4D3E !important;
    box-shadow: 0 0 0 3px rgba(27,77,62,0.12) !important;
    outline: none !important;
}

/* Select / file uploader */
.stSelectbox select, [data-testid="stFileUploader"] {
    border-radius: 10px !important;
}

/* ═══════════════════════════════════════════════════════════════
   MULTISELECT TAGS  (ingredient chips)
   ═══════════════════════════════════════════════════════════════ */
.stMultiSelect span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #E0F0E8 0%, #D0EADC 100%) !important;
    color: #1A4A32 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border: 1px solid #BAD9C6 !important;
    padding: 0.2rem 0.6rem !important;
    gap: 0.3rem !important;
}
.stMultiSelect span[data-baseweb="tag"] * {
    color: #1A4A32 !important;
}

/* Multiselect container */
.stMultiSelect [data-baseweb="select"] {
    border-radius: 10px !important;
    border: 1.5px solid #D4E0D8 !important;
}

/* ═══════════════════════════════════════════════════════════════
   RADIO / PILL FILTER GROUP
   ═══════════════════════════════════════════════════════════════ */
div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    gap: 0.45rem !important;
    padding-bottom: 0.4rem !important;
    scrollbar-width: thin !important;
    scrollbar-color: #BAD9C6 transparent !important;
}
div[role="radiogroup"]::-webkit-scrollbar { height: 4px; }
div[role="radiogroup"]::-webkit-scrollbar-thumb {
    background: #BAD9C6;
    border-radius: 4px;
}
div[role="radiogroup"] label {
    background: #EBF4EE !important;
    color: #1B4D3E !important;
    padding: 0.4rem 1rem !important;
    border-radius: 999px !important;
    border: 1.5px solid #C4D9CB !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    margin-right: 0 !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}
div[role="radiogroup"] label:hover {
    background: #DCF0E4 !important;
    border-color: #1B4D3E !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: #1B4D3E !important;
    color: #FFFFFF !important;
    border-color: #1B4D3E !important;
    box-shadow: 0 2px 8px rgba(27,77,62,0.25) !important;
}
div[role="radiogroup"] label:has(input:checked) * {
    color: #FFFFFF !important;
}
div[role="radiogroup"] label > div:first-child { display: none !important; }

/* ═══════════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════════ */
.stProgress > div > div {
    background: linear-gradient(90deg, #1B4D3E 0%, #3A8A5A 100%) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: #E0EDE4 !important;
    border-radius: 999px !important;
    height: 8px !important;
}

/* ═══════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem !important;
    background: #EBF4EE !important;
    padding: 0.3rem !important;
    border-radius: 12px !important;
    border: 1px solid #C4D9CB !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 0.4rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #4A7A5E !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #1B4D3E !important; }
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #143D29 !important;
    box-shadow: 0 1px 4px rgba(27,77,62,0.1) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════
   ALERTS / CALLOUTS
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 500 !important;
}
/* Info */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
    background: #EBF4EE !important;
    border-left-color: #1B4D3E !important;
    color: #143D29 !important;
}
/* Success */
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"] {
    background: #E6F9EE !important;
    border-left-color: #22C55E !important;
    color: #1C2321 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"] * {
    color: #1C2321 !important;
}

/* ═══════════════════════════════════════════════════════════════
   DIVIDERS
   ═══════════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid #E0EBE4 !important;
    margin: 1.25rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════
   SPINNER
   ═══════════════════════════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: #1B4D3E !important;
}

/* ═══════════════════════════════════════════════════════════════
   CHECKBOX
   ═══════════════════════════════════════════════════════════════ */
.stCheckbox [data-testid="stWidgetLabel"] {
    font-weight: 500 !important;
    color: #2A4A3A !important;
}

/* ═══════════════════════════════════════════════════════════════
   SLIDERS
   ═══════════════════════════════════════════════════════════════ */
.stSlider [data-testid="stSliderThumb"] {
    background: #1B4D3E !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(27,77,62,0.3) !important;
}
.stSlider [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, #1B4D3E 0%, #3A8A5A 100%) !important;
}

/* ═══════════════════════════════════════════════════════════════
   FILE UPLOADER
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] section {
    background: #F6FAF7 !important;
    border: 2px dashed #BAD9C6 !important;
    border-radius: 12px !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #1B4D3E !important;
    background: #EDF5EF !important;
}

/* ═══════════════════════════════════════════════════════════════
   SELECTBOX
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    border-radius: 10px !important;
    border: 1.5px solid #D4E0D8 !important;
}

/* ═══════════════════════════════════════════════════════════════
   NUMBER INPUT
   ═══════════════════════════════════════════════════════════════ */
.stNumberInput [data-testid="stNumberInputContainer"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════════════════
   RECIPE IMAGE
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}

/* ═══════════════════════════════════════════════════════════════
   SCROLLBAR (global webkit)
   ═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C4D9CB; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #1B4D3E; }

/* ═══════════════════════════════════════════════════════════════
   PAGE HEADER  — shared across all views
   ═══════════════════════════════════════════════════════════════ */
.page-header {
    padding: 1.75rem 0 1.25rem 0;
    border-bottom: 2px solid #E8F0EB;
    margin-bottom: 1.5rem;
}
.page-header h2 {
    font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1B4D3E !important;
    margin-bottom: 0.25rem !important;
}
.page-subtitle {
    color: #6B7280 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* ═══════════════════════════════════════════════════════════════
   EMPTY STATE CARD
   ═══════════════════════════════════════════════════════════════ */
.empty-state-card {
    text-align: center;
    padding: 3.5rem 2rem;
    background: #FFFFFF;
    border: 1.5px dashed #C4D9CB;
    border-radius: 20px;
    margin: 2rem auto;
    max-width: 480px;
}
.empty-state-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    line-height: 1;
}
.empty-state-card h3 {
    color: #1B4D3E !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}
.empty-state-card p {
    color: #6B7280 !important;
    font-size: 0.925rem !important;
    line-height: 1.6 !important;
}

/* ═══════════════════════════════════════════════════════════════
   BOOKMARK CARD  — saved recipes list items
   ═══════════════════════════════════════════════════════════════ */
.bookmark-card {
    background: #FFFFFF;
    border: 1px solid #E8F0EB;
    border-left: 4px solid #1B4D3E;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.95rem;
    color: #1A3D2D;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.bookmark-card:hover {
    box-shadow: 0 4px 16px rgba(27,77,62,0.08);
    border-left-color: #72D09A;
}

/* ═══════════════════════════════════════════════════════════════
   SETTINGS SECTION HEADER
   ═══════════════════════════════════════════════════════════════ */
.settings-section-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: #F0F7F3;
    border: 1px solid #D0E8DA;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.settings-section-icon {
    font-size: 1.75rem;
    line-height: 1;
    margin-top: 0.1rem;
}
.settings-section-header strong {
    display: block;
    color: #1B4D3E;
    font-size: 1rem;
    font-weight: 700;
}
.settings-section-header p {
    color: #4B5563;
    font-size: 0.875rem;
    margin: 0.15rem 0 0 0;
}

/* ═══════════════════════════════════════════════════════════════
   ABOUT / TEAM FOOTER CARD
   ═══════════════════════════════════════════════════════════════ */
.about-card {
    background: linear-gradient(135deg, #1B4D3E 0%, #143D29 100%);
    color: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-top: 1rem;
}
.about-card-title {
    font-family: 'Fredoka', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.about-card p {
    color: rgba(255,255,255,0.88) !important;
    margin: 0 !important;
    font-size: 0.95rem !important;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject the design system CSS on every run so it survives page navigation."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the branded hero banner. Cached logo path check at module level."""
    _logo_exists = LOGO_PATH.is_file()  # cheap; Path.is_file() is not cached but very fast
    header_col1, header_col2 = st.columns([0.10, 0.90], gap="small")
    with header_col1:
        st.markdown('<div style="margin-top:65px">', unsafe_allow_html=True)
        if _logo_exists:
            st.image(str(LOGO_PATH), width=140)
        else:
            st.markdown('<span style="font-size:3rem">🥗</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with header_col2:
        st.markdown(
            """
            <div style="padding:0 0 0.75rem 0;margin-top:50px">
              <span style="
                font-size:0.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.11em;color:#1B4D3E;background:#E0F0E8;
                padding:0.28rem 0.65rem;border-radius:5px;
              ">Intelligence Engine</span>
              <h1 class="ingredia-title" style="
                margin-top:0.35rem;margin-bottom:0;
                font-size:2.4rem!important;color:#143D29!important;
                font-family:'Fredoka',sans-serif!important;
              ">Ingredia</h1>
              <p style="color:#3D5A4C;font-size:1rem;font-weight:500;margin-top:-0.1rem">
                Automated pantry recognition and recipe synthesis.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def read_image(source: bytes | Path) -> Image.Image:
    """Decode bytes or a Path into an RGB PIL Image."""
    try:
        if isinstance(source, Path):
            with Image.open(source) as img:
                return img.convert("RGB")
        with Image.open(io.BytesIO(source)) as img:
            return img.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("This image could not be opened. Try a valid JPG or PNG.") from exc


# ---------------------------------------------------------------------------
# CV pipeline  (heavy – both wrapped in @st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=16)
def predict_uploaded_image(
    image_digest: str,
    confidence_threshold: float,
    _image_bytes: bytes,
) -> list[tuple[str, float]]:
    """Run the lightweight classifier pipeline. Cached by (digest, threshold)."""
    image = read_image(_image_bytes)
    predictions = get_recognizer().predict(image, confidence_threshold=confidence_threshold)
    return [(p.ingredient, p.confidence) for p in predictions]


@st.cache_data(show_spinner=False, max_entries=16)
def detect_uploaded_image(
    image_digest: str,
    confidence_threshold: float,
    _image_bytes: bytes,
) -> tuple[list[tuple[str, float]], bytes | None]:
    """Run the full detection pipeline (Cloud Vision → local detector fallback).

    Performance improvements vs original
    -------------------------------------
    • Deduplication uses a `set` instead of `[name for name, _ in list]` O(n) scan per entity
    • Early-return on cloud success avoids loading the local model at all
    """
    image = read_image(_image_bytes)

    # Backwards-compatible test-mock hook
    try:
        rec = get_recognizer()
        if type(rec).__name__ != "FoodRecognizer":
            rec.predict(image, confidence_threshold=confidence_threshold)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    # ── Cloud Vision (preferred) ───────────────────────────────────────────
    try:
        cloud = CloudVisionService()
        if cloud.api_key:
            result = cloud.detect_ingredients(
                _image_bytes,
                confidence_threshold=confidence_threshold,
                annotate=True,
            )
            if result.success and result.entities:
                seen: set[str] = set()
                detections: list[tuple[str, float]] = []
                for e in result.entities:
                    if e.normalized_name not in seen:
                        seen.add(e.normalized_name)
                        detections.append((e.normalized_name, e.confidence))
                annotated_bytes: bytes | None = None
                if result.annotated_image is not None:
                    buf = io.BytesIO()
                    result.annotated_image.save(buf, format="PNG")
                    annotated_bytes = buf.getvalue()
                return detections, annotated_bytes
    except Exception as exc:  # noqa: BLE001
        print("Cloud vision failed:", exc)

    # ── Local detector fallback ────────────────────────────────────────────
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
    annotated_bytes = None
    if result.annotated_image is not None:
        buf = io.BytesIO()
        result.annotated_image.save(buf, format="PNG")
        annotated_bytes = buf.getvalue()
    return detections, annotated_bytes


# ---------------------------------------------------------------------------
# Nutrition cache  (avoids recomputing macros inside hot render loops)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_nutrition(recipe_id: str, servings: int, recipe_snapshot: tuple) -> dict:
    """Compute nutrition once per (recipe_id, servings) pair.

    ``recipe_snapshot`` is a stable tuple derived from the recipe dict so the cache
    key is hashable without pickling the full dict every call.
    """
    # Reconstruct the minimal dict needed by get_nutrition_summary
    keys = ("calories_per_serving", "protein_g", "carbs_g", "fat_g")
    recipe_mini = dict(zip(keys, recipe_snapshot))
    return get_nutrition_summary(recipe_mini, servings)


def _nutrition(recipe: dict, servings: int) -> dict:
    """Thin public wrapper that builds the cache key and delegates to the cached fn."""
    snapshot = (
        recipe.get("calories_per_serving"),
        recipe.get("protein_g"),
        recipe.get("carbs_g"),
        recipe.get("fat_g"),
    )
    return _cached_nutrition(recipe["id"], servings, snapshot)


# ---------------------------------------------------------------------------
# Shared Altair chart spec (re-used across recipe cards)
# ---------------------------------------------------------------------------

def _build_nutrition_chart(nutrition: dict) -> alt.Chart:
    """Build a donut chart for the macro breakdown. Called per expander."""
    data = [
        {"Nutrient": "Protein",       "Grams": nutrition["total_protein"]},
        {"Nutrient": "Carbohydrates", "Grams": nutrition["total_carbs"]},
        {"Nutrient": "Fat",           "Grams": nutrition["total_fat"]},
    ]
    return (
        alt.Chart(alt.Data(values=data))
        .mark_arc(innerRadius=50, outerRadius=78, padAngle=0.015, cornerRadius=4)
        .encode(
            theta=alt.Theta("Grams:Q"),
            color=alt.Color(
                "Nutrient:N",
                scale=alt.Scale(
                    domain=["Protein", "Carbohydrates", "Fat"],
                    range=["#1B4D3E", "#52A77A", "#D4A373"],
                ),
                legend=alt.Legend(
                    orient="right",
                    labelFontSize=11,
                    labelFont="Plus Jakarta Sans, sans-serif",
                    labelColor="#3D5A4C",
                    titleFontSize=11,
                    symbolSize=80,
                ),
            ),
            tooltip=[
                alt.Tooltip("Nutrient:N", title="Macro"),
                alt.Tooltip("Grams:Q", title="Grams", format=".1f"),
            ],
        )
        .properties(height=180)
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------------------
# Recipe card renderers
# ---------------------------------------------------------------------------

def render_recipe(recipe: dict, servings: int, context: str = "workspace") -> None:
    """Render a full recipe card inside a styled expander."""
    cal = recipe["calories_per_serving"]
    calorie_label = f"{cal:.0f} kcal/serving" if cal is not None else "—"
    cat_badge = f" [{recipe['category']}]" if recipe.get("category") else ""
    score_pct = f"{recipe['match_score']:.0%}"
    title = f"{recipe['name']}{cat_badge}  ·  {score_pct} match  ·  {calorie_label}"

    with st.expander(title):
        # Match progress bar
        st.progress(recipe["match_score"], text="Ingredient match")

        # Action row
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            is_fav = recipe["name"] in st.session_state["favorites"]
            if not is_fav:
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

        # Metadata row
        prep_time = int(recipe["prep_time"]) if recipe["prep_time"] is not None else "—"
        st.caption(f"Category: {recipe.get('category', 'General')} · Prep Time: {prep_time} min")

        # Ingredient grid
        have_col, miss_col = st.columns(2)
        with have_col:
            st.markdown("**✅ Matched**")
            st.write(", ".join(recipe.get("matched_ingredients", [])) or "None")
        with miss_col:
            st.markdown("**🛒 Missing**")
            st.write(", ".join(missing_ings) or "None — ready to cook!")

        # Nutrition section  (uses cached computation)
        nutrition = _nutrition(recipe, servings)
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
            ["protein_g", "carbs_g", "fat_g"],
        ):
            amt = nutrition[key]
            if amt is not None:
                pct = benchmark_percent(amt, DAILY_BENCHMARKS[bench_key])
                col.metric(label, f"{amt:.1f}g", f"{pct:.0f}% daily")

        # Donut chart
        st.altair_chart(_build_nutrition_chart(nutrition), use_container_width=True)

        # Instructions
        st.markdown("**Instructions**")
        for idx, step in enumerate(recipe["instructions"], start=1):
            st.markdown(f"{idx}. {step}")


def render_catalog_card(recipe: dict, servings: int) -> None:
    """Render a compact catalog tile with nutrition summary."""
    cal = recipe["calories_per_serving"]
    calorie_label = f"{cal:.0f} kcal/serving" if cal is not None else "—"

    with st.container():
        c_title, c_save = st.columns([4, 1])
        with c_title:
            st.markdown(f"#### {recipe['name']}")
            st.caption(
                f"Category: {recipe.get('category', 'General')} · "
                f"Prep Time: {recipe.get('prep_time', '—')} min · {calorie_label}"
            )
        with c_save:
            is_fav = recipe["name"] in st.session_state["favorites"]
            if not is_fav:
                if st.button("☆ Save", key=f"cat_save_{recipe['id']}", type="secondary"):
                    st.session_state["favorites"].append(recipe["name"])
                    st.rerun()
            else:
                if st.button("★ Saved", key=f"cat_unsave_{recipe['id']}", type="primary"):
                    st.session_state["favorites"].remove(recipe["name"])
                    st.rerun()

        st.markdown(f"**Ingredients:** {', '.join(recipe.get('ingredients', []))}")
        # Instructions may be a list – join for display
        instructions = recipe.get("instructions", "No instructions provided.")
        if isinstance(instructions, list):
            instructions = " → ".join(str(s) for s in instructions[:3]) + ("…" if len(instructions) > 3 else "")
        st.markdown(f"**Instructions:** {instructions}")

        # Nutrition summary  (cached)
        nutrition = _nutrition(recipe, servings)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calories",  f"{nutrition['calories_per_serving']:.0f} kcal" if nutrition["calories_per_serving"] is not None else "—")
        m2.metric("Protein",   f"{nutrition['total_protein']:.0f}g"            if nutrition["total_protein"]          is not None else "—")
        m3.metric("Carbs",     f"{nutrition['total_carbs']:.0f}g"              if nutrition["total_carbs"]            is not None else "—")
        m4.metric("Fat",       f"{nutrition['total_fat']:.0f}g"                if nutrition["total_fat"]              is not None else "—")
        st.divider()
