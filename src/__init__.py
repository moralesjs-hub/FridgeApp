"""FridgeApp source package."""

from src.detector import BoundingBox, DetectedEntity, DetectionResult, FoodDetector
from src.nutrition import DAILY_BENCHMARKS, benchmark_percent, estimate_nutrition
from src.recipe_engine import RecipeMatcher
from src.utils import (
    CULINARY_SUBSTITUTIONS,
    PANTRY_STAPLES,
    get_substitutes,
    ingredient_from_label,
    is_pantry_staple,
    normalize_ingredient,
    normalize_ingredients,
)
from src.vision_model import FoodRecognizer, Prediction

__all__ = [
    "BoundingBox",
    "CULINARY_SUBSTITUTIONS",
    "DAILY_BENCHMARKS",
    "DetectedEntity",
    "DetectionResult",
    "FoodDetector",
    "FoodRecognizer",
    "PANTRY_STAPLES",
    "Prediction",
    "RecipeMatcher",
    "benchmark_percent",
    "estimate_nutrition",
    "get_substitutes",
    "ingredient_from_label",
    "is_pantry_staple",
    "normalize_ingredient",
    "normalize_ingredients",
]




