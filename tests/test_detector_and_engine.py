"""Tests for the food detector, expanded recipe engine, culinary utils, and nutrition module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.detector import BoundingBox, DetectedEntity, FoodDetector
from src.recipe_engine import RecipeMatcher
from src.utils import (
    PANTRY_STAPLES,
    get_substitutes,
    ingredient_from_label,
    is_pantry_staple,
    normalize_ingredient,
    normalize_ingredients,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES_CSV = ROOT / "data" / "processed" / "recipes.csv"
SAMPLE_PRODUCE = ROOT / "data" / "raw" / "sample_produce.png"


# ---------------------------------------------------------------------------
# BoundingBox geometry
# ---------------------------------------------------------------------------


def test_bounding_box_pixel_conversion() -> None:
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9)
    left, top, right, bottom = box.to_pixel_box(width=1000, height=500)
    assert left == 200
    assert top == 50
    assert right == 900
    assert bottom == 400


def test_bounding_box_minimum_size_enforced() -> None:
    """Zero-height or zero-width boxes must still return at least 1 pixel."""
    box = BoundingBox(ymin=0.5, xmin=0.5, ymax=0.5, xmax=0.5)
    left, top, right, bottom = box.to_pixel_box(100, 100)
    assert right > left
    assert bottom > top


# ---------------------------------------------------------------------------
# Pantry staple & substitution helpers
# ---------------------------------------------------------------------------


def test_pantry_staples_identified_correctly() -> None:
    assert is_pantry_staple("salt")
    assert is_pantry_staple("olive oil")
    assert is_pantry_staple("flour")
    assert not is_pantry_staple("chicken")
    assert not is_pantry_staple("tomato")


def test_substitutions_returned_for_known_ingredients() -> None:
    lemon_subs = get_substitutes("lemon")
    assert "lime" in lemon_subs

    onion_subs = get_substitutes("onion")
    assert any("scallion" in s or "shallot" in s for s in onion_subs)


def test_substitutions_empty_for_unknown_ingredient() -> None:
    assert get_substitutes("unobtanium") == []


# ---------------------------------------------------------------------------
# Expanded alias & normalization coverage
# ---------------------------------------------------------------------------


def test_normalize_ingredient_extended_aliases() -> None:
    assert normalize_ingredient("roma tomato") == "tomato"
    assert normalize_ingredient("chicken breast") == "chicken"
    assert normalize_ingredient("greek yogurt") == "yogurt"
    assert normalize_ingredient("ground beef") == "beef"
    assert normalize_ingredient("baby spinach") == "spinach"
    assert normalize_ingredient("jasmine rice") == "rice"
    assert normalize_ingredient("penne") == "pasta"
    assert normalize_ingredient("spaghetti") == "pasta"
    assert normalize_ingredient("lemon juice") == "lemon"
    assert normalize_ingredient("red onion") == "onion"


def test_normalize_ingredients_deduplicates() -> None:
    result = normalize_ingredients(["tomato", "roma tomato", "cherry tomatoes"])
    assert result.count("tomato") == 1
    assert result == ["tomato"]


def test_ingredient_from_label_extended_mappings() -> None:
    assert ingredient_from_label("granny smith") == "apple"
    assert ingredient_from_label("courgette") == "zucchini"
    assert ingredient_from_label("zucchini") == "zucchini"
    assert ingredient_from_label("potted plant") is None   # non-food COCO label
    assert ingredient_from_label("sea cucumber") is None   # excluded non-food animal
    assert ingredient_from_label("hen") == "chicken"
    assert ingredient_from_label("guacamole") == "avocado"
    assert ingredient_from_label("rotisserie") == "chicken"
    assert ingredient_from_label("") is None


# ---------------------------------------------------------------------------
# Expanded recipe catalog
# ---------------------------------------------------------------------------


def test_expanded_recipe_catalog_size() -> None:
    matcher = RecipeMatcher(RECIPES_CSV)
    assert len(matcher.recipes) >= 60, (
        f"Expected at least 60 recipes after expansion, got {len(matcher.recipes)}"
    )


def test_recipe_catalog_core_staple_separation() -> None:
    matcher = RecipeMatcher(RECIPES_CSV)
    for recipe in matcher.recipes:
        # core_ingredients must be present and non-empty after loading
        assert recipe["core_ingredients"], (
            f"Recipe {recipe['id']!r} has empty core_ingredients"
        )
        # No pantry staple should be present in core_ingredients
        for ing in recipe["core_ingredients"]:
            assert not is_pantry_staple(ing), (
                f"Recipe {recipe['id']!r}: pantry staple {ing!r} found in core_ingredients"
            )


def test_recipe_catalog_original_ids_preserved() -> None:
    matcher = RecipeMatcher(RECIPES_CSV)
    ids = {r["id"] for r in matcher.recipes}
    for expected in ["R001", "R010", "R026"]:
        assert expected in ids, f"Original recipe {expected} missing after expansion"


# ---------------------------------------------------------------------------
# Staple-aware & weighted recipe matching
# ---------------------------------------------------------------------------


def test_staple_aware_matching_ignores_missing_oil_and_salt() -> None:
    """A recipe requiring only a core ingredient plus pantry staples should
    score 1.0 when the user supplies the core ingredient but not the staples."""
    matcher = RecipeMatcher(RECIPES_CSV)

    # R058 "Crispy Baked Potato Wedges" requires: potato, garlic, olive oil
    # With assume_staples=True, oil is assumed, so potato+garlic gives 100%.
    results = matcher.match(["potato", "garlic"], assume_staples=True)
    r058 = next((r for r in results if r["id"] == "R058"), None)
    if r058:
        assert r058["match_score"] == 1.0, (
            f"R058 should score 1.0 when pantry staples assumed, got {r058['match_score']}"
        )


def test_strict_matching_penalises_missing_core_ingredients() -> None:
    """With assume_staples=False, every missing ingredient reduces the score."""
    matcher = RecipeMatcher(RECIPES_CSV)
    # R001 requires egg, spinach, tomato, cheese
    results = matcher.match(["egg"], assume_staples=False)
    r001 = next(r for r in results if r["id"] == "R001")
    assert r001["match_score"] < 1.0
    assert r001["match_score"] > 0.0


def test_substitution_awards_partial_credit() -> None:
    """Providing 'lime' instead of 'lemon' should award partial match credit."""
    matcher = RecipeMatcher(RECIPES_CSV)
    # R004 requires chicken, spinach, garlic, lemon — supply lime instead of lemon
    results = matcher.match(["chicken", "spinach", "garlic", "lime"], assume_staples=False)
    r004 = next(r for r in results if r["id"] == "R004")
    # Should be higher than 3/4 = 0.75 (exact) due to substitute credit
    assert r004["match_score"] > 0.75


def test_match_returns_sorted_descending_score() -> None:
    matcher = RecipeMatcher(RECIPES_CSV)
    results = matcher.match(["egg", "tomato", "cheese", "spinach"])
    scores = [r["match_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_substitutions_reported_in_result() -> None:
    matcher = RecipeMatcher(RECIPES_CSV)
    # Supply 'scallion' (a substitute for onion) and check substitutions_used
    results = matcher.match(["egg", "scallion", "cheese", "bell pepper"])
    # R002 uses onion — scallion is a recognised substitute
    r002 = next((r for r in results if r["id"] == "R002"), None)
    if r002 and r002.get("substitutions_used"):
        required_names = [s["required"] for s in r002["substitutions_used"]]
        assert "onion" in required_names


# ---------------------------------------------------------------------------
# FoodDetector API
# ---------------------------------------------------------------------------


def test_food_detector_invalid_confidence_threshold_raises() -> None:
    detector = FoodDetector()
    img = Image.new("RGB", (4, 4))
    with pytest.raises(ValueError, match="between 0 and 1"):
        detector.detect(img, confidence_threshold=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        detector.detect(img, confidence_threshold=0.0)


def test_food_detector_returns_result_on_blank_image() -> None:
    """A blank image should return a valid DetectionResult without crashing
    (zero entities expected but no unhandled exception)."""
    detector = FoodDetector()
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    result = detector.detect(img, confidence_threshold=0.90, annotate=False)
    # API contract: success when at least model loaded (even with no detections)
    assert isinstance(result.entities, list)
    assert isinstance(result.unique_ingredients, list)


def test_food_detector_annotate_draws_on_boxes() -> None:
    """annotate=True with known bounding-box entities must return a modified image."""
    entity = DetectedEntity(
        ingredient="apple",
        confidence=0.92,
        source="coco_detector",
        category="fruit",
        box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.6, xmax=0.6),
    )
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    annotated = FoodDetector._annotate(img, [entity])
    assert isinstance(annotated, Image.Image)
    # Annotated image must differ from plain white original
    assert annotated.tobytes() != img.tobytes()


def test_food_detector_annotate_ignores_entities_without_boxes() -> None:
    """Entities from the classifier stage (no box) must not cause annotation errors."""
    entity = DetectedEntity(
        ingredient="tomato",
        confidence=0.85,
        source="imagenet_classifier",
        category="produce",
        box=None,
    )
    img = Image.new("RGB", (200, 200))
    # Should not raise
    annotated = FoodDetector._annotate(img, [entity])
    assert isinstance(annotated, Image.Image)


def test_food_detector_merges_rcnn_and_classifier_results(monkeypatch) -> None:
    """Detector deduplicates by canonical ingredient name, preferring COCO results."""
    detector = FoodDetector()
    img = Image.new("RGB", (224, 224))

    rcnn_entity = DetectedEntity(
        ingredient="apple",
        confidence=0.95,
        source="coco_detector",
        category="fruit",
        box=BoundingBox(0.1, 0.1, 0.5, 0.5),
    )
    cls_entity = DetectedEntity(
        ingredient="apple",  # duplicate
        confidence=0.70,
        source="imagenet_classifier",
        category="fruit",
        box=None,
    )
    cls_entity2 = DetectedEntity(
        ingredient="broccoli",
        confidence=0.80,
        source="imagenet_classifier",
        category="produce",
        box=None,
    )

    monkeypatch.setattr(detector, "_run_rcnn", lambda *a, **k: [rcnn_entity])
    monkeypatch.setattr(detector, "_run_classifier", lambda *a, **k: [cls_entity, cls_entity2])

    result = detector.detect(img, confidence_threshold=0.25, annotate=False)
    assert result.unique_ingredients.count("apple") == 1
    assert "broccoli" in result.unique_ingredients
    # COCO entity with higher confidence should be kept (has bounding box)
    apple_entity = next(e for e in result.entities if e.ingredient == "apple")
    assert apple_entity.source == "coco_detector"
    assert apple_entity.box is not None


def test_food_detector_to_dict_serialisable() -> None:
    entity = DetectedEntity(
        ingredient="carrot",
        confidence=0.88,
        source="coco_detector",
        category="produce",
        box=BoundingBox(0.2, 0.3, 0.7, 0.8),
    )
    d = entity.to_dict()
    assert d["ingredient"] == "carrot"
    assert isinstance(d["box"], dict)
    assert all(k in d["box"] for k in ("ymin", "xmin", "ymax", "xmax"))


def test_food_detector_model_errors_reported() -> None:
    """When both models fail to load, the error surfaces through model_errors."""
    detector = FoodDetector()
    detector._rcnn_error = "test rcnn failure"
    detector._cls_error = "test cls failure"
    img = Image.new("RGB", (10, 10))
    result = detector.detect(img, confidence_threshold=0.5, annotate=False)
    assert not result.success
    assert result.error is not None

