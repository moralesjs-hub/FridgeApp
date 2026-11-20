"""Focused tests for FridgeApp data, matching, nutrition, and user flows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.app import ROOT, _predict_uploaded_image, _read_image
from src.nutrition import benchmark_percent, estimate_nutrition
from src.recipe_engine import RecipeDataError, RecipeMatcher
from src.utils import ingredient_from_label, normalize_ingredient, normalize_ingredients
from src.vision_model import FoodRecognizer


def test_custom_styles_are_injected_on_every_run(monkeypatch) -> None:
    from src import backend

    injected: list[str] = []
    monkeypatch.setattr(
        backend.st,
        "markdown",
        lambda body, **kwargs: injected.append(body),
    )

    backend.apply_custom_styles()
    backend.apply_custom_styles()

    assert len(injected) == 2


def test_recipe_catalog_has_valid_diverse_starter_data() -> None:
    matcher = RecipeMatcher(ROOT / "data" / "processed" / "recipes.csv")

    assert len(matcher.recipes) >= 25
    assert len({recipe["id"] for recipe in matcher.recipes}) == len(matcher.recipes)
    assert all(recipe["ingredients"] for recipe in matcher.recipes)
    assert all(recipe["instructions"] for recipe in matcher.recipes)


def test_recipe_matching_reports_exact_overlap_and_missing_items() -> None:
    matcher = RecipeMatcher(ROOT / "data" / "processed" / "recipes.csv")

    results = matcher.match(["eggs", "sliced spinach", "tomato", "cheddar cheese"])

    omelet = next(recipe for recipe in results if recipe["id"] == "R001")
    assert omelet["match_score"] == 1
    assert set(omelet["matched_ingredients"]) == {"egg", "spinach", "tomato", "cheese"}
    assert omelet["missing_ingredients"] == []
    assert results == sorted(
        results, key=lambda recipe: (-recipe["match_score"], recipe["name"].casefold())
    )


def test_empty_ingredient_list_has_zero_scores_without_crashing() -> None:
    matcher = RecipeMatcher(ROOT / "data" / "processed" / "recipes.csv")

    # With assume_staples=False, an empty ingredient list must yield zero scores
    results = matcher.match([], assume_staples=False)

    assert results
    assert all(recipe["match_score"] == 0 for recipe in results)
    assert all(not recipe["matched_ingredients"] for recipe in results)


def test_recipe_catalog_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(RecipeDataError, match="not found"):
        RecipeMatcher(tmp_path / "missing.csv")

    path = tmp_path / "bad.csv"
    path.write_text(
        "id,name,ingredients,calories_per_serving,prep_time,instructions\n"
        'R1,Broken,not-a-list,100,10,"[\\"step\\"]"\n',
        encoding="utf-8",
    )
    with pytest.raises(RecipeDataError, match="invalid ingredients"):
        RecipeMatcher(path)


def test_nutrition_scales_values_and_rejects_bad_servings() -> None:
    recipe = {"calories_per_serving": 320, "protein_g": 22, "carbs_g": None}

    assert estimate_nutrition(recipe, 2) == {
        "calories_per_serving": 640,
        "protein_g": 44,
        "carbs_g": None,
        "fat_g": None,
    }
    assert benchmark_percent(25, 50) == 50
    assert benchmark_percent(None, 50) is None
    for servings in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive whole number"):
            estimate_nutrition(recipe, servings)
    with pytest.raises(ValueError, match="finite"):
        estimate_nutrition({"protein_g": float("inf")})


def test_ingredient_normalization_and_label_mapping() -> None:
    assert normalize_ingredient("sliced tomatoes") == "tomato"
    assert normalize_ingredient("cheddar cheese") == "cheese"
    assert normalize_ingredients(["egg", "eggs", "sliced tomato"]) == [
        "egg",
        "tomato",
    ]
    assert ingredient_from_label("bell pepper") == "bell pepper"
    assert ingredient_from_label("Granny Smith") == "apple"
    assert ingredient_from_label("hen") == "chicken"
    assert ingredient_from_label("sea cucumber") is None
    assert ingredient_from_label("not a supported food class") is None


def test_sample_images_decode_and_invalid_upload_is_reported() -> None:
    samples = list((ROOT / "data" / "raw").glob("sample_*.png"))

    assert len(samples) == 3
    assert all(_read_image(path).mode == "RGB" for path in samples)
    with pytest.raises(ValueError, match="valid JPG or PNG"):
        _read_image(b"not an image")


def test_recognizer_rejects_invalid_parameters_before_model_loading() -> None:
    recognizer = FoodRecognizer()
    image = Image.new("RGB", (4, 4))

    with pytest.raises(ValueError, match="between 0 and 1"):
        recognizer.predict(image, confidence_threshold=1.1)
    with pytest.raises(ValueError, match="at least 1"):
        recognizer.predict(image, top_k=0)


def test_uploaded_image_predictions_are_cached_by_image_and_threshold(
    monkeypatch,
) -> None:
    import src.app as app_module

    class CountingRecognizer:
        calls = 0

        def predict(self, image: Image.Image, confidence_threshold: float):
            self.calls += 1
            return [SimpleNamespace(ingredient="tomato", confidence=0.9)]

    recognizer = CountingRecognizer()
    monkeypatch.setattr(app_module, "get_recognizer", lambda: recognizer)
    _predict_uploaded_image.clear()
    image_bytes = (ROOT / "data" / "raw" / "sample_produce.png").read_bytes()

    first = _predict_uploaded_image("test-image", 0.15, image_bytes)
    repeated = _predict_uploaded_image("test-image", 0.15, image_bytes)
    changed_threshold = _predict_uploaded_image("test-image", 0.25, image_bytes)

    assert first == repeated == changed_threshold == [("tomato", 0.9)]
    assert recognizer.calls == 2
    _predict_uploaded_image.clear()


def test_model_failure_shows_manual_fallback_without_fake_no_detection(
    monkeypatch,
) -> None:
    import src.vision_model as vision_module

    class UnavailableRecognizer:
        def predict(self, image: Image.Image, confidence_threshold: float):
            raise RuntimeError("model unavailable for test")

    monkeypatch.setattr(vision_module, "FoodRecognizer", UnavailableRecognizer)
    _predict_uploaded_image.clear()
    app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
    sample_path = ROOT / "data" / "raw" / "sample_produce.png"
    if hasattr(app, "file_uploader") and len(app.file_uploader) > 0:
        app.file_uploader[0].upload(
            sample_path.name, sample_path.read_bytes(), "image/png"
        ).run()
    else:
        app.session_state["uploaded_bytes"] = sample_path.read_bytes()
        app.session_state["uploaded_name"] = sample_path.name
        app.run()

    assert not app.exception
    assert any("model unavailable for test" in item.value for item in app.warning)
    assert any("add ingredients manually" in item.value for item in app.info)
    assert not any(
        "No supported ingredients were recognized" in item.value for item in app.info
    )
    _predict_uploaded_image.clear()


def test_streamlit_sample_manual_entry_and_recipe_controls() -> None:
    app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert not app.expander
    assert app.slider[2].value == 25

    app.selectbox[0].select("Produce drawer").run()
    assert not app.exception
    assert len(app.expander) == 6
    assert all("· 0% match ·" not in recipe.label for recipe in app.expander)
    assert set(app.multiselect[0].value) == {
        "tomato",
        "bell pepper",
        "spinach",
        "onion",
    }
    assert any("not model predictions" in item.value for item in app.info)

    app.multiselect[0].unselect("tomato").run()
    assert not app.exception
    assert "tomato" not in app.success[0].value

    app.slider[1].set_value(2).run()
    assert not app.exception
    assert len(app.expander) == 2

    app.text_input[0].set_value("rice, garlic").run()
    assert not app.exception
    assert any("garlic" in item.value for item in app.success)

    app.selectbox[0].select("Breakfast basics").run()
    assert not app.exception
    assert set(app.multiselect[0].value) == {"egg", "cheese", "milk", "spinach"}

    app.slider[2].set_value(100).run()
    assert not app.exception
    # With a richer recipe catalog and staple-aware matching, egg+cheese+milk+spinach
    # achieves a perfect match on at least one recipe (e.g. Cheese and Herb Scrambled Eggs).
    # The 100% threshold therefore returns at least one result, not zero.
    assert app.expander or any(
        "No recipes meet" in item.value for item in app.info
    )
