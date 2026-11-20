"""Recipe catalog loading, validation, and intelligent ingredient-overlap ranking."""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Any

from src.utils import (
    CULINARY_SUBSTITUTIONS,
    PANTRY_STAPLES,
    get_substitutes,
    is_pantry_staple,
    normalize_ingredients,
)


class RecipeDataError(ValueError):
    """Raised when the recipe catalog is missing or malformed."""


class RecipeMatcher:
    """Load recipes and rank them by core ingredient coverage."""

    REQUIRED_COLUMNS = {
        "id",
        "name",
        "ingredients",
        "calories_per_serving",
        "prep_time",
        "instructions",
    }

    SUBSTITUTE_CREDIT = 0.80

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.recipes = self._load_recipes()
        self._inverted_index: dict[str, set[int]] = self._build_inverted_index()

    def _build_inverted_index(self) -> dict[str, set[int]]:
        index: dict[str, set[int]] = {}
        for idx, recipe in enumerate(self.recipes):
            for ing in recipe["core_ingredients"]:
                index.setdefault(ing, set()).add(idx)
        return index

    @staticmethod
    def _parse_list(value: str, field: str, recipe_id: str) -> list[str]:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise RecipeDataError(
                f"Recipe {recipe_id!r} has invalid {field}; expected a list."
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise RecipeDataError(
                f"Recipe {recipe_id!r} has invalid {field}; expected a list of strings."
            )
        return parsed

    @staticmethod
    def _optional_nonnegative_float(value: str | None) -> float | None:
        if value is None or not str(value).strip():
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Values cannot be negative.")
        return number

    @staticmethod
    def _infer_category(name: str, ingredients: list[str], prep_time: float | None) -> str:
        text = f"{name} {' '.join(ingredients)}".lower()
        if any(k in text for k in ["omelet", "frittata", "pancake", "smoothie", "toast", "oatmeal", "parfait", "french toast", "waffle", "eggs benedict"]):
            return "Breakfast"
        if any(k in text for k in ["pasta", "risotto", "gnocchi", "bruschetta", "calzone", "panini", "minestrone", "lasagna", "carbonara", "scampi", "caprese", "pizza", "bolognese", "penne", "parmesan", "italian"]):
            return "Italian"
        if any(k in text for k in ["taco", "fajita", "burrito", "enchilada", "quesadilla", "tamale", "pozole", "tostada", "chile", "flauta", "rancheros", "guacamole", "salsa", "mexican"]):
            return "Mexican"
        if any(k in text for k in ["stir-fry", "fried rice", "lo mein", "dumpling", "spring roll", "pho", "ramen", "teriyaki", "bao", "miso", "tofu", "kimchi", "soy sauce", "edamame", "asian"]):
            return "Asian"
        if any(k in text for k in ["burger", "steak", "bbq", "meatloaf", "macaroni", "pot pie", "hot dog", "chowder", "cheesesteak", "blt", "baked potato", "shepherd's pie", "wings", "american"]):
            return "American"
        if any(k in text for k in ["falafel", "hummus", "gyro", "shawarma", "tabbouleh", "kebab", "moussaka", "souvlaki", "baklava", "spanakopita", "shakshuka", "quinoa", "greek", "mediterranean"]):
            return "Mediterranean"
        if any(k in text for k in ["ratatouille", "quiche", "coq au vin", "bouillabaisse", "cassoulet", "nicoise", "croissant", "bourguignon", "souffle", "creme brulee", "french"]):
            return "French"
        if any(k in text for k in ["tikka masala", "paneer", "biryani", "dal", "chole", "samosa", "naan", "vada", "rogan josh", "aloo gobi", "butter chicken", "curry", "indian"]):
            return "Indian"
        if any(k in text for k in ["pad thai", "green curry", "tom yum", "massaman", "papaya", "pad see ew", "mango sticky rice", "satay", "tom kha", "panang", "thai", "coconut milk"]):
            return "Thai"
        if prep_time is not None and prep_time <= 20:
            return "Quick Meals"
        cuisines = ["Italian", "Mexican", "Asian", "American", "Mediterranean", "French", "Indian", "Thai"]
        return cuisines[sum(ord(c) for c in name) % len(cuisines)]

    def _load_recipes(self) -> list[dict[str, Any]]:
        if not self.csv_path.is_file():
            raise RecipeDataError(f"Recipe catalog not found: {self.csv_path}")
        try:
            with self.csv_path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                columns = set(reader.fieldnames or [])
                missing = self.REQUIRED_COLUMNS - columns
                if missing:
                    raise RecipeDataError(
                        f"Recipe catalog is missing columns: {', '.join(sorted(missing))}."
                    )
                recipes: list[dict[str, Any]] = []
                identifiers: set[str] = set()
                for row_number, row in enumerate(reader, start=2):
                    recipe_id = (row.get("id") or "").strip()
                    name = (row.get("name") or "").strip()
                    if not recipe_id or not name:
                        raise RecipeDataError(
                            f"Recipe catalog row {row_number} needs both an id and name."
                        )
                    if recipe_id in identifiers:
                        raise RecipeDataError(f"Duplicate recipe id: {recipe_id}.")

                    all_ingredients = normalize_ingredients(
                        self._parse_list(
                            row.get("ingredients") or "",
                            "ingredients",
                            recipe_id,
                        )
                    )
                    instructions = self._parse_list(
                        row.get("instructions") or "", "instructions", recipe_id
                    )
                    if not all_ingredients:
                        raise RecipeDataError(
                            f"Recipe {recipe_id!r} must include at least one ingredient."
                        )

                    core = [i for i in all_ingredients if not is_pantry_staple(i)]
                    staples = [i for i in all_ingredients if is_pantry_staple(i)]
                    if not core:
                        core = all_ingredients
                        staples = []

                    try:
                        calories = self._optional_nonnegative_float(
                            row.get("calories_per_serving")
                        )
                        prep_time = self._optional_nonnegative_float(
                            row.get("prep_time")
                        )
                        nutrition = {
                            key: self._optional_nonnegative_float(row.get(key))
                            for key in ("protein_g", "carbs_g", "fat_g")
                        }
                    except ValueError as exc:
                        raise RecipeDataError(
                            f"Recipe {recipe_id!r} has a negative or invalid numeric value."
                        ) from exc

                    raw_category = (row.get("category") or "").strip()
                    category = raw_category if raw_category else self._infer_category(name, all_ingredients, prep_time)

                    recipes.append(
                        {
                            "id": recipe_id,
                            "name": name,
                            "category": category,
                            "ingredients": all_ingredients,
                            "core_ingredients": core,
                            "pantry_staples": staples,
                            "calories_per_serving": calories,
                            "prep_time": prep_time,
                            "instructions": instructions,
                            **nutrition,
                        }
                    )
                    identifiers.add(recipe_id)
        except (OSError, csv.Error) as exc:
            raise RecipeDataError(f"Could not read recipe catalog: {exc}") from exc
        if not recipes:
            raise RecipeDataError("The recipe catalog contains no recipes.")
        return recipes

    def match(
        self,
        available_ingredients: list[str],
        assume_staples: bool = True,
    ) -> list[dict[str, Any]]:
        available = set(normalize_ingredients(available_ingredients))
        if assume_staples:
            available |= PANTRY_STAPLES

        available_substitutes: dict[str, str] = {}
        for ing in list(available):
            for sub in get_substitutes(ing):
                if sub not in available_substitutes:
                    available_substitutes[sub] = ing

        ranked: list[dict[str, Any]] = []
        for recipe in self.recipes:
            core = recipe["core_ingredients"]
            if not core:
                continue

            matched_exact: list[str] = []
            matched_via_sub: list[tuple[str, str]] = []
            missing: list[str] = []
            weighted_score = 0.0
            total_weight = float(len(core))

            for ingredient in core:
                if ingredient in available:
                    matched_exact.append(ingredient)
                    weighted_score += 1.0
                elif ingredient in available_substitutes:
                    sub_used = available_substitutes[ingredient]
                    matched_via_sub.append((ingredient, sub_used))
                    weighted_score += self.SUBSTITUTE_CREDIT
                else:
                    subs_for_ing = set(CULINARY_SUBSTITUTIONS.get(ingredient, []))
                    found_sub = subs_for_ing & available
                    if found_sub:
                        sub_used = next(iter(found_sub))
                        matched_via_sub.append((ingredient, sub_used))
                        weighted_score += self.SUBSTITUTE_CREDIT
                    else:
                        missing.append(ingredient)

            match_score = weighted_score / total_weight if total_weight > 0 else 0.0

            matched_all = sorted(matched_exact) + sorted(
                f"{req} (via {sub})" for req, sub in matched_via_sub
            )

            ranked.append(
                {
                    **recipe,
                    "matched_ingredients": matched_all,
                    "missing_ingredients": sorted(missing),
                    "match_score": round(match_score, 4),
                    "substitutions_used": [
                        {"required": req, "substitute": sub}
                        for req, sub in matched_via_sub
                    ],
                }
            )

        return sorted(
            ranked,
            key=lambda r: (-r["match_score"], r["name"].casefold()),
        )