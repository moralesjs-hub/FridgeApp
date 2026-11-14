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
    """Load recipes and rank them by core ingredient coverage.

    Scoring algorithm
    -----------------
    Only *core* ingredients (non-pantry-staple items) contribute to the
    match score.  Pantry staples (oil, salt, flour, water, butter, etc.)
    are assumed to be available in every kitchen and do not penalise a
    recipe if the user does not explicitly list them.

    Partial substitution credit (80 %) is awarded when the user holds a
    recognised culinary substitute for a missing core ingredient.

    score = Σ weight_i  (over matched core ingredients)
            ─────────────────────────────────────────────
            Σ weight_j  (over all core ingredients)

    Each core ingredient carries a base weight of 1.0; substitutes
    resolve to 0.80 of that weight.
    """

    REQUIRED_COLUMNS = {
        "id",
        "name",
        "ingredients",
        "calories_per_serving",
        "prep_time",
        "instructions",
    }

    # Weight awarded when a culinary substitute is detected instead of the
    # exact ingredient.
    SUBSTITUTE_CREDIT = 0.80

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.recipes = self._load_recipes()

    # ------------------------------------------------------------------
    # CSV parsing helpers
    # ------------------------------------------------------------------

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

                    # Separate core ingredients from pantry staples
                    core = [i for i in all_ingredients if not is_pantry_staple(i)]
                    staples = [i for i in all_ingredients if is_pantry_staple(i)]
                    if not core:
                        # If every listed ingredient is a pantry staple, treat
                        # all of them as core so the recipe is still matchable.
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

                    recipes.append(
                        {
                            "id": recipe_id,
                            "name": name,
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

    # ------------------------------------------------------------------
    # Matching & ranking
    # ------------------------------------------------------------------

    def match(
        self,
        available_ingredients: list[str],
        assume_staples: bool = True,
    ) -> list[dict[str, Any]]:
        """Return recipes sorted by descending weighted core-ingredient coverage.

        Parameters
        ----------
        available_ingredients:
            Normalised list of ingredients the user has on hand.
        assume_staples:
            When *True* (default), pantry staples are considered available
            even if the user did not list them.  Set to *False* to perform
            strict matching that penalises every missing ingredient.
        """
        available = set(normalize_ingredients(available_ingredients))
        if assume_staples:
            available |= PANTRY_STAPLES

        # Build a reverse lookup from each substitute to the available set
        # so we can quickly resolve substitute credit.
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
            matched_via_sub: list[tuple[str, str]] = []  # (required, substitute_used)
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
                    # Try checking if any available item is a substitute for this ingredient
                    subs_for_ing = set(CULINARY_SUBSTITUTIONS.get(ingredient, []))
                    found_sub = subs_for_ing & available
                    if found_sub:
                        sub_used = next(iter(found_sub))
                        matched_via_sub.append((ingredient, sub_used))
                        weighted_score += self.SUBSTITUTE_CREDIT
                    else:
                        missing.append(ingredient)

            match_score = weighted_score / total_weight if total_weight > 0 else 0.0

            # Compile human-readable matched list including substitute labels
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
