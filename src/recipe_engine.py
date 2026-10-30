"""Recipe catalog loading, validation, and ingredient-overlap ranking."""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Any

from src.utils import normalize_ingredients


class RecipeDataError(ValueError):
    """Raised when the recipe catalog is missing or malformed."""


class RecipeMatcher:
    """Load recipes and rank them by the fraction of ingredients available."""

    REQUIRED_COLUMNS = {
        "id",
        "name",
        "ingredients",
        "calories_per_serving",
        "prep_time",
        "instructions",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.recipes = self._load_recipes()

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

    def _load_recipes(self) -> list[dict[str, Any]]:
        if not self.csv_path.is_file():
            raise RecipeDataError(f"Recipe catalog not found: {self.csv_path}")
        try:
            with self.csv_path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                columns = set(reader.fieldnames or [])
                missing = self.REQUIRED_COLUMNS - columns
                if missing:
                    missing_columns = ", ".join(sorted(missing))
                    raise RecipeDataError(
                        f"Recipe catalog is missing columns: {missing_columns}."
                    )
                recipes: list[dict[str, Any]] = []
                identifiers: set[str] = set()
                for row_number, row in enumerate(reader, start=2):
                    recipe_id = (row.get("id") or "").strip()
                    name = (row.get("name") or "").strip()
                    if not recipe_id or not name:
                        raise RecipeDataError(
                            f"Recipe catalog row {row_number} needs "
                            "both an id and name."
                        )
                    if recipe_id in identifiers:
                        raise RecipeDataError(
                            f"Duplicate recipe id: {recipe_id}."
                        )
                    ingredients = normalize_ingredients(
                        self._parse_list(
                            row.get("ingredients") or "",
                            "ingredients",
                            recipe_id,
                        )
                    )
                    instructions = self._parse_list(
                        row.get("instructions") or "", "instructions", recipe_id
                    )
                    if not ingredients:
                        raise RecipeDataError(
                            f"Recipe {recipe_id!r} must include "
                            "at least one ingredient."
                        )
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
                            f"Recipe {recipe_id!r} has a negative "
                            "or invalid numeric value."
                        ) from exc
                    recipes.append(
                        {
                            "id": recipe_id,
                            "name": name,
                            "ingredients": ingredients,
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

    @staticmethod
    def _optional_nonnegative_float(value: str | None) -> float | None:
        if value is None or not value.strip():
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Values cannot be negative.")
        return number

    def match(self, available_ingredients: list[str]) -> list[dict[str, Any]]:
        """Return recipes sorted by descending coverage and deterministic name."""
        available = set(normalize_ingredients(available_ingredients))
        ranked: list[dict[str, Any]] = []
        for recipe in self.recipes:
            recipe_ingredients = set(recipe["ingredients"])
            matched = sorted(available & recipe_ingredients)
            missing = sorted(recipe_ingredients - available)
            ranked.append(
                {
                    **recipe,
                    "matched_ingredients": matched,
                    "missing_ingredients": missing,
                    "match_score": len(matched) / len(recipe_ingredients),
                }
            )
        return sorted(
            ranked,
            key=lambda recipe: (-recipe["match_score"], recipe["name"].casefold()),
        )
