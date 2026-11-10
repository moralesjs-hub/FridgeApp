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
        "id", "name", "ingredients", "calories_per_serving", "prep_time", "instructions"
    }

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.recipes = self._load_recipes()

    @staticmethod
    def _parse_list(value: str, field: str, recipe_id: str) -> list[str]:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise RecipeDataError(f"Recipe {recipe_id!r} has invalid {field}; expected a list.") from exc
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise RecipeDataError(f"Recipe {recipe_id!r} has invalid {field}; expected a list of strings.")
        return parsed

    @staticmethod
    def _optional_nonnegative_float(value: str | None) -> float | None:
        if value is None or not value.strip():
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
                missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
                if missing:
                    raise RecipeDataError(f"Recipe catalog is missing columns: {', '.join(sorted(missing))}.")

                recipes, seen_ids = [], set()
                for row_num, row in enumerate(reader, start=2):
                    rid, name = (row.get("id") or "").strip(), (row.get("name") or "").strip()
                    if not rid or not name:
                        raise RecipeDataError(f"Recipe catalog row {row_num} needs both an id and name.")
                    if rid in seen_ids:
                        raise RecipeDataError(f"Duplicate recipe id: {rid}.")

                    ingredients = normalize_ingredients(
                        self._parse_list(row.get("ingredients") or "", "ingredients", rid)
                    )
                    instructions = self._parse_list(row.get("instructions") or "", "instructions", rid)
                    if not ingredients:
                        raise RecipeDataError(f"Recipe {rid!r} must include at least one ingredient.")

                    try:
                        cal = self._optional_nonnegative_float(row.get("calories_per_serving"))
                        prep = self._optional_nonnegative_float(row.get("prep_time"))
                        macros = {
                            k: self._optional_nonnegative_float(row.get(k))
                            for k in ("protein_g", "carbs_g", "fat_g")
                        }
                    except ValueError as exc:
                        raise RecipeDataError(f"Recipe {rid!r} has a negative or invalid numeric value.") from exc

                    recipes.append({
                        "id": rid, "name": name, "ingredients": ingredients,
                        "calories_per_serving": cal, "prep_time": prep,
                        "instructions": instructions, **macros,
                    })
                    seen_ids.add(rid)
        except (OSError, csv.Error) as exc:
            raise RecipeDataError(f"Could not read recipe catalog: {exc}") from exc

        if not recipes:
            raise RecipeDataError("The recipe catalog contains no recipes.")
        return recipes

    def match(self, available_ingredients: list[str]) -> list[dict[str, Any]]:
        """Return recipes sorted by descending coverage and deterministic name."""
        available = set(normalize_ingredients(available_ingredients))
        ranked = []
        for r in self.recipes:
            r_ings = set(r["ingredients"])
            matched, missing = sorted(available & r_ings), sorted(r_ings - available)
            ranked.append({
                **r, "matched_ingredients": matched,
                "missing_ingredients": missing,
                "match_score": len(matched) / len(r_ings),
            })
        return sorted(ranked, key=lambda r: (-r["match_score"], r["name"].casefold()))
