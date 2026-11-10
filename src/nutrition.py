"""Serving-aware estimates for the recipe catalog's nutrition values."""

from __future__ import annotations

import math
from typing import Any

DAILY_BENCHMARKS = {"calories": 2000.0, "protein_g": 50.0, "carbs_g": 275.0, "fat_g": 78.0}


def estimate_nutrition(recipe: dict[str, Any], servings: int = 1) -> dict[str, float | None]:
    """Scale per-serving nutrition by a positive number of servings."""
    if isinstance(servings, bool) or not isinstance(servings, int) or servings < 1:
        raise ValueError("Servings must be a positive whole number.")
    result: dict[str, float | None] = {}
    for key in ("calories_per_serving", "protein_g", "carbs_g", "fat_g"):
        val = recipe.get(key)
        if val is None:
            result[key] = None
        else:
            amt = float(val)
            if not math.isfinite(amt) or amt < 0:
                raise ValueError(f"{key} must be a finite, non-negative number.")
            result[key] = amt * servings
    return result


def benchmark_percent(value: float | None, benchmark: float) -> float | None:
    """Return a percentage of a positive reference value, or None if unavailable."""
    if value is None:
        return None
    if benchmark <= 0:
        raise ValueError("benchmark must be positive.")
    return max(0.0, value) / benchmark * 100.0
