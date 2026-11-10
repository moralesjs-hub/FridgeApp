"""Shared ingredient normalization and text helpers."""

from __future__ import annotations

import re
from typing import Iterable

_ALIASES = {
    "capsicum": "bell pepper", "pepper": "bell pepper", "bell pepper": "bell pepper",
    "tomatoes": "tomato", "tomato": "tomato", "eggs": "egg", "egg": "egg",
    "cheddar": "cheese", "cheddar cheese": "cheese", "mozzarella": "cheese", "parmesan": "cheese",
    "green onion": "onion", "scallion": "onion", "spring onion": "onion", "onions": "onion",
    "chickpeas": "chickpea", "garbanzo beans": "chickpea", "garbanzo bean": "chickpea",
    "black beans": "black bean", "kidney beans": "kidney bean", "white beans": "white bean",
    "apples": "apple", "granny smith": "apple", "carrots": "carrot",
    "potatoes": "potato", "sweet potatoes": "sweet potato",
    "spinach leaves": "spinach", "salmon fillet": "salmon",
    "chicken breast": "chicken", "chicken thighs": "chicken", "chicken thigh": "chicken",
    "greek yogurt": "yogurt", "plain yogurt": "yogurt", "all-purpose flour": "flour",
    "olive oil": "olive oil", "vegetable oil": "oil", "cooking oil": "oil",
}

_PREP_WORDS = {
    "fresh", "chopped", "diced", "sliced", "minced", "grated", "shredded",
    "cooked", "uncooked", "large", "small", "medium", "ripe", "boneless",
    "skinless", "optional", "divided", "to", "taste",
}

SUPPORTED_FOODS = {
    "apple", "avocado", "banana", "bell pepper", "broccoli", "carrot",
    "cauliflower", "cheese", "chicken", "cucumber", "egg", "lemon",
    "lettuce", "milk", "mushroom", "onion", "orange", "pear", "potato",
    "spinach", "strawberry", "tomato", "yogurt",
}


def clean_text(value: str) -> str:
    """Return lowercase text with punctuation and repeated whitespace removed."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", value.lower())).strip()


def normalize_ingredient(value: str) -> str:
    """Normalize common ingredient aliases and remove preparation descriptors."""
    cleaned = clean_text(value)
    if not cleaned:
        return ""
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    words = [w for w in cleaned.split() if w not in _PREP_WORDS]
    norm = " ".join(words)
    if norm in _ALIASES:
        return _ALIASES[norm]
    if norm.endswith("ies") and len(norm) > 4:
        norm = norm[:-3] + "y"
    elif norm.endswith("s") and not norm.endswith(("ss", "us")):
        norm = norm[:-1]
    return _ALIASES.get(norm, norm)


def normalize_ingredients(values: Iterable[str]) -> list[str]:
    """Normalize an iterable and preserve first-seen order without duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        ing = normalize_ingredient(value)
        if ing and ing not in seen:
            normalized.append(ing)
            seen.add(ing)
    return normalized


def ingredient_from_label(label: str) -> str | None:
    """Map a supported ImageNet class label to one canonical ingredient."""
    cleaned = clean_text(label)
    direct = normalize_ingredient(cleaned)
    if direct in SUPPORTED_FOODS:
        return direct
    words = set(cleaned.split())
    mappings = (
        ({"bell", "pepper"}, "bell pepper"),
        ({"sweet", "pepper"}, "bell pepper"),
        ({"hen"}, "chicken"),
    )
    for tokens, ingredient in mappings:
        if tokens.issubset(words):
            return ingredient
    return None
