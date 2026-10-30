"""Shared ingredient normalization and text helpers."""

from __future__ import annotations

import re
from typing import Iterable


_ALIASES = {
    "bell pepper": "bell pepper",
    "capsicum": "bell pepper",
    "pepper": "bell pepper",
    "tomato": "tomato",
    "tomatoes": "tomato",
    "egg": "egg",
    "eggs": "egg",
    "cheddar": "cheese",
    "cheddar cheese": "cheese",
    "mozzarella": "cheese",
    "parmesan": "cheese",
    "green onion": "onion",
    "scallion": "onion",
    "spring onion": "onion",
    "chickpeas": "chickpea",
    "garbanzo beans": "chickpea",
    "garbanzo bean": "chickpea",
    "black beans": "black bean",
    "kidney beans": "kidney bean",
    "white beans": "white bean",
    "apples": "apple",
    "granny smith": "apple",
    "potatoes": "potato",
    "sweet potatoes": "sweet potato",
    "carrots": "carrot",
    "onions": "onion",
    "spinach leaves": "spinach",
    "chicken breast": "chicken",
    "chicken thighs": "chicken",
    "chicken thigh": "chicken",
    "salmon fillet": "salmon",
    "greek yogurt": "yogurt",
    "plain yogurt": "yogurt",
    "all-purpose flour": "flour",
    "olive oil": "olive oil",
    "vegetable oil": "oil",
    "cooking oil": "oil",
}

_PREPARATION_WORDS = {
    "fresh",
    "chopped",
    "diced",
    "sliced",
    "minced",
    "grated",
    "shredded",
    "cooked",
    "uncooked",
    "large",
    "small",
    "medium",
    "ripe",
    "boneless",
    "skinless",
    "optional",
    "divided",
    "to",
    "taste",
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
    words = [word for word in cleaned.split() if word not in _PREPARATION_WORDS]
    normalized = " ".join(words)
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("s") and not normalized.endswith(("ss", "us")):
        normalized = normalized[:-1]
    return _ALIASES.get(normalized, normalized)


def normalize_ingredients(values: Iterable[str]) -> list[str]:
    """Normalize an iterable and preserve first-seen order without duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        ingredient = normalize_ingredient(value)
        if ingredient and ingredient not in seen:
            normalized.append(ingredient)
            seen.add(ingredient)
    return normalized


def ingredient_from_label(label: str) -> str | None:
    """Map a supported ImageNet class label to one canonical ingredient."""
    cleaned = clean_text(label)
    direct = normalize_ingredient(cleaned)
    supported = {
        "apple", "avocado", "banana", "bell pepper", "broccoli", "carrot",
        "cauliflower", "cheese", "chicken", "cucumber", "egg", "lemon",
        "lettuce", "milk", "mushroom", "onion", "orange", "pear", "potato",
        "spinach", "strawberry", "tomato", "yogurt",
    }
    if direct in supported:
        return direct
    label_words = set(cleaned.split())
    mappings = (
        ({"bell", "pepper"}, "bell pepper"),
        ({"sweet", "pepper"}, "bell pepper"),
        ({"hen"}, "chicken"),
    )
    for tokens, ingredient in mappings:
        if tokens.issubset(label_words):
            return ingredient
    return None
