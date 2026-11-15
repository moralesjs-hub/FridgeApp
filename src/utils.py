"""Shared ingredient normalization, culinary taxonomy, and text helpers."""

from __future__ import annotations

import re
from typing import Iterable

# Canonical pantry staples that do not unfairly penalize recipe scores if omitted
PANTRY_STAPLES = {
    "salt",
    "black pepper",
    "pepper",
    "olive oil",
    "vegetable oil",
    "cooking oil",
    "oil",
    "water",
    "butter",
    "flour",
    "sugar",
    "brown sugar",
    "baking powder",
    "baking soda",
    "vinegar",
}

# Recognized culinary substitutions for flexible recipe matching
CULINARY_SUBSTITUTIONS: dict[str, list[str]] = {
    "lemon": ["lime", "lemon juice", "white wine vinegar"],
    "lime": ["lemon", "lime juice"],
    "onion": ["shallot", "green onion", "scallion", "red onion", "yellow onion", "white onion"],
    "shallot": ["onion", "scallion"],
    "green onion": ["scallion", "chives", "onion"],
    "scallion": ["green onion", "chives", "onion"],
    "garlic": ["garlic powder", "shallot"],
    "butter": ["olive oil", "oil", "margarine", "ghee"],
    "olive oil": ["vegetable oil", "oil", "canola oil", "butter"],
    "vegetable oil": ["olive oil", "canola oil", "oil"],
    "cheese": ["cheddar", "mozzarella", "parmesan", "feta"],
    "cheddar": ["cheese", "monterey jack", "colby"],
    "mozzarella": ["cheese", "provolone"],
    "parmesan": ["cheese", "pecorino romano", "asiago"],
    "yogurt": ["greek yogurt", "sour cream", "plain yogurt"],
    "sour cream": ["yogurt", "greek yogurt"],
    "milk": ["almond milk", "oat milk", "soy milk", "heavy cream"],
    "chicken": ["turkey", "chicken breast", "chicken thigh"],
    "beef": ["ground beef", "steak", "pork"],
    "pork": ["bacon", "pork chop", "ham"],
    "salmon": ["tuna", "trout", "whitefish"],
    "honey": ["maple syrup", "agave", "sugar"],
    "soy sauce": ["tamari", "coconut aminos"],
    "pasta": ["noodles", "spaghetti", "penne"],
    "rice": ["brown rice", "white rice", "jasmine rice", "quinoa"],
    "spinach": ["kale", "arugula", "chard"],
    "parsley": ["cilantro", "basil", "spinach", "chives"],
    "chickpea": ["garbanzo bean", "white bean", "cannellini bean"],
}

_ALIASES = {
    # Alliums & Aromatics
    "onion": "onion",
    "onions": "onion",
    "yellow onion": "onion",
    "white onion": "onion",
    "red onion": "onion",
    "shallot": "onion",
    "shallots": "onion",
    "green onion": "onion",
    "green onions": "onion",
    "scallion": "onion",
    "scallions": "onion",
    "spring onion": "onion",
    "spring onions": "onion",
    "garlic": "garlic",
    "garlic clove": "garlic",
    "garlic cloves": "garlic",
    "minced garlic": "garlic",
    "ginger": "ginger",
    "fresh ginger": "ginger",
    # Nightshades & Peppers
    "tomato": "tomato",
    "tomatoes": "tomato",
    "roma tomato": "tomato",
    "cherry tomato": "tomato",
    "cherry tomatoes": "tomato",
    "crushed tomato": "tomato",
    "crushed tomatoes": "tomato",
    "canned tomato": "tomato",
    "canned tomatoes": "tomato",
    "bell pepper": "bell pepper",
    "bell peppers": "bell pepper",
    "capsicum": "bell pepper",
    "pepper": "bell pepper",
    "red bell pepper": "bell pepper",
    "green bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    "jalapeno": "jalapeno",
    "jalapenos": "jalapeno",
    "chili pepper": "chili pepper",
    "eggplant": "eggplant",
    "aubergine": "eggplant",
    # Leafy Greens & Brassicas
    "spinach": "spinach",
    "spinach leaves": "spinach",
    "baby spinach": "spinach",
    "broccoli": "broccoli",
    "broccoli floret": "broccoli",
    "broccoli florets": "broccoli",
    "cauliflower": "cauliflower",
    "cabbage": "cabbage",
    "head cabbage": "cabbage",
    "red cabbage": "cabbage",
    "kale": "kale",
    "lettuce": "lettuce",
    "romaine": "lettuce",
    "romaine lettuce": "lettuce",
    "arugula": "arugula",
    # Squash, Root Vegetables & Others
    "carrot": "carrot",
    "carrots": "carrot",
    "potato": "potato",
    "potatoes": "potato",
    "russet potato": "potato",
    "russet potatoes": "potato",
    "sweet potato": "sweet potato",
    "sweet potatoes": "sweet potato",
    "cucumber": "cucumber",
    "cucumbers": "cucumber",
    "zucchini": "zucchini",
    "courgette": "zucchini",
    "squash": "squash",
    "butternut squash": "squash",
    "spaghetti squash": "squash",
    "acorn squash": "squash",
    "mushroom": "mushroom",
    "mushrooms": "mushroom",
    "button mushroom": "mushroom",
    "cremini mushroom": "mushroom",
    "celery": "celery",
    "asparagus": "asparagus",
    "corn": "corn",
    "green bean": "green bean",
    "green beans": "green bean",
    "peas": "pea",
    "pea": "pea",
    # Fruits
    "apple": "apple",
    "apples": "apple",
    "granny smith": "apple",
    "banana": "banana",
    "bananas": "banana",
    "orange": "orange",
    "oranges": "orange",
    "lemon": "lemon",
    "lemons": "lemon",
    "lemon juice": "lemon",
    "lime": "lime",
    "limes": "lime",
    "lime juice": "lime",
    "strawberry": "strawberry",
    "strawberries": "strawberry",
    "blueberry": "blueberry",
    "blueberries": "blueberry",
    "avocado": "avocado",
    "avocados": "avocado",
    "pineapple": "pineapple",
    # Dairy, Eggs & Alternatives
    "egg": "egg",
    "eggs": "egg",
    "large egg": "egg",
    "large eggs": "egg",
    "cheese": "cheese",
    "cheddar": "cheese",
    "cheddar cheese": "cheese",
    "mozzarella": "cheese",
    "mozzarella cheese": "cheese",
    "parmesan": "cheese",
    "parmesan cheese": "cheese",
    "feta": "cheese",
    "feta cheese": "cheese",
    "milk": "milk",
    "whole milk": "milk",
    "skim milk": "milk",
    "almond milk": "milk",
    "oat milk": "milk",
    "heavy cream": "cream",
    "cream": "cream",
    "yogurt": "yogurt",
    "plain yogurt": "yogurt",
    "greek yogurt": "yogurt",
    "butter": "butter",
    "unsalted butter": "butter",
    "salted butter": "butter",
    # Proteins & Seafood
    "chicken": "chicken",
    "chicken breast": "chicken",
    "chicken breasts": "chicken",
    "chicken thigh": "chicken",
    "chicken thighs": "chicken",
    "rotisserie chicken": "chicken",
    "beef": "beef",
    "ground beef": "beef",
    "steak": "beef",
    "pork": "pork",
    "pork chop": "pork",
    "pork chops": "pork",
    "bacon": "bacon",
    "turkey": "turkey",
    "ground turkey": "turkey",
    "salmon": "salmon",
    "salmon fillet": "salmon",
    "salmon fillets": "salmon",
    "tuna": "tuna",
    "canned tuna": "tuna",
    "shrimp": "shrimp",
    "tofu": "tofu",
    # Legumes & Grains
    "chickpea": "chickpea",
    "chickpeas": "chickpea",
    "garbanzo bean": "chickpea",
    "garbanzo beans": "chickpea",
    "black bean": "black bean",
    "black beans": "black bean",
    "kidney bean": "kidney bean",
    "kidney beans": "kidney bean",
    "white bean": "white bean",
    "white beans": "white bean",
    "lentil": "lentil",
    "lentils": "lentil",
    "pasta": "pasta",
    "spaghetti": "pasta",
    "penne": "pasta",
    "noodle": "pasta",
    "noodles": "pasta",
    "macaroni": "pasta",
    "rice": "rice",
    "white rice": "rice",
    "brown rice": "rice",
    "jasmine rice": "rice",
    "basmati rice": "rice",
    "quinoa": "quinoa",
    "oat": "oats",
    "oats": "oats",
    "rolled oats": "oats",
    "bread": "bread",
    "tortilla": "tortilla",
    "tortillas": "tortilla",
    # Pantry, Oils & Seasonings
    "all-purpose flour": "flour",
    "flour": "flour",
    "sugar": "sugar",
    "granulated sugar": "sugar",
    "brown sugar": "brown sugar",
    "honey": "honey",
    "maple syrup": "maple syrup",
    "olive oil": "olive oil",
    "extra-virgin olive oil": "olive oil",
    "vegetable oil": "oil",
    "canola oil": "oil",
    "cooking oil": "oil",
    "oil": "oil",
    "soy sauce": "soy sauce",
    "salt": "salt",
    "black pepper": "black pepper",
    "water": "water",
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
    "raw",
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
    "ground",
    "crushed",
    "canned",
    "frozen",
    "dried",
}

# Explicit non-food or non-culinary ImageNet terms that must NOT be mapped to ingredients
_NON_FOOD_EXCLUSIONS = {
    "sea cucumber",
    "slug",
    "snail",
    "worm",
    "spider",
    "crab",
    "tick",
    "frog",
    "toad",
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


def is_pantry_staple(ingredient: str) -> bool:
    """Check if normalized ingredient is a ubiquitous pantry staple."""
    return ingredient in PANTRY_STAPLES


def get_substitutes(ingredient: str) -> list[str]:
    """Return recognized culinary alternatives for an ingredient."""
    norm = normalize_ingredient(ingredient)
    return CULINARY_SUBSTITUTIONS.get(norm, [])


def ingredient_from_label(label: str) -> str | None:
    """Map a supported ImageNet/COCO class label to one canonical ingredient."""
    cleaned = clean_text(label)
    if not cleaned:
        return None

    # Check explicit non-food exclusions first (e.g. 'sea cucumber' is an echinoderm animal, not food)
    if cleaned in _NON_FOOD_EXCLUSIONS:
        return None

    # Supported direct canonical ingredients
    supported = {
        "apple", "avocado", "banana", "bell pepper", "broccoli",
        "carrot", "cauliflower", "cabbage", "cheese", "chicken",
        "cucumber", "egg", "lemon", "lime", "lettuce", "milk",
        "mushroom", "onion", "orange", "pear", "potato",
        "spinach", "strawberry", "tomato", "yogurt",
        "bread", "beef", "pork", "sausage", "pasta", "rice",
        "squash", "zucchini", "pineapple", "corn", "garlic",
        "beans", "parsley", "pomegranate", "fig", "artichoke"
    }

    direct = normalize_ingredient(cleaned)
    if direct in supported:
        return direct

    # Multi-token token matching for compound ImageNet labels
    label_words = set(cleaned.split())
    mappings: tuple[tuple[set[str], str], ...] = (
        ({"bell", "pepper"}, "bell pepper"),
        ({"sweet", "pepper"}, "bell pepper"),
        ({"granny", "smith"}, "apple"),
        ({"french", "loaf"}, "bread"),
        ({"bagel"}, "bread"),
        ({"beigel"}, "bread"),
        ({"pretzel"}, "bread"),
        ({"head", "cabbage"}, "cabbage"),
        ({"zucchini"}, "zucchini"),
        ({"courgette"}, "zucchini"),
        ({"butternut", "squash"}, "squash"),
        ({"spaghetti", "squash"}, "squash"),
        ({"acorn", "squash"}, "squash"),
        ({"mashed", "potato"}, "potato"),
        ({"cheeseburger"}, "cheese"),
        ({"pizza"}, "cheese"),
        ({"rotisserie"}, "chicken"),
        ({"potpie"}, "chicken"),
        ({"hen"}, "chicken"),
        ({"meat", "loaf"}, "beef"),
        ({"meatloaf"}, "beef"),
        ({"hotdog"}, "sausage"),
        ({"hot", "dog"}, "sausage"),
        ({"carbonara"}, "pasta"),
        ({"guacamole"}, "avocado"),
        ({"carton"}, "milk"),
        ({"eggnog"}, "egg"),
        ({"custard", "apple"}, "apple"),
        ({"pineapple"}, "pineapple"),
        ({"ananas"}, "pineapple"),
        ({"pomegranate"}, "pomegranate"),
        ({"fig"}, "fig"),
    )
    for tokens, ingredient in mappings:
        if tokens.issubset(label_words):
            return ingredient

    return None
