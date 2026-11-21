
import csv
import json
from collections import Counter
from pathlib import Path
# checks for duplicate recipes, invalid recipes, recipe errors
# checks for structure and validity of recipes
# Find the recipe database from the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPE_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.csv"

REQUIRED_COLUMNS = [
    "id", "name", "ingredients", "calories_per_serving",
    "prep_time", "instructions", "protein_g", "carbs_g", "fat_g"
]

NUMERIC_COLUMNS = [
    "calories_per_serving", "prep_time",
    "protein_g", "carbs_g", "fat_g"
]


def validate_recipes():
    errors = []

    if not RECIPE_FILE.exists():
        print("ERROR: Recipe database was not found.")
        print(RECIPE_FILE)
        return False

    with RECIPE_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            print("ERROR: The CSV is empty.")
            return False

        if reader.fieldnames != REQUIRED_COLUMNS:
            print("ERROR: CSV columns do not match the expected format.")
            print("Expected:", REQUIRED_COLUMNS)
            print("Found:", reader.fieldnames)
            return False

        recipes = list(reader)

    ids = []
    names = []

    for row_number, recipe in enumerate(recipes, start=2):
        recipe_id = (recipe.get("id") or "").strip()
        name = (recipe.get("name") or "").strip()

        if None in recipe or any(value is None for value in recipe.values()):
            errors.append(f"Row {row_number}: Incorrect CSV column count.")

        if not recipe_id:
            errors.append(f"Row {row_number}: Missing recipe ID.")
        else:
            ids.append(recipe_id)

        if not name:
            errors.append(f"Row {row_number}: Missing recipe name.")
        else:
            names.append(name.lower())

        for column in ["ingredients", "instructions"]:
            try:
                items = json.loads(recipe[column] or "")

                if (
                    not isinstance(items, list)
                    or not items
                    or not all(
                        isinstance(item, str) and item.strip()
                        for item in items
                    )
                ):
                    errors.append(
                        f"Row {row_number}: {column} must be a nonempty list of text."
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"Row {row_number}: Invalid JSON in {column}."
                )

        for column in NUMERIC_COLUMNS:
            try:
                value = float(recipe[column])

                if not (0 <= value < float("inf")):
                    errors.append(
                        f"Row {row_number}: Invalid number in {column}."
                    )

                if column in ["calories_per_serving", "prep_time"] and value == 0:
                    errors.append(
                        f"Row {row_number}: {column} must be greater than zero."
                    )

            except (ValueError, TypeError):
                errors.append(
                    f"Row {row_number}: {column} must contain a number."
                )

    duplicate_ids = [
        item for item, count in Counter(ids).items() if count > 1
    ]
    duplicate_names = [
        item for item, count in Counter(names).items() if count > 1
    ]

    for item in duplicate_ids:
        errors.append(f"Duplicate recipe ID: {item}")

    for item in duplicate_names:
        errors.append(f"Duplicate recipe name: {item}")

    print("\nRECIPE DATABASE VALIDATION")
    print("--------------------------")
    print(f"Total recipes: {len(recipes)}")
    print(f"Duplicate IDs: {len(duplicate_ids)}")
    print(f"Duplicate names: {len(duplicate_names)}")
    print(f"Validation errors: {len(errors)}")

    if errors:
        print("\nISSUES FOUND:")
        for error in errors:
            print("-", error)
        print("\nVALIDATION FAILED")
        return False

    print("\nVALIDATION PASSED!")
    print("Recipe database structure looks good.")
    return True


if __name__ == "__main__":
    success = validate_recipes()
    raise SystemExit(0 if success else 1)
