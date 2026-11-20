"""Explore Recipe Catalog — browse all recipes by category or keyword."""

from __future__ import annotations

import streamlit as st

from src.backend import (
    apply_custom_styles,
    init_session_state,
    load_recipe_matcher,
    render_catalog_card,
)

# Constant list defined once at module level (not rebuilt every render)
_CATEGORIES = [
    "All", "Italian", "Mexican", "Asian", "American", "Mediterranean",
    "French", "Indian", "Thai", "Quick Meals", "Breakfast", "High Protein", "Low Calorie",
]


def show() -> None:
    apply_custom_styles()
    init_session_state()
    matcher = load_recipe_matcher()
    servings: int = int(st.session_state["servings"])

    # ── Page header ───────────────────────────────────────────────────────
    st.subheader("Explore Recipe Catalog")
    st.caption("Browse all dishes from the recipe database — filter by category or search by keyword.")

    # ── Search + category filter ──────────────────────────────────────────
    cat_search: str = st.text_input(
        "Search catalog",
        placeholder="e.g. pasta, italian, chicken, quick meals…",
        key="catalog_search_input",
    )

    selected_cat: str = st.radio(
        "Categories",
        _CATEGORIES,
        horizontal=True,
        key="catalog_category_pills",
    )

    # ── Fetch full recipe list (cached inside matcher) ────────────────────
    # matcher.match([]) returns every recipe sorted by name; cheap call since
    # the underlying data is already in memory via @st.cache_resource
    all_recipes = matcher.match([])

    # ── Category filter ───────────────────────────────────────────────────
    if selected_cat != "All":
        if selected_cat == "Quick Meals":
            all_recipes = [
                r for r in all_recipes
                if r.get("category") == "Quick Meals"
                or (r.get("prep_time") is not None and r["prep_time"] <= 20)
            ]
        elif selected_cat == "High Protein":
            all_recipes = [r for r in all_recipes if (r.get("protein_g") or 0) >= 25]
        elif selected_cat == "Low Calorie":
            all_recipes = [
                r for r in all_recipes
                if r.get("calories_per_serving") is not None and r["calories_per_serving"] <= 350
            ]
        else:
            cat_lower = selected_cat.lower()
            all_recipes = [
                r for r in all_recipes
                if r.get("category", "").lower() == cat_lower
            ]

    # ── Keyword filter ────────────────────────────────────────────────────
    if cat_search.strip():
        q = cat_search.strip().lower()
        all_recipes = [
            r for r in all_recipes
            if q in r["name"].lower()
            or q in r.get("category", "").lower()
            or any(q in ing.lower() for ing in r.get("ingredients", []))
        ]

    # ── Render results ────────────────────────────────────────────────────
    display = all_recipes[:15]
    if display:
        st.caption(f"Showing **{len(display)}** of **{len(all_recipes)}** recipe(s):")
        for rec in display:
            render_catalog_card(rec, servings)
    else:
        st.info("No recipes match your filter. Try a different category or search term.")


if __name__ == "__main__" or "st" in globals():
    show()
