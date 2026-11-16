"""Find Recipes view — photo upload, AI detection, manual pantry additions, and ranked results."""

from __future__ import annotations

from hashlib import sha256

import streamlit as st

from src.backend import (
    _ANIMAL_SET,
    _COMMON_INGREDIENTS_SET,
    _MEAT_SET,
    SAMPLE_IMAGES,
    apply_custom_styles,
    detect_uploaded_image,
    init_session_state,
    load_recipe_matcher,
    read_image,
    render_recipe,
)
from src.utils import normalize_ingredients


def show() -> None:
    apply_custom_styles()
    init_session_state()
    matcher = load_recipe_matcher()

    # ── Read settings once from session_state ─────────────────────────────
    confidence: float = st.session_state["confidence"]
    servings:   int   = int(st.session_state["servings"])
    max_results: int  = int(st.session_state["max_results"])
    min_match:   int  = int(st.session_state["min_match"])

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: Image input (left) + Ingredient review (right)
    # ══════════════════════════════════════════════════════════════════════
    col_left, col_right = st.columns([0.55, 0.45], gap="large")

    # ── Left column: image source ─────────────────────────────────────────
    with col_left:
        with st.container():
            st.subheader("1. Add a fridge photo")

            sample_choice: str = st.selectbox(
                "Try an included sample",
                ["Choose a sample…", *SAMPLE_IMAGES.keys()],
            )

            def _save_uploaded_image() -> None:
                """Persist file-uploader bytes to session_state on change."""
                photo = st.session_state["fridge_photo"]
                if photo is not None:
                    st.session_state["uploaded_bytes"] = photo.getvalue()
                    st.session_state["uploaded_name"]  = photo.name
                else:
                    st.session_state["uploaded_bytes"] = None
                    st.session_state["uploaded_name"]  = None
                st.session_state["_ingredient_source"] = None

            st.file_uploader(
                "Upload image (JPG / PNG)",
                type=["jpg", "jpeg", "png"],
                key="fridge_photo",
                on_change=_save_uploaded_image,
            )

            # ── Resolve the active image source ──────────────────────────
            image           = None
            annotated_image = None
            detections: list[tuple[str, float]] = []
            is_demo         = False
            inference_failed= False
            image_source    = "none"
            image_digest    = ""

            uploaded_bytes: bytes | None = st.session_state.get("uploaded_bytes")
            if uploaded_bytes is not None:
                image_digest = sha256(uploaded_bytes).hexdigest()
                image_source = f"upload:{st.session_state['uploaded_name']}:{image_digest}"
                try:
                    image = read_image(uploaded_bytes)
                except ValueError as exc:
                    st.error(str(exc))

            elif sample_choice != "Choose a sample…":
                image_source = f"sample:{sample_choice}"
                sample = SAMPLE_IMAGES[sample_choice]
                try:
                    image    = read_image(sample["path"])
                    detections = sample["ingredients"]
                    is_demo  = True
                except ValueError as exc:
                    st.error(f"Could not open sample: {exc}")

            # ── Render image + run CV pipeline ────────────────────────────
            if image is not None:
                if is_demo:
                    st.image(image, caption="Sample view", use_container_width=True)
                    st.info(
                        "Offline demo — ingredients are pre-seeded for a predictable walkthrough; "
                        "these are not model predictions."
                    )
                else:
                    with st.spinner("Processing computer vision pipeline…"):
                        try:
                            detections, annotated_bytes = detect_uploaded_image(
                                image_digest, confidence, uploaded_bytes
                            )
                            if annotated_bytes:
                                annotated_image = read_image(annotated_bytes)
                        except RuntimeError as exc:
                            inference_failed = True
                            st.warning(str(exc))
                            st.info("Try an included sample or add ingredients manually below.")

                    display_img = annotated_image if annotated_image is not None else image
                    caption = (
                        "Detected items with bounding boxes"
                        if annotated_image is not None
                        else "Uploaded image"
                    )
                    st.image(display_img, caption=caption, use_container_width=True)
                    if not detections and not inference_failed:
                        st.info(
                            "No supported ingredients were recognized at this confidence threshold. "
                            "Add ingredients manually or try an included sample."
                        )

    # ── Right column: ingredient review panel ─────────────────────────────
    with col_right:
        with st.container():
            st.subheader("2. Review your ingredients")
            detected_names: list[str] = [name for name, _ in detections]

            # Sync saved_ingredients when image source changes
            if st.session_state.get("_ingredient_source") != image_source:
                st.session_state["saved_ingredients"] = detected_names
                st.session_state["_ingredient_source"] = image_source

            tab_detected, tab_manual = st.tabs(["🔍 Detected Items", "✏️ Manual Additions"])

            with tab_detected:
                st.caption("Review AI-detected items and common pantry ingredients.")

                # Performance: set-union is O(n) with frozenset; sorted() is O(n log n) but
                # only called once per render since the set is frozen at module level.
                options = sorted(
                    _COMMON_INGREDIENTS_SET
                    | set(detected_names)
                    | set(st.session_state["saved_ingredients"])
                )
                selected_ingredients: list[str] = st.multiselect(
                    "Detected and pantry ingredients",
                    options=options,
                    default=st.session_state["saved_ingredients"],
                    help="Remove false detections or add pantry staples.",
                )
                st.session_state["saved_ingredients"] = selected_ingredients

                if detections and not is_demo:
                    badges = " · ".join(
                        f"`{name}` ({score:.0%})" for name, score in detections
                    )
                    st.markdown(f"**Current detections:** {badges}")

            with tab_manual:
                st.caption("Add staple items or extras not detected in the photo.")
                extra_raw: str = st.text_input(
                    "Add another ingredient",
                    placeholder="e.g. rice, garlic, lemon",
                    help="Comma-separated ingredients to add to your inventory.",
                )
                extra_list = (
                    normalize_ingredients(extra_raw.split(",")) if extra_raw.strip() else []
                )
                c_btn, _ = st.columns([1, 2])
                with c_btn:
                    if st.button("Add to inventory", key="btn_add_manual_item", type="secondary"):
                        if extra_list:
                            existing = set(st.session_state["manual_ingredients"])
                            for item in extra_list:
                                if item not in existing:
                                    st.session_state["manual_ingredients"].append(item)
                                    existing.add(item)
                            st.rerun()

                if st.session_state["manual_ingredients"]:
                    st.markdown("**Added manual items:**")
                    for man_item in st.session_state["manual_ingredients"][:]:
                        mc1, mc2 = st.columns([4, 1])
                        mc1.write(f"• {man_item.capitalize()}")
                        if mc2.button("✕", key=f"rm_{man_item}", type="tertiary"):
                            st.session_state["manual_ingredients"].remove(man_item)
                            st.rerun()

            # Merged confirmed ingredient list
            confirmed_ingredients = normalize_ingredients(
                selected_ingredients + extra_list + st.session_state["manual_ingredients"]
            )

            st.divider()
            if not confirmed_ingredients:
                st.info("Add an image or type ingredients above to begin analysis.")
            else:
                st.success(
                    f"**{len(confirmed_ingredients)} ingredient(s):** "
                    + ", ".join(confirmed_ingredients)
                )

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: Recommended recipes
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("3. Recommended recipes")

    col_search, col_diet = st.columns([2, 1])
    with col_search:
        search_query: str = st.text_input(
            "Search recipes by name, category, or ingredient",
            placeholder="e.g. pasta, italian, chicken, salad…",
            key="workspace_search_query",
        )
    with col_diet:
        diet: str = st.radio(
            "Dietary Filter",
            ["All recipes", "Vegetarian", "Vegan", "High protein", "Low calorie"],
            horizontal=True,
            key="workspace_diet_filter",
        )

    if not confirmed_ingredients:
        st.info("Your recipe matches will appear here once you add ingredients.")
        return

    # ── Match & filter ────────────────────────────────────────────────────
    ranked = matcher.match(confirmed_ingredients)

    if search_query.strip():
        q = search_query.strip().lower()
        ranked = [
            r for r in ranked
            if q in r["name"].lower()
            or q in r.get("category", "").lower()
            or any(q in ing.lower() for ing in r.get("matched_ingredients", []) + r.get("missing_ingredients", []))
        ]

    # Dietary filter — use frozen sets for O(1) membership tests
    if diet == "Vegetarian":
        ranked = [
            r for r in ranked
            if not _MEAT_SET.intersection(r["matched_ingredients"] + r["missing_ingredients"])
        ]
    elif diet == "Vegan":
        ranked = [
            r for r in ranked
            if not _ANIMAL_SET.intersection(r["matched_ingredients"] + r["missing_ingredients"])
        ]
    elif diet == "High protein":
        ranked = [r for r in ranked if (r.get("protein_g") or 0) >= 20]
    elif diet == "Low calorie":
        ranked = [
            r for r in ranked
            if r["calories_per_serving"] is not None and r["calories_per_serving"] <= 300
        ]

    qualifying = [r for r in ranked if r["match_score"] * 100 >= min_match][:max_results]
    if not qualifying:
        st.info(
            "No recipes meet that match threshold. "
            "Lower the minimum match % in Settings or add more ingredients."
        )
        return

    st.caption(
        f"Ranked by ingredient coverage · **{len(qualifying)}** recipe(s) shown · "
        "scores don't account for quantities or substitutions."
    )

    ready_recipes   = [r for r in qualifying if not r["missing_ingredients"]]
    missing_recipes = [r for r in qualifying if     r["missing_ingredients"]]

    st.markdown("### ✅ Ready to cook")
    if ready_recipes:
        for recipe in ready_recipes:
            render_recipe(recipe, servings, context="ready")
    else:
        st.info("No complete matches yet — try adding more ingredients.")

    if missing_recipes:
        st.markdown("### 🛒 Almost ready to cook")
        for recipe in missing_recipes:
            render_recipe(recipe, servings, context="almost")


if __name__ == "__main__" or "st" in globals():
    show()
